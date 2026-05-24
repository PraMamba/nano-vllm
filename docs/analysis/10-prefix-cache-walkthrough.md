# nano-vllm 源码走读：Prefix Cache 实现解析

> 在前面的文章中，我们已经走读了 nano-vllm 的 PagedAttention 和 BlockManager 的基础设计。Block 是 KV 缓存的最小管理单元，BlockManager 负责分配和回收。但到目前为止，我们讨论的 Block 都是"用完即扔"的——一个 Sequence 完成推理后，它占用的 Block 全部释放回空闲池，下一个 Sequence 即使有完全相同的 prompt，也要从头做一遍 prefill 计算。
>
> 这在多轮对话、shared system prompt、few-shot 等场景下是巨大的浪费。Prefix Cache 的核心目标就是：**当多个请求共享相同的 token 前缀时，让它们复用已经算好的 KV 缓存 Block，跳过重复的 prefill 计算。**
>
> 本文不展开 prefix caching 的理论设计（如 RadixAttention 等学术工作），而是聚焦 nano-vllm 源码，分析一个极简推理引擎如何在约 120 行代码中把 prefix caching 接入已有的 PagedAttention 调度链路，以及它带来了哪些收益和限制。


## 前言

### 业务背景

大模型推理的典型场景中，大量请求共享相同的前缀。最常见的例子是 system prompt：一个聊天服务可能有数千个并发请求，每个请求都以相同的 2000 token 系统指令开头，仅在用户消息部分有差异。如果每个请求都从头做 prefill，2000 token × 1000 个请求 = 200 万 token 的重复 QKV 投影、RoPE、Attention 计算。

### 核心矛盾

PagedAttention 的 Block 管理机制天然适合 prefix caching——Block 是固定大小的物理单元，内容独立于 Sequence 身份。但要实现跨请求的 Block 复用，需要解决三个工程问题：

1. **识别问题**：如何判断两个 Sequence 的某段前缀对应相同的 KV 内容？不能靠 Sequence ID（不同请求），必须靠内容本身。
2. **生命周期问题**：一个 Sequence 用完释放了 Block，但另一个 Sequence 可能稍后还需要相同的 Block。Block 不能真的"删除"，需要保留数据以备复用。
3. **调度衔接问题**：缓存命中后，prefill 阶段只需要计算后缀部分的 token，但 Attention 仍需要看到完整的 KV 序列（含已缓存部分）。Q 长度和 K 长度不再相等，ModelRunner 和 Attention 层都需要感知这种不对称。

### 本文主线

本文围绕以上三个矛盾，分五章展开：

- **第一章**：内容寻址哈希——如何识别相同前缀
- **第二章**：Block 生命周期与惰性淘汰——如何让 Block "死而不删"
- **第三章**：调度层如何感知缓存命中并减少 prefill 工作量
- **第四章**：ModelRunner 与 Attention 层如何处理 Q/K 不对称
- **第五章**：完整主路径串联与性能分析

### 不展开的内容

不讲 RadixAttention 原理、不讲 vLLM 完整实现的 LRU Evictor / Prefix Tree / Copy-on-Write 等机制。只讲 nano-vllm 自身的 ~120 行实现。

### 核心文件表

| 文件 | 职责 |
|---|---|
| `nanovllm/engine/block_manager.py` | 核心：Block 数据结构、xxhash 链式哈希、分配/释放/复用、哈希注册 |
| `nanovllm/engine/scheduler.py` | 调度层：缓存命中检测、prefill token 数量缩减、hash_blocks 注册调用 |
| `nanovllm/engine/sequence.py` | 序列状态：`num_cached_tokens`、`block_table`、`block()` 方法 |
| `nanovllm/engine/model_runner.py` | 执行层：`prepare_prefill` 中跳过缓存 token、Q/K 不对称检测、block_tables 传递 |
| `nanovllm/layers/attention.py` | 注意力计算：prefix cache 模式下切换到 paged KV 读取 |
| `nanovllm/utils/context.py` | 隐式上下文传递：将 block_tables 从 ModelRunner 传到 Attention 层 |
| `nanovllm/config.py` | 配置：`kvcache_block_size`（默认 256，影响缓存粒度） |
| `nanovllm/engine/llm_engine.py` | 引擎主循环：完全不感知 prefix cache，关注点分离的体现 |


---

## 一、内容寻址哈希：如何识别相同的前缀

### 1.1 设计哲学与核心问题

要在不同的 Sequence 之间共享 KV 缓存 Block，第一个问题是：**怎么知道两个 Block 里存的 KV 值是一样的？**

最直觉的方案是按"Sequence ID + 位置"寻址——但这行不通，因为不同的请求有不同的 Sequence ID，即使它们的 prompt 完全相同。

nano-vllm 选择的方案是**内容寻址**：对每个 Block 内的 token 序列做 xxhash，相同内容 → 相同哈希 → 可以复用同一个物理 Block。

但简单的逐 Block 哈希有一个致命问题：transformer 的 attention 是因果的，同样的 256 个 token，如果它们前面的 token 不同，算出来的 KV 值也完全不同。所以 Block 的哈希必须包含它所有前置 Block 的信息。

nano-vllm 的解法是**链式哈希**（类似 Merkle 链）：Block N 的哈希 = xxh64(Block N-1 的哈希 ∥ Block N 的 token IDs)。

### 1.2 源码入口与关键对象

```
nanovllm/engine/block_manager.py
  - Block（行 8-23）：单个物理块的元数据，含 ref_count、hash、token_ids
  - BlockManager.compute_hash（行 35-41）：链式 xxhash 计算
  - BlockManager.hash_to_block_id（行 31）：哈希 → 物理块 ID 的映射表
```

### 1.3 主流程拆解

#### Block 数据结构

```python
# block_manager.py:8-23
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1          # -1 = 未哈希
        self.token_ids = []     # 存储实际 token，用于碰撞校验
```

每个 Block 除了物理 ID 和引用计数外，额外存储了两个 prefix cache 专用字段：
- `hash`：该 Block 的内容哈希，-1 表示尚未计算或已被重置
- `token_ids`：该 Block 内的实际 token ID 列表，用于哈希碰撞时的二次校验

`token_ids` 是纯粹为碰撞安全而存在的冗余数据。Sequence 对象本身已经持有完整的 token 序列。

#### 链式哈希计算

```python
# block_manager.py:35-41
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

这是整个 prefix cache 的数学基础。调用方式：

```
Block 0: h0 = xxh64(token_ids_0)
Block 1: h1 = xxh64(h0 || token_ids_1)
Block 2: h2 = xxh64(h1 || token_ids_2)
...
```

`prefix` 参数是前一个 Block 的哈希值。第一个 Block 传 `-1`，此时 `h.update(prefix.to_bytes(...))` 被跳过，只哈希 token IDs。后续 Block 先把前一个 Block 的哈希值（8 字节小端序）喂入 hasher，再喂入当前 Block 的 token ID 字节。

链式设计的意义：如果两个 Sequence 的前 3 个 Block token 完全相同，它们的 h0、h1、h2 也完全相同，可以共享这 3 个物理 Block。但如果 Block 0 有一个 token 不同，h0 就不同，h1 也不同（因为 h1 依赖 h0），整条链全部失效。这正是 transformer 因果 attention 的语义要求。

#### 哈希映射表

```python
# block_manager.py:31
self.hash_to_block_id: dict[int, int] = dict()
```

标准 Python 字典，O(1) 平均查找。键是 xxh64 产生的 64 位无符号整数，值是物理 Block ID。没有大小限制，没有 LRU 策略——条目只在物理 Block 被重新分配给不同内容时才被惰性删除。

### 1.4 关键细节与误区澄清

**误区一：哈希里包含位置或层信息？**

不包含。`compute_hash` 只接受 `token_ids` 和 `prefix`（前一个 Block 的哈希值）。没有 layer ID，没有 position offset。这是正确的，因为：
- KV 缓存的 Block 在物理上是跨层共享编号的（`kv_cache` 张量形状 `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`，Block ID 是第三维的索引，对所有层通用）
- 位置信息隐含在 token 序列的顺序中（通过链式哈希传递）

**误区二：xxh64 的碰撞安全够用吗？**

xxh64 产生 64 位摘要，空间 2^64 ≈ 1.8 × 10^19。根据生日悖论，在 100 万个不同 Block 的情况下碰撞概率约 2.7 × 10^-8，实际可忽略。但代码依然在 `can_allocate` 的第 66 行做了二次校验：

```python
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

即使哈希匹配，也要逐 token 比较内容。这是防御性编程。代价是每个 Block 额外存储 256 个 int（约 7KB 的 Python 对象内存）。

**误区三：`-1` 作为哨兵值安全吗？**

`xxh64().intdigest()` 返回 0 到 2^64-1 的无符号整数。Python 的 `-1` 是有符号的，和 `2^64 - 1` 是不同的 Python 对象。所以 `-1` 永远不会作为合法哈希值出现。安全，但隐晦。

### 1.5 本章小结

> 💡 小结
> - nano-vllm 使用 xxh64 链式哈希实现内容寻址，Block N 的哈希包含 Block 0 到 N 的完整前缀信息
> - 哈希只依赖 token IDs 和前驱哈希，不含位置、层等冗余维度
> - 碰撞安全通过二次 token 内容校验保障，代价是每 Block 约 7KB 的冗余内存
> - 哈希表是纯 Python dict，无大小限制，条目惰性删除


---

## 二、Block 生命周期与惰性淘汰：如何让 Block "死而不删"

### 2.1 设计哲学与核心问题

传统的内存管理是二态的：allocated 或 free。但 prefix cache 需要第三种状态：**"已释放但数据仍有效"**。一个 Sequence 完成推理后释放了 Block，但 GPU 显存中的 KV 数据没有被覆写。如果下一个 Sequence 恰好需要相同的前缀，这些 Block 可以直接"复活"，省掉重新计算。

这就要求：释放 Block 时，不清除它的哈希和数据；只有当 Block 的物理空间真正被其他内容征用时，才删除它的哈希条目。这就是**惰性淘汰**。

### 2.2 源码入口与关键对象

```
nanovllm/engine/block_manager.py
  - BlockManager.free_block_ids: deque[int]（行 32）——空闲 Block 队列
  - BlockManager.used_block_ids: set[int]（行 33）——正在使用的 Block 集合
  - BlockManager._allocate_block（行 43-51）——分配时惰性清除旧哈希
  - BlockManager._deallocate_block（行 53-56）——释放时保留哈希
  - BlockManager.deallocate（行 94-101）——引用计数递减，到 0 才释放
```

### 2.3 主流程拆解

#### Block 的三种隐式状态

nano-vllm 没有显式的状态枚举，但 Block 实际上有三种状态：

```
┌───────────────────────────────────────────────────────────────┐
│              Block 生命周期状态机                               │
│                                                               │
│  ┌─────────┐  _allocate_block   ┌──────────┐  deallocate     │
│  │  Free   │ ────────────────→  │  Active  │ ────────────→   │
│  │(hash=-1)│                    │(ref>0)   │   ref_count--   │
│  └─────────┘                    └──────────┘                  │
│       ↑                              │                        │
│       │  _allocate_block             │ ref_count → 0          │
│       │  (覆写旧哈希)               ↓                        │
│  ┌──────────────┐              ┌───────────────┐              │
│  │ 被征用的旧块   │←────────── │   Cached      │              │
│  │ (hash 被删除)  │            │ (ref=0,       │              │
│  └──────────────┘              │  hash 有效,    │              │
│                                │  在 free 队列) │              │
│                                └───────────────┘              │
│                                     ↑                         │
│                                     │ can_allocate 命中        │
│                                     │ allocate 复活            │
│                                     ↓                         │
│                                ┌──────────┐                   │
│                                │  Active  │                   │
│                                │(ref=1)   │                   │
│                                └──────────┘                   │
└───────────────────────────────────────────────────────────────┘
```

1. **Active**（`block_id ∈ used_block_ids`，`ref_count > 0`）：被至少一个 Sequence 引用。不可淘汰。
2. **Cached**（`block_id ∈ free_block_ids`，`hash ≠ -1`）：没有 Sequence 引用，但哈希和 GPU KV 数据仍然有效。可以被复用，也可以被征用。
3. **Free**（`block_id ∈ free_block_ids`，`hash = -1`）：完全空白的 Block。

Cached 和 Free 之间没有显式区分——都在同一个 `free_block_ids` 队列里。区别仅在于 Block 对象的 `hash` 字段是否为 `-1`。

#### 释放：保留哈希

```python
# block_manager.py:53-56
def _deallocate_block(self, block_id: int):
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)  # 放到队尾
```

注意：没有 `block.hash = -1`，没有 `del self.hash_to_block_id[block.hash]`。Block 的哈希和 token_ids 完整保留。Block 被追加到 `free_block_ids` 的**右端**（队尾）。

#### 分配：惰性清除

```python
# block_manager.py:43-51
def _allocate_block(self) -> int:
    block_id = self.free_block_ids.popleft()  # 从队头取
    block = self.blocks[block_id]
    assert block.ref_count == 0
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]  # 惰性清除
    block.reset()  # ref_count=1, hash=-1, token_ids=[]
    self.used_block_ids.add(block_id)
    return block_id
```

分配新 Block 时，从 `free_block_ids` 的**左端**（队头）弹出。如果这个 Block 还有残留的哈希条目，就从 `hash_to_block_id` 中删除。然后重置 Block 状态。

队头弹出 + 队尾追加 = **FIFO 淘汰顺序**：最早释放的 Block 最先被征用。刚释放的 Block 排在队尾，有更长的"存活窗口"等待潜在的缓存命中。

#### 引用计数与共享

```python
# block_manager.py:94-101
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

释放是**反向遍历** `block_table`，逐块递减引用计数。只有 `ref_count` 降到 0 才真正放回空闲队列。

反向遍历有一个微妙的效果：后缀 Block（更唯一，更不可能被复用）先进入空闲队列右端。但由于 FIFO 从左端取，这些后缀 Block 反而"存活更久"。前缀 Block 后进入右端，但因为通常 `ref_count > 1`（被多个 Sequence 共享），它们可能根本不会降到 0。最终效果是：**共享度高的前缀 Block 通过引用计数天然保持活跃**，而共享度低的 Block 靠 FIFO 队列自然淘汰。

#### 缓存复活路径

当 `allocate` 发现一个缓存 Block 在空闲队列中（`block_id not in used_block_ids`）时：

```python
# block_manager.py:85-88
else:  # block_id 在 free_block_ids 中
    block.ref_count = 1
    self.free_block_ids.remove(block_id)  # ⚠️ O(n) 操作！
    self.used_block_ids.add(block_id)
```

Block 被从空闲队列中移除（复活），放回 `used_block_ids`。**这里的 `deque.remove()` 是 O(n) 操作**，是整个 prefix cache 实现中最大的性能瓶颈。

### 2.4 关键细节与误区澄清

**误区四：释放 Block 会清空 GPU 显存中的 KV 数据？**

不会。`_deallocate_block` 只改变 Python 层的元数据（`used_block_ids`、`free_block_ids`）。GPU 上的 KV 缓存张量 `self.kv_cache` 完全不受影响。KV 数据只在下一次 `store_kvcache` 内核写入该 Block 的 slot 时才会被覆写。这就是为什么"复活"一个 Cached Block 是安全的——GPU 数据从未改变。

**误区五：`hash_to_block_id` 的覆写行为**

在 `hash_blocks`（行 120）中：

```python
self.hash_to_block_id[h] = block.block_id
```

这是无条件覆写。如果两个 Sequence（A 和 B）共享相同前缀，且都经过 `hash_blocks`，B 的 Block ID 会覆盖 A 的 Block ID。当 B 完成并释放时，哈希表指向的是 B 的（已释放的）Block。A 的 Block 仍然有效，但已经"失联"——无法通过哈希表找到。

这不会导致正确性问题（`can_allocate` 会验证 Block 内容），但会降低缓存命中率。一个更好的策略是：只在 Block 第一次被哈希时写入映射，或者优先映射到 `used_block_ids` 中的 Block。

### 2.5 本章小结

> 💡 小结
> - Block 有三种隐式状态：Active（被引用）、Cached（空闲但哈希有效）、Free（完全空白）
> - 释放 Block 不清除哈希也不清除 GPU 数据，实现"惰性淘汰"
> - 淘汰策略是隐式 FIFO：空闲队列左端取出、右端追加
> - 缓存复活时 `deque.remove()` 是 O(n)，是已知的性能瓶颈
> - `hash_to_block_id` 的无条件覆写可能导致有效 Block 失联


---

## 三、调度层如何感知缓存命中

### 3.1 设计哲学与核心问题

Prefix Cache 的所有"策略决策"都发生在 Scheduler 层，而 BlockManager 只提供"机制"。Scheduler 需要回答两个问题：

1. 一个新 Sequence 有多少前缀 Block 可以复用？（`can_allocate`）
2. 复用后，实际需要 prefill 多少 token？（`num_tokens` 的调整）

这两个答案直接决定了一个 step 中的计算量和 Block 消耗量。

### 3.2 源码入口与关键对象

```
nanovllm/engine/scheduler.py
  - Scheduler.schedule（行 25-73）：prefill 调度主循环
  - Scheduler.preempt（行 75-79）：抢占释放
  - Scheduler.postprocess（行 81-92）：步后哈希注册与状态推进
```

### 3.3 主流程拆解

#### 缓存探测与 token 数量缩减

```python
# scheduler.py:30-52（简化注释）
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0:
        break
    if not seq.block_table:  # 首次调度
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1:  # 空间不足
            break
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size  # ← 关键缩减
    else:  # 恢复 chunked prefill
        num_tokens = seq.num_tokens - seq.num_cached_tokens
```

核心逻辑在第 39 行：`num_tokens = seq.num_tokens - num_cached_blocks * self.block_size`。如果一个 1024-token 的 prompt 命中了 3 个 Block（每个 256 token），那么实际需要 prefill 的只有 1024 - 768 = 256 个 token。这个缩减后的 `num_tokens` 直接影响 `num_batched_tokens` 的预算消耗。

#### can_allocate 的匹配逻辑

```python
# block_manager.py:58-73
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):    # 注意：排除最后一个 Block
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break                          # 首次 miss 即停止
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1            # 已用的 Block 不需要新空间
    if len(self.free_block_ids) < num_new_blocks:
        return -1
    return num_cached_blocks
```

关键设计点：

1. **`range(seq.num_blocks - 1)`**：最后一个 Block 永远不参与缓存匹配。因为最后一个 Block 可能只部分填充，而 `hash_blocks` 只对完整 Block 计算哈希。
2. **首次 miss 即停止**：由于链式哈希，Block N miss 意味着 Block N+1 的哈希一定不同（即使 token 相同），所以贪心匹配等价于最优匹配。
3. **`num_new_blocks` 的记账**：初始值为 `seq.num_blocks`（所有 Block 都需要新空间）。如果命中的 Block 在 `used_block_ids` 中（已被其他 Sequence 引用），那么不需要额外空间（`num_new_blocks -= 1`）。如果命中的 Block 在空闲队列中（Cached 状态），不递减 `num_new_blocks`——因为复活它需要从空闲队列中取出一个槽位。
4. **可行性检查**：最终检查 `len(self.free_block_ids) >= num_new_blocks`。不够就返回 -1，Scheduler 停止调度。

#### allocate 的两阶段分配

```python
# block_manager.py:75-92
def allocate(self, seq: Sequence, num_cached_blocks: int):
    # 阶段 1：复用缓存 Block
    for i in range(num_cached_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1           # 共享：引用计数 +1
        else:
            block.ref_count = 1            # 复活：从空闲队列移除
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        seq.block_table.append(block_id)
    # 阶段 2：分配新 Block
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

分配完成后，`seq.num_cached_tokens` 被设置为缓存命中的 token 总数。这个值将被 ModelRunner 用来决定输入的起始位置。

#### 抢占与缓存保留

```python
# scheduler.py:75-79
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

抢占发生在 decode 阶段空间不足时。被抢占的 Sequence 释放所有 Block，回到等待队列。由于 `deallocate` 不清除哈希，这些 Block 带着有效的哈希进入空闲队列。当该 Sequence 被重新调度时，`can_allocate` 可能找回这些 Block——只要它们没被其他分配征用。

但有一个代价：`seq.block_table.clear()` 和 `seq.num_cached_tokens = 0`。重新调度时必须从 Block 0 重新走一遍哈希链。如果链中间有某个 Block 已被征用（哈希条目被删），链就断裂，后续所有 Block 即使数据完好也无法找回。

#### hash_blocks 的注册时机

```python
# block_manager.py:110-121
def hash_blocks(self, seq: Sequence):
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
    if start == end: return
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id
```

这个方法在 `postprocess` 中调用（`scheduler.py:83`），时机是**模型前向计算完成之后**。只哈希本次 step 中"刚好填满"的 Block（通过整除 `block_size` 判断）。部分填充的最后一个 Block 不会被哈希。

`start` 和 `end` 的含义：
- `start`：上一次已缓存/已哈希到哪个 Block
- `end`：本次计算后应该哈希到哪个 Block

对于 decode 阶段，每次只处理 1 个 token，所以只有当恰好跨越 Block 边界时（每 256 个 decode step 一次）才会触发哈希注册。

### 3.4 关键细节与误区澄清

**误区六：Prefix Cache 需要用户手动开启？**

不需要。nano-vllm 没有 `enable_prefix_caching` 开关。缓存逻辑无条件运行——每次 `can_allocate` 都会尝试哈希匹配，每次 `postprocess` 都会注册新哈希。即使所有请求的 prompt 都不同（如 `bench.py` 中的随机 token），这些逻辑依然执行，只是永远匹配不上。

这意味着 prefix cache 的哈希计算开销（每 Block 约 2-3 微秒，主要是 `np.array().tobytes()` 的开销）是始终存在的。对于 prompt 完全随机的 benchmark 场景，这是纯开销。

**误区七：`num_new_blocks` 的名字暗示"新分配的 Block 数量"？**

不是。`num_new_blocks` 的真实含义是"需要从空闲队列中获取的 Block 数量"——包括要复活的 Cached Block 和全新分配的 Block。只有已在 `used_block_ids` 中的共享 Block 才会减少 `num_new_blocks`。这个命名容易误导。

### 3.5 本章小结

> 💡 小结
> - 调度层通过 `can_allocate` 做贪心最长前缀匹配，首次 miss 即停止
> - 缓存命中直接缩减 prefill token 数量，影响批量 token 预算消耗
> - 最后一个 Block 永远不参与缓存（`range(num_blocks - 1)`），单 Block prompt 无法受益
> - 抢占后重新调度需要从 Block 0 重走哈希链，代价与 prompt 长度成正比


---

## 四、ModelRunner 与 Attention 层：Q/K 不对称的处理

### 4.1 设计哲学与核心问题

前三章解决了"在 Python 调度层，如何识别和复用 Block"。但真正的计算发生在 GPU 上。Prefix Cache 命中后，模型只需要计算后缀 token 的 QKV 投影和 MLP，但 Attention 仍然需要看到完整序列的 K/V（包括已缓存的前缀部分）。

这导致一个不对称：Q 的长度 = 后缀 token 数量，K/V 的长度 = 前缀 + 后缀 = 完整序列长度。FlashAttention 的 `flash_attn_varlen_func` 必须接收分离的 `cu_seqlens_q` 和 `cu_seqlens_k`，以及 `block_table` 来寻址 paged KV 缓存。

### 4.2 源码入口与关键对象

```
nanovllm/engine/model_runner.py
  - prepare_prefill（行 129-170）：构建 Q/K 不对称的输入
  - prepare_block_tables（行 123-127）：Block Table 张量化

nanovllm/layers/attention.py
  - Attention.forward（行 59-75）：prefix cache 模式的分支逻辑
  - store_kvcache / store_kvcache_kernel（行 10-40）：KV 写入 paged cache

nanovllm/utils/context.py
  - Context dataclass（行 5-14）：携带 block_tables 等元数据到 Attention 层
```

### 4.3 主流程拆解

#### prepare_prefill 中跳过已缓存 token

```python
# model_runner.py:129-170（关键行标注）
def prepare_prefill(self, seqs: list[Sequence]):
    ...
    for seq in seqs:
        start = seq.num_cached_tokens           # ← 行 139：从缓存位置之后开始
        seqlen_q = seq.num_scheduled_tokens      # ← 行 140：Q 长度 = 后缀 token 数
        end = start + seqlen_q
        seqlen_k = end                           # ← 行 142：K 长度 = 完整序列到 end

        input_ids.extend(seq[start:end])         # ← 只取后缀 token
        positions.extend(range(start, end))      # ← 位置从 start 开始，不是从 0
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
```

这是 prefix cache 对模型输入的核心影响：

- `input_ids` 只包含 `[start:end]` 的 token，跳过了前 `num_cached_tokens` 个已缓存 token
- `positions` 从 `start` 开始而非 0，保证 RoPE 位置编码的正确性
- `seqlen_q ≠ seqlen_k`：Q 只有后缀，K 包含完整序列（因为 Attention 需要看到缓存部分的 K/V）

#### Slot Mapping 的精确控制

```python
# model_runner.py:151-161
start_block = start // self.block_size
end_block = (end + self.block_size - 1) // self.block_size
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * self.block_size
    if i == start_block:
        slot_start += start % self.block_size    # Block 内偏移
    ...
    slot_mapping.extend(range(slot_start, slot_end))
```

Slot Mapping 决定了 Triton 内核 `store_kvcache_kernel` 将新计算的 K/V 写入 KV 缓存的哪些位置。关键点：`start_block` 从 `num_cached_tokens // block_size` 开始，这意味着已缓存 Block 的 slot 不会出现在 slot_mapping 中——**已缓存 Block 的 GPU 数据不会被覆写**。

#### 不对称检测与 block_tables 传递

```python
# model_runner.py:162-163
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache 检测
    block_tables = self.prepare_block_tables(seqs)
```

这是整个 ModelRunner 层感知 prefix cache 的唯一判断点。如果批次中任何 Sequence 有缓存命中，累计 K 长度就会超过累计 Q 长度，触发 `block_tables` 的准备。

#### Attention 层的 Prefix Cache 分支

```python
# attention.py:59-75
def forward(self, q, k, v):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)    # 写入新 K/V
    if context.is_prefill:
        if context.block_tables is not None:    # ← prefix cache 分支
            k, v = k_cache, v_cache              # ← 用整个 paged KV 缓存替代
        o = flash_attn_varlen_func(q, k, v,
            cu_seqlens_q=context.cu_seqlens_q, cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q, max_seqlen_k=context.max_seqlen_k,
            softmax_scale=self.scale, causal=True, block_table=context.block_tables)
```

这里发生了什么：

1. **无论是否有缓存**，新计算的 K/V（后缀 token 的 QKV 投影结果）都被 `store_kvcache` 写入 paged 缓存
2. 如果 `block_tables` 不为 None（有缓存命中），`k` 和 `v` 被替换为完整的 `k_cache` 和 `v_cache` 张量
3. `flash_attn_varlen_func` 通过 `block_table` 从 paged 缓存中读取 K/V——包括已缓存的前缀部分和刚写入的后缀部分

没有缓存命中时（`block_tables is None`），`k` 和 `v` 保持为刚计算的张量，`flash_attn_varlen_func` 在连续内存上做标准变长 Attention（`cu_seqlens_q == cu_seqlens_k`，不需要 paged 读取）。

#### store_kvcache_kernel 的 slot=-1 保护

```python
# attention.py:10-30
@triton.jit
def store_kvcache_kernel(...):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return    # ← 跳过无效 slot
    ...
```

Triton 内核有 `slot == -1` 的保护。虽然在 prefix cache 的正常路径下不会产生 `-1` slot（slot_mapping 只包含非缓存 token 的 slot），但这个保护为 CUDA Graph 模式下的 padding（`model_runner.py:206`，`slot_mapping.fill_(-1)`）提供了安全网。

### 4.4 关键细节与误区澄清

**误区八：混合批次（部分命中、部分未命中）会有问题？**

不会。当 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 时，整个批次切换到 paged KV 读取模式。对于批次中没有缓存命中的 Sequence，其 `seqlen_q == seqlen_k`，FlashAttention 会正确处理——它读取的 K/V 数量等于该 Sequence 的 Q 数量。已通过 `store_kvcache` 刚写入 paged 缓存的 K/V 会被正确读出（CUDA 流内的操作顺序保证）。

**误区九：Prefix Cache 会影响 CUDA Graph？**

不会直接影响。Prefix Cache 只在 prefill 阶段生效，而 prefill 始终走 eager 模式（`model_runner.py:197`）：

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
```

CUDA Graph 只用于 decode，而 decode 阶段没有 prefix cache 逻辑。间接影响是：prefix cache 让更多 Sequence 同时进入 decode 阶段（因为 prefill 更快），可能导致使用更大的 CUDA Graph batch size。

**误区十：Prefix Cache 模式下 Attention 计算量和无缓存时一样？**

不一样。计算量从 O(seqlen_k²) 降低到 O(seqlen_q × seqlen_k)。例如 seqlen_k = 1024、seqlen_q = 256（768 token 缓存命中），Attention FLOPs 降低约 4 倍。再加上 QKV 投影、MLP 等也只处理 seqlen_q 个 token，总计算量大幅缩减。

### 4.5 本章小结

> 💡 小结
> - ModelRunner 通过 `num_cached_tokens` 跳过已缓存 token 的输入，但保留正确的位置编码
> - Q/K 不对称通过分离的 `cu_seqlens_q` 和 `cu_seqlens_k` 传递给 FlashAttention
> - 不对称检测是批次级的（比较累计长度），一个 Sequence 命中就整个批次切换到 paged 模式
> - Slot Mapping 精确排除已缓存 Block，保证 GPU 数据不被覆写


---

## 五、完整主路径串联

### 5.1 完整调用栈

以一个典型场景为例：Sequence A（1024 token prompt）已完成推理并释放。Sequence B（相同的前 768 token + 不同的后 256 token）到来。

```
User: llm.generate([prompt_B], sampling_params)
  │
  ├─ Step 1: add_request
  │     └─ Scheduler.add(seq_B)  →  waiting 队列
  │
  ├─ Step 2: schedule()  →  prefill 调度
  │     ├─ seq_B 没有 block_table（首次调度）
  │     ├─ BlockManager.can_allocate(seq_B)
  │     │     ├─ Block 0: compute_hash → 查 hash_to_block_id → 命中 ✓
  │     │     │    └─ token_ids 校验通过，num_cached_blocks=1
  │     │     ├─ Block 1: compute_hash(prefix=h0) → 命中 ✓，num_cached_blocks=2
  │     │     ├─ Block 2: compute_hash(prefix=h1) → 命中 ✓，num_cached_blocks=3
  │     │     └─ Block 3: (最后一个 Block，跳过)
  │     │     → return 3
  │     ├─ num_tokens = 1024 - 3*256 = 256  ← 只需 prefill 256 token
  │     ├─ BlockManager.allocate(seq_B, 3)
  │     │     ├─ Block 0,1,2: 已在 used_block_ids → ref_count++
  │     │     └─ Block 3: _allocate_block() → 新分配
  │     └─ seq_B.num_cached_tokens = 768
  │
  ├─ Step 3: model_runner.call("run", [seq_B], is_prefill=True)
  │     ├─ prepare_prefill
  │     │     ├─ start = 768, end = 1024
  │     │     ├─ input_ids = seq_B[768:1024]  ← 只有 256 个 token
  │     │     ├─ positions = range(768, 1024)  ← 正确的绝对位置
  │     │     ├─ cu_seqlens_q = [0, 256]       ← Q 长度 = 256
  │     │     ├─ cu_seqlens_k = [0, 1024]      ← K 长度 = 1024
  │     │     ├─ slot_mapping: 只覆盖 Block 3 的 slot
  │     │     └─ block_tables: [[B0, B1, B2, B3]]  ← 因为 K > Q
  │     ├─ run_model (eager, 不用 CUDA Graph)
  │     │     └─ 模型前向：Embedding(256 token) → 28 层 Transformer
  │     │          └─ 每层 Attention.forward:
  │     │               ├─ store_kvcache: 将 256 个新 K/V 写入 Block 3
  │     │               ├─ block_tables is not None → k,v = k_cache, v_cache
  │     │               └─ flash_attn_varlen_func(q[256], k_cache, v_cache,
  │     │                    cu_seqlens_q=[0,256], cu_seqlens_k=[0,1024],
  │     │                    block_table=[[B0,B1,B2,B3]])
  │     └─ Sampler: logits → token_id
  │
  └─ Step 4: postprocess
        ├─ hash_blocks(seq_B)
        │     ├─ start=3, end=4 → Block 3 刚好填满（1024/256=4）
        │     ├─ h = blocks[B2].hash  ← 前驱 Block 的已知哈希
        │     └─ 计算 Block 3 哈希 → 注册到 hash_to_block_id
        ├─ num_cached_tokens += 256 → 1024
        ├─ append_token(token_id)
        └─ 进入 decode 循环...
```

### 5.2 每一层做了什么

| 阶段 | 输入 | 输出/状态变化 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| can_allocate | Sequence 的 token_ids | num_cached_blocks (int) | 无 | 无 | 每个 Sequence 首次调度 |
| allocate | seq, num_cached_blocks | seq.block_table 填充 | 无 | 无 (Python 元数据) | 每个 Sequence 首次调度 |
| prepare_prefill | seqs 列表 | input_ids, positions, Context | 无 (rank 0) | 少量 CPU→GPU 传输 | 每个 prefill step |
| store_kvcache | 新 K/V 张量, slot_mapping | KV 缓存写入 | 无 | GPU KV 缓存写入 | 每层每 step |
| flash_attn | Q, K_cache, V_cache, block_table | Attention 输出 | 无 | GPU 计算 | 每层每 step |
| hash_blocks | seq 的 block_table | hash_to_block_id 更新 | 无 | 无 (Python 元数据) | 每个 postprocess |

### 5.3 哪些逻辑不在主路径

| 函数/逻辑 | 看似相关的原因 | 实际不在主流程的原因 |
|---|---|---|
| `Block.reset()` | 看起来是 Block 初始化 | 只在 `_allocate_block` 中被调用，不在缓存命中路径 |
| `store_kvcache_kernel` 的 `slot == -1` 保护 | 像是 prefix cache 的 skip 逻辑 | 实际是为 CUDA Graph 的 padding 服务，prefix cache 路径不产生 `-1` slot |
| `Sequence.__getstate__` 中的 `block_table` 序列化 | 看起来 prefix cache 需要特殊处理 | 实际 block_table 无论是否缓存命中都会被序列化到 worker，没有特殊逻辑 |
| `prepare_decode` 中的 `block_tables` | decode 也用 block_tables | 但 decode 阶段不涉及 prefix cache 逻辑，block_tables 纯粹用于 paged KV 读取 |


---

## 六、核心机制深挖

### 6.1 FIFO 淘汰：简单但不最优

nano-vllm 的淘汰策略完全由 `free_block_ids` 的 deque 结构隐式决定：

- 释放 → `append`（右端）
- 分配 → `popleft`（左端）

这产生 FIFO 淘汰：最早释放的 Block 最先被征用。相比 vLLM 的 LRU Evictor，这意味着：

**缺陷场景**：一个高频使用的 system prompt 对应的 Block 被释放后排入队列。随后一个罕见的 prompt 的 Block 也被释放，排在后面。如果需要分配新 Block，高频 Block 因为先释放而先被征用——即使它更有可能被复用。

**部分缓解**：引用计数提供了自然的保护。如果多个 Sequence 同时使用同一个 system prompt Block，该 Block 的 `ref_count > 1`，不会因为单个 Sequence 完成而释放。只有所有使用者都完成后才进入空闲队列。高并发下，热门 Block 的 `ref_count` 天然较高，不易被释放。

**复活的 LRU 效应**：如果一个 Cached Block 在空闲队列中被命中（`allocate` 从空闲队列 remove），它会被移出空闲队列。后续再次释放时，它重新 `append` 到队尾，位置被刷新。这提供了一种弱 LRU 效果——被复用过的 Block 推迟淘汰。

### 6.2 `deque.remove()` 的性能隐患

`allocate` 中复活 Cached Block 的路径：

```python
self.free_block_ids.remove(block_id)  # O(n)
```

Python 的 `collections.deque.remove()` 需要线性扫描。假设系统有 1000 个空闲 Block，一个 Sequence 命中 10 个 Cached Block 需要复活，总代价为 10 × 1000 = 10000 次比较。在 CPython 中，每次比较约 50ns，总计约 500μs。

对比：整个 `can_allocate` 中 10 个 Block 的哈希计算约 30μs（每 Block 3μs）。`deque.remove` 比哈希计算慢一个数量级。

**替代方案**：用 `OrderedDict` 或 `dict` + 双向链表实现空闲集合，可以做到 O(1) 的 remove 和 FIFO/LRU 淘汰。这是一个纯数据结构层面的优化，不影响接口。

### 6.3 链式哈希的断裂与"全有或全无"

链式哈希的特性：Block N 的哈希依赖 Block N-1 的哈希。如果一个 10-Block 的前缀中，Block 5 被征用（数据覆写、哈希清除），那么 Block 6-9 即使数据完好，也无法通过哈希匹配找回——因为 `can_allocate` 在 Block 5 处就 `break` 了。

这是内容寻址的必然代价。transformer 的 KV 值确实依赖完整的前缀上下文，Block 5 被覆写意味着 Block 6-9 的 KV 值是基于旧的 Block 5 计算的，如果前缀变了，这些 KV 值就是错误的。所以"链断即停"在语义上是正确的。

但这也意味着：FIFO 淘汰如果征用了链中间的某个 Block，整条链的后半段全部失效。一个更精细的淘汰策略可以优先征用不属于任何链尾的孤立 Block，减少链断裂的概率。

### 6.4 配置如何影响缓存效果

`kvcache_block_size`（默认 256，`config.py:17`）是影响 prefix cache 效果的最关键参数：

| Block Size | 缓存命中粒度 | 最小受益 prompt 长度 | 内部碎片（平均） | 元数据开销 |
|---|---|---|---|---|
| 256 (默认/最小) | 粗（256 token 一匹配） | 257 token | 128 token/seq | 低 |
| 512 | 更粗 | 513 token | 256 token/seq | 更低 |

vLLM 默认 block_size = 16，可以匹配短至 17 token 的共享前缀。nano-vllm 的 256 意味着：

- system prompt < 256 token → **完全无法受益于 prefix cache**
- 两个 Sequence 共享 500 token 前缀 → 只能缓存 256 token（1 个 Block），浪费 244 token 的共享部分

这是一个重要的架构取舍：大 Block Size 减少了 Block Table 的大小和管理开销，让 FlashAttention 的 paged 读取更高效（更少的页表项），但牺牲了缓存粒度。


---

## 七、显存、性能与通信分析

### 7.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---|---|
| 模型参数 | ❌ | Prefix Cache 不涉及参数 |
| Optimizer State | ❌ | 推理引擎无 optimizer |
| KV 缓存总量 | ❌ | `num_kvcache_blocks` 在启动时固定分配，缓存命中不改变总量 |
| KV 缓存利用率 | ✅ | 共享 Block 让相同物理空间服务更多 Sequence |
| Activation | ✅ | prefill 处理更少 token → 中间激活更小 |
| 输入 Tensor | ✅ | `input_ids` 更短 → Embedding 查找更少 |

**prefix cache 的核心收益不是"省显存"，而是"提高 KV 缓存的利用效率"和"减少计算量"。**

以 Qwen3-0.6B（28 层、4 KV head、128 head_dim、bfloat16）为例：

```
每个 Block 的 KV 显存 = 2(K+V) × 28(层) × 256(token) × 4(KV head) × 128(head_dim) × 2(bytes)
                      = 14,680,064 bytes ≈ 14 MB
```

100 个 Sequence 共享 768 token 前缀（3 个 Block）：
- 无 prefix cache：100 × 3 = 300 个 Block × 14 MB ≈ 4.2 GB
- 有 prefix cache：3 + 100 × 1 = 103 个 Block × 14 MB ≈ 1.4 GB
- **有效节省约 2.8 GB 的 KV 缓存空间**，可以用于服务更多并发 Sequence

### 7.2 计算节省

对于 Qwen3-0.6B，每个 token 每层的主要计算量：
- QKV 投影 + Output 投影：~8.4M FLOPs
- MLP（SwiGLU）：~25.2M FLOPs
- 每 token 每层总计：~33.6M FLOPs
- 每 token 所有层：~940M FLOPs

缓存 768 个 token → 节省 768 × 940M ≈ **720 GFLOPs** 的投影和 MLP 计算。

此外，Attention 的 FLOPs 从 O(1024²) 降到 O(256 × 1024) ≈ 4 倍缩减。

### 7.3 通信开销

**Prefix Cache 不引入额外通信。**

- 哈希计算、Block 分配、调度决策全部在 Rank 0 的主进程完成
- `hash_to_block_id` 字典不需要跨 rank 同步
- Worker 进程通过 `Sequence.__getstate__` 接收 `block_table`，但 block_table 是普通数据，不管是否缓存命中都会被传输
- NCCL 通信（`all_reduce`、`gather`）与 prefix cache 无关，只发生在模型前向计算中

唯一的代价是序列化数据量可能微增：缓存命中导致更多 Sequence 同时进入 decode 阶段，每次 `write_shm` 需要序列化更多 Sequence 对象。但每个 decode Sequence 只传输 `last_token`（单个 int），增量极小。

### 7.4 性能取舍总结

| 维度 | 收益 | 代价 |
|---|---|---|
| Prefill 计算量 | 减少（跳过缓存前缀的 QKV/MLP） | 无 |
| Attention FLOPs | 减少（Q 更短，但 K 不变） | 无 |
| KV 缓存利用率 | 提高（共享 Block） | 无 |
| 调度吞吐 | 提高（更多 Sequence 可以同时 prefill） | 无 |
| 哈希计算 | — | 约 3μs/Block，始终执行 |
| token_ids 存储 | — | 约 7KB/Block 的冗余内存 |
| deque.remove | — | O(n) 的复活代价 |
| 缓存粒度 | — | 最小 256 token 才有收益 |


---

## 八、配置项、边界条件与坑点

| 配置项 | 影响 | 源码路径 | 行为变化 | 风险/坑点 |
|---|---|---|---|---|
| `kvcache_block_size` (默认 256) | 缓存匹配粒度 | `config.py:17` | 值越大粒度越粗、越不容易命中 | 必须是 256 的倍数，不能减小，短 prompt 无法受益 |
| `gpu_memory_utilization` (默认 0.9) | 总 Block 数量 | `model_runner.py:113` | 值越大 Block 越多，缓存池越大 | 过高可能导致 OOM（未留足 activation 空间） |
| `max_num_seqs` (默认 512) | 最大并发 Sequence | `config.py:10` | 更多并发 → 更多共享机会 | Block 不够时触发抢占 |
| `max_num_batched_tokens` (默认 16384) | prefill token 预算 | `config.py:9` | 缓存命中缩减了 token 消耗 → 更多 seq 可同时 prefill | — |
| `enforce_eager` (默认 False) | 是否禁用 CUDA Graph | `config.py:14` | 不影响 prefix cache（prefill 始终 eager） | — |

**边界条件**：

1. **单 Block Sequence**（prompt ≤ 256 token）：`num_blocks = 1`，`range(1-1) = range(0)` → 零次循环 → `num_cached_blocks = 0`。永远不受益于 prefix cache。
2. **恰好填满一个 Block**（prompt = 256 token）：同上，`num_blocks = 1`。
3. **所有 prompt 完全不同**（如 bench.py 的随机 token）：每次 `can_allocate` 都做哈希计算但都 miss。纯开销，无收益。无法关闭。
4. **抢占后缓存链断裂**：被抢占的 Sequence 释放 Block → 某 Block 被其他 Sequence 征用 → 重新调度时链在中间断裂 → 后续 Block 全部需要重新计算。
5. **hash_to_block_id 覆写**：两个相同前缀的 Sequence 交替 `hash_blocks` → 哈希表指向后者的 Block → 前者的 Block 失联 → 降低缓存命中率。


---

## 九、测试、示例与覆盖缺口

### 9.1 已有示例覆盖

| 文件 | 场景 | 是否涉及 Prefix Cache |
|---|---|---|
| `example.py` | 2 个不同 prompt | 不触发（prompt 不共享前缀，且可能少于 256 token） |
| `bench.py` | 256 个随机 token 序列 | 不触发（prompt 全随机，无共享前缀） |

### 9.2 未覆盖风险

| 风险点 | 当前测试 | 可能后果 |
|---|---|---|
| 共享前缀场景的缓存命中 | ❌ 无 | 核心功能没有端到端验证 |
| 抢占后缓存复活 | ❌ 无 | 抢占 → 重新调度 → 哈希链重建的正确性 |
| 并发共享 Block 的引用计数 | ❌ 无 | ref_count 递增/递减的边界条件 |
| hash_to_block_id 覆写导致的 Block 失联 | ❌ 无 | 多 Sequence 共享前缀时缓存命中率降低 |
| Block Size = 256 的粒度限制 | ❌ 无 | 短 prompt 用户无法受益，无感知 |
| 张量并行下 block_table 序列化 | ❌ 无 | worker 侧 KV 缓存寻址正确性 |
| Chunked Prefill + Prefix Cache 交互 | ❌ 无 | num_cached_tokens 跨 chunk 的状态推进 |
| 性能基准（有 vs 无 prefix cache） | ❌ 无 | 无法量化收益 |

**整个项目没有测试套件**（`CLAUDE.md` 中明确指出"There is no test suite or linter configured"），所有路径都依赖手动验证。


---

## 十、局限性与已知优化点

### 10.1 硬约束

- **Block Size ≥ 256**：`config.py:22` 的 `assert self.kvcache_block_size % 256 == 0` 阻止了更细粒度的缓存。vLLM 默认 16。
- **只支持 Qwen3**：新模型需要在 `models/` 中添加新文件，但 prefix cache 本身是模型无关的。
- **不支持 beam search / parallel sampling 的 Copy-on-Write**：共享 Block 是只读的（完整 Block 一旦哈希就不可修改）。如果需要分支（beam 扩展），没有 CoW 机制来处理。
- **`temperature=0` 被禁止**（`SamplingParams` 中有 assert），无法测试确定性采样场景下的缓存命中正确性。

### 10.2 维护成本

- **哈希链无持久化**：引擎重启后所有缓存失效。对于长期运行的服务，每次冷启动都有缓存预热期。
- **无缓存命中指标**：没有 counter、log 或 metric 记录命中率。生产环境无法监控缓存效果。
- **`hash_to_block_id` 的覆写行为**：没有注释说明为什么允许覆写，维护者可能误以为这是 bug 或尝试"修复"它。
- **`free_block_ids.remove()` 的 O(n) 是已知性能债务**：随 Block 数量增长而恶化，但在 ~1200 行的教学代码中可以接受。

### 10.3 性能瓶颈

- **`deque.remove(block_id)` O(n)**：前面已详细分析。
- **`np.array(token_ids).tobytes()` 的堆分配**：每次哈希计算创建一个临时 NumPy 数组。可以通过预分配 buffer 或直接用 `struct.pack` 避免。
- **`can_allocate` 到 `allocate` 之间的重复哈希计算**：两个方法都遍历相同的 Block 做哈希。可以让 `can_allocate` 返回哈希列表，避免 `allocate` 重算。
- **序列化瓶颈**：Python pickle 序列化 Sequence 列表通过共享内存传到 worker。大 batch 时可能耗费数毫秒。

### 10.4 已知优化点

1. **用 `OrderedDict` 替换 `deque`**：O(1) 的 remove + FIFO 淘汰，无接口变化
2. **添加 `enable_prefix_caching` 开关**：随机 prompt 场景可跳过哈希计算
3. **缩小 Block Size**：需要调整 FlashAttention 的 page size 参数和 `config.py` 的 assert
4. **LRU 淘汰**：跟踪 Block 最后访问时间，优先淘汰最久未用的 Block
5. **保留抢占 Sequence 的 block_table**：避免重新调度时的 O(n) 哈希链重建
6. **`hash_to_block_id` 写入时优先保留 `used_block_ids` 中的映射**：减少 Block 失联
7. **缓存命中率指标**：在 `can_allocate` 中统计总检查 Block 数 vs 命中 Block 数，通过 `generate` 的 `pbar` 暴露


---

## 小结与展望

nano-vllm 的 Prefix Cache 实现可以用四个关键词概括：

**关键词一：链式内容寻址**

用 xxh64 链式哈希将"Block 是否相同"的语义问题转化为 O(1) 的字典查找。链式设计保证了因果依赖的正确性——前缀不同则后续哈希全部不同。这是 prefix cache 的数学基础。

**关键词二：惰性淘汰**

Block 释放后不清除哈希、不清除 GPU 数据，只在物理空间被征用时才删除哈希条目。这让 Block 在"逻辑释放"和"物理覆写"之间有一个缓存窗口，最大化复用机会。以 FIFO 队列实现，简单但不最优。

**关键词三：Q/K 不对称透传**

通过分离的 `cu_seqlens_q` 和 `cu_seqlens_k`，加上 `block_tables` 参数，将 prefix cache 的语义无缝传递给 FlashAttention。整个模型代码（Qwen3ForCausalLM）对 prefix cache 完全无感知——所有逻辑封装在 Scheduler、BlockManager 和 Context 三层。

**关键词四：粗粒度取舍**

256 token 的 Block Size 在 prefix cache 场景下显得过粗，但减少了 Block Table 大小和管理开销。这是 nano-vllm 作为教学项目的务实选择——用缓存粒度换实现简洁。

---

**适合的场景**：多个请求共享长 system prompt（> 256 token）的批量推理。高并发下通过引用计数和共享 Block 显著提升 KV 缓存利用率。

**不适合的场景**：短 prompt（< 256 token）、prompt 完全随机、需要 LRU 淘汰策略的长期运行服务。

**与 vLLM 的主要差距**：无 LRU Evictor、无 Prefix Tree（Radix Tree）、无 Copy-on-Write、Block Size 过大（256 vs 16）、无缓存命中指标。

**后续值得继续走读的方向**：
- Chunked Prefill 与 Prefix Cache 的交互细节（`num_cached_tokens` 跨 chunk 的状态机）
- CUDA Graph 在 decode 阶段如何与动态 Block Table 配合
- Tensor Parallelism 下 Sequence 序列化 / 反序列化的正确性边界
