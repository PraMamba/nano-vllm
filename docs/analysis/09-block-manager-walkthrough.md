# nano-vllm 源码走读：BlockManager 实现解析

> 在 LLM 推理场景中，KV Cache 是显存占用的绝对大头。对于一个同时服务数百条请求的推理引擎而言，如何高效管理 KV Cache 的分配、释放和复用，直接决定了系统的吞吐量上限。vLLM 用 PagedAttention 解决了这个问题，而 nano-vllm 作为 vLLM 核心推理引擎的极简复现（~1200 行），同样实现了这套分页式 KV Cache 管理。本文不展开 PagedAttention 的原理推导，而是聚焦源码，从 BlockManager 出发，追踪一个 block_id 如何从 CPU 侧的逻辑分配，一路流转到 GPU 侧的物理 KV Cache 写入，以及 Prefix Caching、抢占式调度等机制是如何围绕它展开的。

---

## 前言

### 业务 / 工程背景

LLM 推理引擎需要为每条正在处理的请求维护一份 KV Cache——Transformer 每层 Attention 的 Key 和 Value 向量。一个 4096 token 的序列，在 28 层、4 个 KV Head、128 维 Head Dim、bf16 精度下，单条序列的 KV Cache 约 58 MiB。同时服务 512 条请求，KV Cache 总量可达 29 GiB。

传统方式是为每条序列预分配最大长度的 KV Cache 空间，这会导致严重的显存浪费——大部分序列不会跑满最大长度，但显存却被锁死了。

### 核心矛盾

**KV Cache 的大小在推理过程中是动态增长的（每 decode 一步增加一个 token），但 GPU 显存不支持高效的动态分配。** 预分配浪费显存，按需分配则面临碎片化和分配开销。

PagedAttention 的解法是借鉴操作系统虚拟内存分页：将 KV Cache 切分为固定大小的 Block（页），按需分配 Block，用 Block Table（页表）维护逻辑到物理的映射。这样，序列的 KV Cache 在物理上可以不连续，消除外部碎片；Block 可以按需分配和释放，避免预分配浪费。

### 本文主线

本文围绕 nano-vllm 的 `BlockManager` 展开，分为以下几个机制：

1. **Block 与 BlockManager 的核心数据结构**：分页内存管理的 CPU 侧实现
2. **Prefix Caching 机制**：基于 xxhash 链式哈希的 KV Cache 复用
3. **Scheduler 与 BlockManager 的协作**：分配、追加、抢占的完整调度流程
4. **从 block_id 到物理 KV Cache**：Slot Mapping 与 Triton Kernel 的写入路径
5. **显存布局与性能分析**：Block Size、碎片化、通信开销的量化评估

### 不展开的内容

本文不讲 PagedAttention 的原始论文推导，不讲 Flash Attention 的底层 Tiling 算法，不讲 Tensor Parallelism 的 NCCL 通信细节。只讲 nano-vllm 源码中 BlockManager 如何实现、如何被调用、如何与上下游模块配合。

### 核心文件表

| 文件 | 职责 |
|---|---|
| `nanovllm/engine/block_manager.py` | Block 和 BlockManager 类：逻辑块分配、释放、Prefix Cache 索引 |
| `nanovllm/engine/scheduler.py` | Scheduler：调度 prefill/decode，调用 BlockManager 的分配/释放/哈希接口 |
| `nanovllm/engine/sequence.py` | Sequence：携带 block_table、token_ids 等状态 |
| `nanovllm/engine/model_runner.py` | ModelRunner：将 block_table 转换为 slot_mapping，分配物理 KV Cache 张量 |
| `nanovllm/layers/attention.py` | Attention + Triton Kernel：用 slot_mapping 写入 KV Cache，用 block_table 读取 |
| `nanovllm/utils/context.py` | 全局 Context：在模型前向中隐式传递 slot_mapping / block_tables |
| `nanovllm/config.py` | 配置：kvcache_block_size、gpu_memory_utilization 等参数 |

---

## 一、Block 与 BlockManager：分页内存管理的 CPU 侧实现

### 1.1 设计哲学与核心问题

BlockManager 要解决的核心问题是：**如何在 CPU 侧高效管理数千个 KV Cache Block 的分配和释放，同时支持跨请求的 Prefix Cache 复用。**

如果没有 BlockManager，推理引擎需要为每条序列预分配连续的 KV Cache 空间。这会导致：
- 外部碎片：已完成的序列释放的空间可能不够下一条序列使用
- 内部浪费：预分配最大长度但实际只用了一部分
- 无法复用：两条共享相同 System Prompt 的请求各自独立计算 KV Cache

BlockManager 通过分页机制消除了前两个问题，通过 Prefix Caching 解决了第三个。

关键设计决策是 **BlockManager 完全不感知 GPU**。它只操作整数（block_id）和 Python 数据结构（deque、dict、set），不持有任何 `torch.Tensor`。逻辑块到物理显存的映射由下游的 ModelRunner 和 Triton Kernel 完成。这是一种干净的关注点分离——分配策略与物理机制解耦。

### 1.2 源码入口与关键对象

```
nanovllm/engine/block_manager.py
  - Block：单个 KV Cache Block 的元数据（block_id, ref_count, hash, token_ids）
  - BlockManager：管理所有 Block 的分配/释放/缓存查找
    - _allocate_block / _deallocate_block：底层分配/释放
    - can_allocate / allocate：带 Prefix Cache 查询的分配
    - can_append / may_append：decode 阶段的追加分配
    - hash_blocks / compute_hash：Prefix Cache 的哈希注册
```

### 1.3 Block 数据结构

每个 Block 对象（`block_manager.py:8-23`）只有四个字段：

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id   # 不可变的物理块编号，对应 kv_cache 张量的第三维索引
        self.ref_count = 0         # 引用计数：有多少条序列正在使用这个块
        self.hash = -1             # 内容哈希：用于 Prefix Cache 查找
        self.token_ids = []        # 该块中存储的 token ID 列表（用于哈希碰撞校验）
```

`block_id` 是 Block 的物理身份——它直接对应 GPU 上 `kv_cache[layer][block_id]` 这块内存。BlockManager 创建时，所有 Block 按 0 到 `num_blocks - 1` 编号，之后 block_id 永远不变。变化的只有 `ref_count`、`hash` 和 `token_ids`。

`reset()` 方法将 `ref_count` 设为 1（不是 0——因为调用 `reset` 意味着正在分配出去），并清除哈希：

```python
def reset(self):
    self.ref_count = 1
    self.hash = -1
    self.token_ids = []
```

### 1.4 BlockManager 的四个核心数据结构

BlockManager 构造函数（`block_manager.py:28-33`）初始化四个数据结构：

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```

| 数据结构 | 类型 | 职责 |
|---|---|---|
| `blocks` | `list[Block]` | 按 block_id 索引的全量 Block 列表，类似物理页帧表 |
| `free_block_ids` | `deque[int]` | 空闲块 ID 的双端队列，前端分配 / 后端回收（FIFO） |
| `used_block_ids` | `set[int]` | 正在使用的块 ID 集合，O(1) 查找 |
| `hash_to_block_id` | `dict[int, int]` | Prefix Cache 索引：内容哈希 → block_id |

`free_block_ids` 使用 deque 而非 list 或 set，这个选择暗含了一个重要的设计意图——后文会详细分析。

### 1.5 Block 的生命周期状态机

一个 Block 在其生命周期中有三种状态：

```
                     ┌──────────────────────────┐
                     │          FREE             │
                     │  free_block_ids 中        │
                     │  ref_count = 0            │
                     │  hash = -1                │
                     └──────────┬───────────────┘
                                │
               _allocate_block()│popleft() + reset()
                                │
                                ▼
                     ┌──────────────────────────┐
                     │        ALLOCATED          │
                     │  used_block_ids 中        │
                     │  ref_count ≥ 1            │
                     │  hash 可能被 hash_blocks  │
                     │  设置为有效值              │
                     └────┬─────────────────┬───┘
                          │                 │
         deallocate()     │                 │ 另一条序列 cache hit
         ref_count -= 1   │                 │ ref_count += 1
         若 ref_count = 0 │                 │
                          ▼                 │
                     ┌──────────────────────┤
                     │  EVICTED-WITH-HASH   │
                     │  free_block_ids 中   │
                     │  ref_count = 0       │
                     │  hash ≠ -1 (仍有效)  │
                     │  hash_to_block_id    │
                     │  仍指向此 block      │
                     └────┬─────────────┬───┘
                          │             │
       _allocate_block()  │             │ Cache hit: allocate()
       (被当作新块回收)    │             │ 从 free 中 remove
       清理 hash 映射     │             │ 放回 used, ref_count=1
       reset()            │             │ (不调用 reset，保留 hash)
                          ▼             ▼
                     ┌────────┐   ┌──────────┐
                     │ FREE/  │   │ALLOCATED │
                     │ALLOCATED│   │(复用)    │
                     └────────┘   └──────────┘
```

**EVICTED-WITH-HASH 是 Prefix Caching 的核心**。当一个 Block 被释放后，它的 hash 和 token_ids 并不清除，仍然保留在 `hash_to_block_id` 索引中。物理 GPU 显存中的 KV 数据也没有被擦除（没有任何代码会主动 zero-out KV Cache）。如果后续有新请求的前缀与此 Block 匹配，这个 Block 可以被"复活"——不需要重新计算 KV Cache。

只有当这个 Block 被 `_allocate_block()` 从 free_block_ids 的前端弹出、重新分配给其他内容时，它的 hash 映射才会被清理（`block_manager.py:47-48`）：

```python
def _allocate_block(self) -> int:
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    assert block.ref_count == 0
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]
    block.reset()
    self.used_block_ids.add(block_id)
    return block_id
```

注意第 47 行的双重条件：不仅检查 `hash != -1`，还检查 `hash_to_block_id.get(block.hash) == block_id`。这是因为另一个 Block 可能已经通过 `hash_blocks` 注册了相同的 hash（相同内容的 Block 被重新计算），此时删除映射会破坏新 Block 的缓存。这个 guard 条件确保只清理"自己名下"的映射。

### 1.6 关键细节与误区澄清

**误区一：`_deallocate_block` 会清除 hash。**

不会。`_deallocate_block`（`block_manager.py:53-56`）只做两件事：从 `used_block_ids` 移除，追加到 `free_block_ids` 尾部。它 **不** 清除 Block 的 `hash` 和 `token_ids`，也 **不** 从 `hash_to_block_id` 删除条目。这正是 EVICTED-WITH-HASH 状态存在的原因——被释放的 Block 带着有效的缓存信息进入 free list，形成一个"幽灵缓存"。

**误区二：`free_block_ids` 中的所有块都是"空白"的。**

不是。free_block_ids 中可能有两种块：
1. 从未被使用过的空白块（hash = -1）
2. 曾被使用、已释放、但仍携带有效 hash 的块（EVICTED-WITH-HASH 状态）

**误区三：`ref_count` 为 0 的 Block 不会被任何代码访问。**

错误。`can_allocate` 在探测 Prefix Cache 时，会通过 `hash_to_block_id` 找到 EVICTED-WITH-HASH 的 Block，检查其 `token_ids` 字段来验证缓存命中。

### 1.7 💡 小结

- BlockManager 是一个纯 CPU 侧的逻辑块分配器，不持有 GPU 资源
- Block 有三种状态：FREE → ALLOCATED → EVICTED-WITH-HASH，其中 EVICTED-WITH-HASH 是 Prefix Caching 的关键
- `free_block_ids` 的 FIFO 特性（前端分配、后端回收）使得最近释放的块在 free list 中存活最久，最大化了缓存复用窗口
- `_allocate_block` 中的 hash cleanup guard 是防止错误删除其他 Block 缓存映射的安全措施

---

## 二、Prefix Caching 机制：基于 xxhash 链式哈希的 KV Cache 复用

### 2.1 设计哲学与核心问题

在实际部署中，大量请求共享相同的前缀——System Prompt、Few-shot 示例、文档上下文等。如果每条请求都独立计算前缀的 KV Cache，是巨大的算力浪费。Prefix Caching 的目标是：**如果两条请求的前缀完全相同，后到的请求可以直接复用前一条请求的 KV Cache，跳过前缀的 prefill 计算。**

核心难点在于"如何判断两个 Block 的内容完全相同"。不能简单比较 token_ids——因为 KV Cache 的值不仅取决于当前 Block 的 token，还取决于所有前序 Block 的 token（Attention 的因果性）。Block 3 的 KV 值受 Block 0-2 所有 token 的影响。因此，两个 Block 只有在 **相同位置** 且 **所有前序 Block 内容也相同** 时，KV Cache 才可以复用。

### 2.2 链式哈希构造

`compute_hash`（`block_manager.py:36-41`）用 xxhash-64 构建了一个类 Merkle 链结构：

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))   # 前一个 block 的 hash 作为前缀
    h.update(np.array(token_ids).tobytes())       # 当前 block 的 token 内容
    return h.intdigest()
```

哈希链的构造如下：

```
Block 0:  h₀ = xxh64(token_ids[0:256])
Block 1:  h₁ = xxh64(h₀_bytes ‖ token_ids[256:512])
Block 2:  h₂ = xxh64(h₁_bytes ‖ token_ids[512:768])
  ...
Block n:  hₙ = xxh64(hₙ₋₁_bytes ‖ token_ids[n*256:(n+1)*256])
```

这意味着 Block n 的哈希隐式编码了 Block 0 到 Block n 的全部 token 内容。两条请求只有在前 (n+1)*256 个 token 完全一致时，前 n+1 个 Block 的哈希才会相同。这正好对应了 KV Cache 可以复用的条件。

选择 xxhash-64 而非其他方案的原因：
- **速度**：xxhash 是最快的非加密哈希之一，2048 字节的输入约 70ns，适合在每个 postprocess step 中调用
- **链式 API**：`h.update()` 支持增量输入，天然适合"前序哈希 + 当前内容"的链式构造
- **64 位输出**：碰撞概率约 1/2⁶⁴，对于几千个 Block 的场景可以忽略

### 2.3 Prefix Cache 的三步流程

#### 步骤一：注册——hash_blocks

每一步 inference 结束后，Scheduler 在 `postprocess` 中调用 `hash_blocks`（`block_manager.py:110-120`）：

```python
def hash_blocks(self, seq: Sequence):
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
    if start == end: return   # 没有新的完整 block
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block.update(h, token_ids)           # 将 hash 和 token_ids 写入 Block 元数据
        self.hash_to_block_id[h] = block.block_id   # 注册到全局索引
```

关键细节：
- **时序**：`hash_blocks` 在 `num_cached_tokens` 更新 **之前** 被调用（`scheduler.py:83-84`），所以它能准确知道哪些 Block 是本轮新完成的
- **只哈希完整 Block**：`end` 通过整除 `block_size` 计算，自动排除了最后一个未满的 Block。未满的 Block 不能被缓存——因为后续 token 还会改变它的 KV 值
- **链式恢复**：`start > 0` 时，从前一个 Block 的 `hash` 字段恢复链式哈希的上下文。这使得 chunked prefill 跨多步也能正确构建哈希链

#### 步骤二：查询——can_allocate

当新请求到达时，Scheduler 调用 `can_allocate`（`block_manager.py:58-73`）做一次"dry run"——不修改任何状态，只返回能命中多少个 Prefix Cache Block：

```python
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):       # 注意：不检查最后一个 block
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break                              # 链断了，后续 block 也不可能命中
        num_cached_blocks += 1
        if block_id in self.used_block_ids:    # 正在使用的块不占用 free 配额
            num_new_blocks -= 1
    if len(self.free_block_ids) < num_new_blocks:
        return -1                              # free 块不够
    return num_cached_blocks
```

这里有一个极其重要的细节在第 69-70 行：**只有当缓存命中的 Block 已经在 `used_block_ids` 中（另一条序列正在使用它），才减少 `num_new_blocks`。** 如果命中的 Block 处于 EVICTED-WITH-HASH 状态（在 free list 中但 hash 仍有效），它 **会** 消耗一个 free block 配额（因为 `allocate` 时需要把它从 free list 中取出）。这是一个精确的容量核算——保证了 `allocate` 阶段不会出现 free block 不足的情况。

第 66 行的 `token_ids` 比较是碰撞防护：即使 xxhash-64 碰撞概率极低（~1/2⁶⁴），仍然用完整的 token 内容验证。如果碰撞发生，链会在此断开，后续 Block 全部视为未命中——安全但可能损失一些缓存收益。

#### 步骤三：分配——allocate

确认可以分配后，Scheduler 调用 `allocate`（`block_manager.py:75-92`）真正执行分配：

```python
def allocate(self, seq: Sequence, num_cached_blocks: int):
    assert not seq.block_table
    h = -1
    for i in range(num_cached_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1                    # 共享：引用计数加一
        else:
            block.ref_count = 1                     # 复活：从 free list 中取出
            self.free_block_ids.remove(block_id)    # ⚠️ O(n) 操作
            self.used_block_ids.add(block_id)
        seq.block_table.append(block_id)
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())   # 新块：从 free list 前端取
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

两条路径：
1. **共享路径**（Block 在 `used_block_ids` 中）：只需 `ref_count += 1`，O(1)。多条序列共享同一个物理 Block
2. **复活路径**（Block 在 `free_block_ids` 中，EVICTED-WITH-HASH 状态）：设 `ref_count = 1`，但需要 `self.free_block_ids.remove(block_id)` 从 deque 中间删除——这是 **O(n)** 操作。不调用 `reset()`，保留 hash 和 token_ids（它们仍然有效）

第 87 行的 `deque.remove()` 是当前实现中唯一的性能隐患。在有数千个 Block 且 Prefix Cache 命中率高的场景下，每次复活一个 Block 都要线性扫描 free list。更高效的实现可以使用带索引的双向链表实现 O(1) 按值删除。不过，考虑到 GPU 端的 kernel 执行时间远大于这个 CPU 开销，实际影响有限。

最后第 92 行设置 `seq.num_cached_tokens`，使得后续 `prepare_prefill` 只处理未缓存的 token。

### 2.4 关键细节与误区澄清

**误区一：`can_allocate` 会修改 BlockManager 的状态。**

不会。`can_allocate` 是纯只读操作——它重新计算哈希、查字典、检查集合，但不修改 `free_block_ids`、`used_block_ids`、`hash_to_block_id` 或任何 Block 的字段。它的返回值（缓存命中的 Block 数量）被传给 `allocate` 使用。

**误区二：最后一个 Block 也会被哈希和缓存。**

不会。`can_allocate` 的循环范围是 `range(seq.num_blocks - 1)`（第 62 行），`hash_blocks` 的 `end` 用整除计算（第 112 行），两者都排除了最后一个不完整的 Block。这是正确的——未满的 Block 后续还会追加 token，其 KV 值会发生变化，不能缓存。

**误区三：一个 Block 不可能同时出现在 `free_block_ids` 和 `hash_to_block_id` 中。**

可以，而且这正是 Prefix Caching 的核心机制。EVICTED-WITH-HASH 状态的 Block 同时存在于两个数据结构中。这种"双重身份"让被释放的 Block 有机会被复活，而不需要重新计算 KV Cache。

**误区四：`compute_hash` 重复计算了——`can_allocate` 算一遍，`allocate` 又算一遍。**

确实如此（`can_allocate` 的第 64 行和 `allocate` 的第 80 行都调用了 `compute_hash`）。这是用计算换简洁性——`can_allocate` 不能返回中间哈希值（因为循环可能提前 break），要共享哈希值需要更复杂的接口。考虑到 xxhash 的速度极快（256 token → ~70ns），这个冗余完全可以接受。

### 2.5 💡 小结

- Prefix Caching 用 xxhash-64 链式哈希实现了内容寻址的 Block 复用
- 哈希链的构造保证了只有"相同位置 + 相同前序内容"的 Block 才能匹配
- 三步流程：`hash_blocks` 注册（postprocess 时）→ `can_allocate` 查询（调度时）→ `allocate` 分配（确认后）
- EVICTED-WITH-HASH 状态是"幽灵缓存"的核心——被释放的 Block 保留身份信息，等待被复活或被覆盖

---

## 三、Scheduler 与 BlockManager 的协作：分配、追加、抢占

### 3.1 设计哲学与核心问题

Scheduler 是 BlockManager 的唯一消费者。它在 nano-vllm 中只有一个实例，在每一步 inference（`LLMEngine.step`）中执行三个阶段：`schedule()`（分配 Block、选择序列）→ GPU 前向 → `postprocess()`（注册哈希、释放已完成序列的 Block）。

核心问题是：**当多条序列竞争有限的 KV Cache Block 时，如何决定谁能运行、谁被暂停？**

### 3.2 两阶段调度：Prefill 优先

`schedule()` 方法（`scheduler.py:25-73`）采用严格的两阶段设计：

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []
    num_batched_tokens = 0

    # 第一阶段：Prefill
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.waiting[0]
        # ... 尝试为 waiting 序列分配 Block 并调度 prefill
    if scheduled_seqs:
        return scheduled_seqs, True   # prefill 阶段有调度结果就直接返回

    # 第二阶段：Decode（只有 prefill 阶段无结果时才进入）
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        # ... 为 running 序列分配追加 Block 并调度 decode
    return scheduled_seqs, False
```

**Prefill 优先** 是关键策略：如果有 waiting 序列可以被调度，decode 阶段完全不执行。这意味着一个调度步只会包含纯 prefill 或纯 decode 的序列，永远不会混合。这简化了 ModelRunner 的 `prepare_prefill` / `prepare_decode` 路径切换。

### 3.3 Prefill 阶段的 Block 分配

Prefill 阶段的核心逻辑（`scheduler.py:30-52`）：

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0:
        break
    if not seq.block_table:                              # 首次遇到该序列
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1:
            break                                        # Block 不够，停止调度
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
    else:                                                # Chunked prefill 续接
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    if remaining < num_tokens and scheduled_seqs:        # ⚠️ 只允许第一条序列 chunked
        break
    if not seq.block_table:
        self.block_manager.allocate(seq, num_cached_blocks)
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    num_batched_tokens += seq.num_scheduled_tokens
    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
        seq.status = SequenceStatus.RUNNING              # 全部 prefill 完成
        self.waiting.popleft()
        self.running.append(seq)
    scheduled_seqs.append(seq)
```

第 42-43 行的 `if remaining < num_tokens and scheduled_seqs` 是一个重要的约束：**只有 batch 中的第一条序列可以使用 chunked prefill。** 后续序列如果无法完整 prefill，直接跳过。这大幅简化了实现——chunked prefill 需要跨步追踪部分进度，只允许一条序列减少了状态管理的复杂度。

### 3.4 Decode 阶段的 Block 追加与抢占

Decode 阶段的逻辑（`scheduler.py:58-73`）：

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())   # LIFO 抢占：从队尾驱逐
        else:
            self.preempt(seq)                  # 自我抢占：所有序列都不行
            break
    else:
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
```

`can_append`（`block_manager.py:103-104`）是一个巧妙的布尔运算：

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

`(len(seq) % self.block_size == 1)` 是一个布尔表达式，Python 中 `True = 1`、`False = 0`。当序列长度对 block_size 取模等于 1 时，说明上一步刚好填满了一个 Block，需要分配新 Block。此时表达式变成 `len(free_block_ids) >= 1`。否则当前 Block 还有空间，表达式变成 `len(free_block_ids) >= 0`（恒真）。

这个写法虽然紧凑，但容易误读。展开后等价于：

```python
needs_new_block = (len(seq) % self.block_size == 1)
return not needs_new_block or len(self.free_block_ids) >= 1
```

### 3.5 抢占机制：Recomputation 策略

当 decode 阶段发现没有足够的 free block 给当前序列追加时，进入抢占（`scheduler.py:75-79`）：

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)       # 插入 waiting 队列头部
```

抢占是 **LIFO** 的——从 running 队列的尾部（`self.running.pop()`）取序列驱逐。越晚加入的序列越先被驱逐，保护了已经做了更多工作的老序列。被抢占的序列：

1. 所有 Block 被释放（`deallocate`），ref_count 递减，降为 0 的 Block 进入 free list
2. 状态重置为 WAITING + is_prefill = True
3. 被插入 waiting 队列的 **头部**（`appendleft`），获得最高调度优先级

`deallocate`（`block_manager.py:94-101`）按 **逆序** 遍历 block_table：

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

逆序遍历是一个微妙的优化：后面的 Block（序列末尾，最不可能被共享）先进入 free list 的尾部，前面的 Block（共享前缀，更可能被复用）后进入尾部。由于 `_allocate_block` 从 free list 的 **头部** 取 Block，末尾的 Block 存活时间更长。结合 FIFO 策略，前缀 Block 获得了最大的"幽灵缓存"存活窗口。

nano-vllm 只支持 **Recomputation** 抢占策略（不支持 vLLM 的 Swap-to-CPU）。被抢占的序列会丢失所有 KV Cache，下次调度时必须从头 prefill。但如果前缀 Block 仍在幽灵缓存中（hash 未被覆盖），Prefix Caching 会跳过前缀的 prefill 计算。

### 3.6 Postprocess：哈希注册与序列终结

`postprocess`（`scheduler.py:81-93`）在每步 inference 后运行：

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)                # ① 先注册哈希
        seq.num_cached_tokens += seq.num_scheduled_tokens  # ② 再更新进度
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue   # Chunked prefill 未完成，不追加 token
        seq.append_token(token_id)                         # ③ 追加新 token
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)              # ④ 释放所有 Block
            self.running.remove(seq)
```

注意 ① 和 ② 的顺序：`hash_blocks` 必须在 `num_cached_tokens` 更新之前调用，因为它需要用旧的 `num_cached_tokens` 来确定哪些 Block 是本轮新完成的。

### 3.7 关键细节与误区澄清

**误区：`while ... else` 是多余的语法。**

第 60 行的 `while not self.block_manager.can_append(seq):` 后跟 `else:` 块。Python 中 `while...else` 的 `else` 块只在循环条件自然变为 False 时执行，通过 `break` 退出时不执行。这意味着如果当前序列自我抢占（第 64 行 `break`），`else` 块不会执行，序列不会被加入 `scheduled_seqs`。这是一个 Python 的冷门但正确的语法用法。

### 3.8 💡 小结

- Scheduler 采用两阶段调度：Prefill 优先，Decode 其次，不混合
- Chunked prefill 只允许 batch 中的第一条序列使用，控制复杂度
- 抢占是 LIFO 的（优先驱逐最新的序列），被抢占序列进入 waiting 队列头部
- `deallocate` 的逆序遍历 + free list 的 FIFO 策略 = 前缀 Block 获得最长的幽灵缓存生存时间
- nano-vllm 只支持 Recomputation 抢占，不支持 Swap-to-CPU

---

## 四、从 block_id 到物理 KV Cache：Slot Mapping 与 Triton Kernel

### 4.1 设计哲学与核心问题

BlockManager 输出的是 `block_table`——一个整数列表。GPU 上的 Attention 层需要知道的是：**每个 token 的 Key 和 Value 应该写到 KV Cache 张量的哪个位置？已经缓存的 Key 和 Value 应该从哪里读？**

这个从逻辑块到物理地址的翻译，分为两步：
1. **ModelRunner**：将 `block_table` 转换为 `slot_mapping`（一个一维整数张量），并构造 `block_tables`（一个二维整数张量）传给 Flash Attention
2. **Triton Kernel / Flash Attention**：用 `slot_mapping` 写 KV Cache，用 `block_tables` 读 KV Cache

### 4.2 KV Cache 张量的物理布局

`allocate_kv_cache`（`model_runner.py:103-121`）分配全局 KV Cache 张量：

```python
self.kv_cache = torch.empty(
    2,                        # [0]=K, [1]=V
    hf_config.num_hidden_layers,   # 每层一个切片
    config.num_kvcache_blocks,     # Block 数量
    self.block_size,               # 256 tokens per block
    num_kv_heads,                  # KV Head 数（已除以 TP size）
    head_dim                       # 每个 Head 的维度
)
```

每个 Attention 层获得自己的切片：

```python
module.k_cache = self.kv_cache[0, layer_id]  # shape: [num_blocks, 256, num_kv_heads, head_dim]
module.v_cache = self.kv_cache[1, layer_id]
```

当 Triton Kernel 需要写入 block_id=7、offset=4 位置的 KV 值时，它访问的是 `k_cache[7][4]`。但 Kernel 不使用多维索引——它将 KV Cache 视为一个展平的 `[num_blocks * block_size, num_kv_heads * head_dim]` 的二维张量，用 `slot * D` 作为偏移量。

### 4.3 Slot Mapping 的计算

**"slot" 是一个展平的一维索引**：`slot = block_id * block_size + offset_in_block`。

以一个具体例子说明：

```
序列 A: block_table = [3, 7, 1]
序列 A 的 token[260]:
  → block index = 260 // 256 = 1
  → block_id = block_table[1] = 7
  → offset = 260 % 256 = 4
  → slot = 7 * 256 + 4 = 1796
```

#### Prefill 的 slot_mapping 计算

`prepare_prefill`（`model_runner.py:129-170`）为每个需要计算 KV 的 token 生成一个 slot：

```python
start = seq.num_cached_tokens
seqlen_q = seq.num_scheduled_tokens
end = start + seqlen_q
# ...
start_block = start // self.block_size
end_block = (end + self.block_size - 1) // self.block_size
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * self.block_size
    if i == start_block:
        slot_start += start % self.block_size    # 起始块内偏移
    if i != end_block - 1:
        slot_end = seq.block_table[i] * self.block_size + self.block_size
    else:
        slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
    slot_mapping.extend(range(slot_start, slot_end))
```

每个 slot 对应一个 token 的 KV 写入位置。slot_mapping 的长度等于本次 prefill 的 token 总数。

当 Prefix Cache 命中时（`num_cached_tokens > 0`），`start` 不为零，slot_mapping 只覆盖未缓存的部分。已缓存的前缀 Block 的 KV 值已经在 GPU 内存中了（上一次 prefill 时写入的，且物理块没有被覆盖），无需重写。

#### Decode 的 slot_mapping 计算

`prepare_decode`（`model_runner.py:172-188`）更简单——每条序列只有一个新 token：

```python
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

`seq.block_table[-1]` 是最后一个 Block，`seq.last_block_num_tokens - 1` 是最后一个 token 在该 Block 内的偏移（0-based）。

### 4.4 Triton Kernel：store_kvcache_kernel

`attention.py:10-30` 定义了一个 Triton kernel，负责将模型前向计算产出的 K/V 张量写入分页 KV Cache：

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    D: tl.constexpr,          # D = num_kv_heads * head_dim
):
    idx = tl.program_id(0)     # 每个 program instance 处理一个 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return      # CUDA Graph 模式下的 padding 哨兵
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

核心映射：`cache_offsets = slot * D`。由于 `k_cache` 的物理布局是 `[num_blocks, block_size, num_kv_heads, head_dim]`，且张量是行主序连续的，`slot * D` 正好等价于 `k_cache[block_id, offset_in_block].flatten()` 的起始偏移。

Kernel 对每个 token 读 2*D 个元素（K 和 V），写 2*D 个元素到 cache。对于 Qwen3-0.6B（D = 4 * 128 = 512，bf16），每个 token 读写 2 * 512 * 2 = 2048 字节。

### 4.5 Flash Attention 的 Block Table 读取

Attention 层（`attention.py:59-75`）根据 prefill/decode 模式选择不同的 Flash Attention 接口：

```python
def forward(self, q, k, v):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # 先写入 cache
    if context.is_prefill:
        if context.block_tables is not None:    # Prefix Cache 生效
            k, v = k_cache, v_cache             # 用整个 cache 替代 freshly computed K/V
        o = flash_attn_varlen_func(q, k, v, ..., block_table=context.block_tables)
    else:
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                    cache_seqlens=context.context_lens,
                                    block_table=context.block_tables, ...)
```

**Prefill + Prefix Cache**（第 65-66 行）：当 `block_tables` 不为 None 时（`prepare_prefill` 在第 162-163 行检测到 `cu_seqlens_k > cu_seqlens_q`），表明有缓存前缀。此时将 K/V 替换为完整的 `k_cache`/`v_cache`，让 `flash_attn_varlen_func` 通过 block_table 在分页 cache 中读取已缓存的 KV 值。当前 chunk 新计算的 KV 已经在第 63 行被 `store_kvcache` 写入了 cache。

**Decode**（第 72-74 行）：`flash_attn_with_kvcache` 始终使用分页模式。`cache_seqlens` 告诉 Flash Attention 每条序列在 cache 中有多少有效 token，`block_table` 告诉它这些 token 分布在哪些物理 Block 中。

### 4.6 完整数据流图

```
BlockManager (CPU)                   ModelRunner (CPU→GPU)              Attention (GPU)
═══════════════════                  ═══════════════════               ═══════════════

allocate()                           prepare_prefill/decode()          store_kvcache_kernel()
  │                                    │                                │
  │ 输出 block_table                   │ block_table → slot_mapping     │ slot_mapping[idx] → slot
  │ = [blk3, blk7, blk1]             │ = [768,769,...,1796,...,260]    │ cache_offsets = slot * D
  │                                    │                                │ 写入 k_cache, v_cache
  └──→ seq.block_table ──────────→    slot_mapping (GPU tensor) ──→   Triton Kernel
                                       │
                                       │ block_table → block_tables
                                       │ = [[3,7,1], [5,2,-1]] (padded)
                                       │
                                       └──→ block_tables (GPU tensor) ──→ flash_attn_*()
                                                                           读取 k_cache, v_cache
```

### 4.7 关键细节与误区澄清

**误区：BlockManager 直接操作 GPU 显存。**

不是。BlockManager 的代码中没有任何 `torch` 导入，也不持有 GPU 张量。block_id 只是一个整数，它的物理含义（对应 `kv_cache[:, :, block_id, :, :, :]`）是由 ModelRunner 在 `allocate_kv_cache` 中建立的。两者通过 Sequence.block_table 这个整数列表间接关联。

**误区：CUDA Graph 模式下 slot_mapping 每步都重新分配。**

不是。CUDA Graph 要求输入张量的地址不变。`model_runner.py:250` 中 `graph_vars` 预分配了 `slot_mapping` 张量，之后每步只修改其内容（第 206-207 行 `fill_(-1)` 然后赋值）。`-1` 是哨兵值，Triton Kernel 中 `if slot == -1: return` 跳过这些无效位置。

### 4.8 💡 小结

- slot = block_id * block_size + offset_in_block，是逻辑块到物理 KV Cache 位置的一维映射
- Triton Kernel 每 token 一个 program instance，用 slot * D 计算 cache 偏移
- Prefix Cache 生效时，Attention 层将 K/V 替换为 k_cache/v_cache，让 Flash Attention 通过 block_table 读取已缓存数据
- CUDA Graph 模式下用 -1 哨兵值 padding slot_mapping，避免越界写入

---

## 五、完整主路径串联

### 5.1 完整调用栈

以 `LLMEngine.step()`（`llm_engine.py:49-55`）为入口，一次 inference step 的完整调用栈：

```
LLMEngine.step()
  │
  ├─ Step 1: scheduler.schedule()                              [CPU]
  │     │
  │     ├─ Prefill 阶段:
  │     │     ├─ block_manager.can_allocate(seq)               查询 Prefix Cache + 容量检查
  │     │     ├─ block_manager.allocate(seq, num_cached)       分配 Block（缓存复用 / 新分配）
  │     │     └─ 设置 seq.num_scheduled_tokens
  │     │
  │     └─ Decode 阶段:
  │           ├─ block_manager.can_append(seq)                 检查是否需要新 Block
  │           ├─ block_manager.may_append(seq)                 按需分配新 Block
  │           └─ [可能] preempt → block_manager.deallocate     抢占释放 Block
  │
  ├─ Step 2: model_runner.call("run", seqs, is_prefill)        [CPU + GPU]
  │     │
  │     ├─ prepare_prefill / prepare_decode                    block_table → slot_mapping + block_tables
  │     ├─ set_context(...)                                    设置全局 Context
  │     ├─ run_model(input_ids, positions, is_prefill)         [GPU]
  │     │     ├─ model.forward()
  │     │     │     └─ 每层 Attention:
  │     │     │           ├─ store_kvcache(k, v, ...)          Triton Kernel 写 KV Cache
  │     │     │           └─ flash_attn_*(...)                 读 KV Cache 计算 Attention
  │     │     └─ model.compute_logits()
  │     ├─ sampler(logits, temperatures)                       采样生成 token
  │     └─ reset_context()
  │
  └─ Step 3: scheduler.postprocess(seqs, token_ids, is_prefill) [CPU]
        ├─ block_manager.hash_blocks(seq)                      注册新完成的 Block 到 Prefix Cache
        ├─ 更新 seq.num_cached_tokens
        ├─ seq.append_token(token_id)                          追加新 token
        └─ [可能] block_manager.deallocate(seq)                序列完成，释放所有 Block
```

### 5.2 每一层做了什么

| 阶段 | 输入 | 输出 / 状态变化 | 是否触发 GPU 通信 | 每步都执行？ |
|---|---|---|---|---|
| `can_allocate` | seq.token_ids | 返回 num_cached_blocks 或 -1 | 否 | 仅 prefill 首次 |
| `allocate` | seq, num_cached_blocks | 填充 seq.block_table | 否 | 仅 prefill 首次 |
| `can_append` | seq | 返回 bool | 否 | 每步 decode |
| `may_append` | seq | 可能追加 block_table 元素 | 否 | 每步 decode |
| `prepare_*` | seqs | slot_mapping, block_tables 张量 | pin_memory→GPU 异步传输 | 每步 |
| `store_kvcache` | k, v, slot_mapping | 写入 kv_cache 张量 | 否（单卡本地写入） | 每步每层 |
| `flash_attn_*` | q, kv_cache, block_tables | attention 输出 | 否（单卡本地读取） | 每步每层 |
| `hash_blocks` | seq | Block.hash, hash_to_block_id 更新 | 否 | 每步（通常无操作） |
| `deallocate` | seq | 释放 Block，清空 block_table | 否 | 序列完成或被抢占 |

### 5.3 哪些逻辑不在主路径

| 看似相关的逻辑 | 是否在主流程 | 正确理解 |
|---|---|---|
| `Block.__init__` | 仅初始化时 | 只在 `BlockManager.__init__` 中批量调用一次 |
| `Block.update` | 间接在主流程 | 只在 `hash_blocks` 中被调用，且仅当有新的完整 Block 时 |
| `Block.reset` | 间接在主流程 | 只在 `_allocate_block` 中调用（分配新块时） |
| `compute_hash` | 在主流程但不频繁 | 只在 `can_allocate`、`allocate`、`hash_blocks` 中调用 |
| `warmup_model` | 仅初始化时 | 用 dummy 输入预热模型，确定 peak 显存 |
| `allocate_kv_cache` | 仅初始化时 | 根据剩余显存计算 num_blocks 并分配 KV Cache 张量 |
| `capture_cudagraph` | 仅初始化时 | 为不同 batch size 捕获 CUDA Graph |

---

## 六、核心机制深挖

### 6.1 Free List 的 FIFO 策略：为什么用 deque 而不是 set？

`free_block_ids` 使用 `deque`，分配时 `popleft()`（取最老的），回收时 `append()`（放到最新的位置）。

如果改用 `set`，分配时 `pop()` 获取的是任意元素（CPython 实现中通常是最小的）。这会导致刚释放的 Block 可能立即被重新分配给不同内容，摧毁其 Prefix Cache 价值。

FIFO 策略的效果类似 LRU 替换：
- 最老的 free block（在 deque 头部）最先被回收——它在 free list 中存活时间最长，cache 被命中的可能性最低
- 最新释放的 block（在 deque 尾部）最后被回收——给它最大的窗口等待潜在的 cache hit

这是一种零额外开销的 cache-friendly 设计：deque 的 popleft/append 都是 O(1)，不需要额外的 LRU 链表或时间戳。

### 6.2 引用计数：共享块的正确性保证

多条序列可以通过 Prefix Caching 共享同一个物理 Block。`ref_count` 确保共享块在所有引用者释放后才回收。

考虑以下场景：

```
Time 0: 序列 A 到达，token_ids = [t0...t511]（2 blocks）
        allocate(A, 0)：block_table = [10, 11]，两者 ref_count = 1
        postprocess: hash_blocks 为 block 10 注册 h0

Time 1: 序列 B 到达，token_ids = [t0...t511, t512...t600]（3 blocks，前 2 共享）
        can_allocate(B): 找到 h0 匹配 block 10 → num_cached_blocks = 1
        allocate(B, 1): block 10 ref_count → 2, block 12 新分配
        B.block_table = [10, 12]

Time 2: 序列 A 完成
        deallocate(A): block 11 ref_count → 0 (释放), block 10 ref_count → 1 (保留)
        ✅ block 10 没有被释放，因为 B 还在用

Time 3: 序列 B 完成
        deallocate(B): block 12 ref_count → 0 (释放), block 10 ref_count → 0 (释放)
        block 10 进入 EVICTED-WITH-HASH 状态
```

### 6.3 Context 全局变量：隐式参数传递的利弊

`set_context` / `get_context`（`context.py`）用模块级全局变量将 slot_mapping、block_tables 等张量传递给 Attention 层。这避免了修改模型的 `forward` 签名（Qwen3 模型不需要接收 block_table 参数），但也引入了隐式状态。

```python
_CONTEXT = Context()   # 模块级全局变量

def set_context(is_prefill, ...):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, ...)

def get_context():
    return _CONTEXT
```

**安全性**：每个进程有独立的 `_CONTEXT`（Python 模块级变量是进程隔离的），且推理是单线程的，所以没有线程安全问题。`reset_context()` 在每步 `run()` 结束后被调用（`model_runner.py:219`），防止 stale context 泄漏。

**代价**：代码可读性降低——Attention 层的行为取决于一个在调用栈外部设置的全局变量。如果将来引入异步推理或流水线并行，这个设计会成为障碍。

### 6.4 num_kvcache_blocks 的动态计算：显存预算公式

`allocate_kv_cache`（`model_runner.py:103-115`）的显存预算公式值得拆解：

```python
free, total = torch.cuda.mem_get_info()
used = total - free
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
```

拆解含义：
- `used - current`：非 PyTorch 的 GPU 开销（CUDA context、driver 等），约 1-2 GiB
- `peak - current`：warmup 期间的峰值激活值内存（已被 empty_cache 释放但代表运行时需求）
- `total * 0.9 - (used - current) - peak`：KV Cache 的可用预算

这个公式的前提是 `warmup_model()` 已经执行过（`model_runner.py:91-101`），用最大 batch size 跑了一次前向以记录 peak。这确保了 KV Cache 的分配不会侵占激活值需要的显存。

---

## 七、显存、性能与通信分析

### 7.1 显存收益范围

| 内容 | 是否由分页管理节省 | 原因 |
|---|---|---|
| KV Cache | ✅ 核心收益 | 按需分配 Block，避免预分配最大长度的浪费 |
| 模型参数 | ❌ | BlockManager 不涉及参数管理 |
| 激活值 | ❌ | 前向计算的中间张量独立于 KV Cache |
| Logits / 采样 | ❌ | 在 KV Cache 之外的独立张量 |
| Prefix Cache 节省 | ✅ 间接收益 | 共享 Block 避免重复存储相同前缀的 KV 值 |

**真正的显存大头是 KV Cache 本身**。以 Qwen3-0.6B（28 层、4 KV Head、128 Head Dim、bf16）为例：

```
单个 Block 的 KV 字节数：
  = 2 (K+V) × 28 (层) × 256 (tokens) × 4 (KV heads) × 128 (head_dim) × 2 (bf16)
  = 14,680,064 字节 ≈ 14.0 MiB

在 80GB A100 上（90% 利用率）：
  可用 KV Cache 预算 ≈ 65-68 GiB
  Block 数量 ≈ 65 GiB / 14 MiB ≈ 4,750 blocks
  总 token 容量 ≈ 4,750 × 256 ≈ 1,216,000 tokens
```

### 7.2 内部碎片分析

block_size = 256 带来的内部碎片：

- 每条序列的最后一个 Block 平均浪费 127.5 个 slot
- 每个浪费 slot 的 KV 字节数 = 2 × 28 × 4 × 128 × 2 = 57,344 字节 ≈ 56 KiB
- 每条序列的平均碎片 = 127.5 × 56 KiB ≈ 7.0 MiB
- 512 条并发序列的总碎片 ≈ 3.5 GiB

对于 80GB A100 上 65 GiB 的 KV Cache 预算，3.5 GiB 碎片约占 5.4%——可以接受但并不小。对于短序列（如 100 token），碎片率更严重：每条只用 1 个 Block（256 slot），实际使用 100 slot，碎片率 61%。

### 7.3 Prefix Cache 的性能收益

假设 1000 条请求共享 2048 token 的 System Prompt：

- 共享前缀 = 8 个完整 Block（2048 / 256）
- 第 1 条请求：需要完整 prefill 2048 token
- 第 2-1000 条请求：Prefix Cache 命中 8 Block，跳过 2048 token 的 prefill

节省的计算量 = 999 × 2048 token 的 Transformer 前向，这在长 prompt 场景下是巨大的吞吐量提升。

### 7.4 通信开销

| 通信类型 | 发生场景 | 频率 | 涉及组 |
|---|---|---|---|
| NCCL all_reduce | RowParallelLinear 前向 | 每步每层 | TP group |
| NCCL all_gather | ParallelLMHead | 每步 1 次 | TP group |
| Shared Memory 序列化 | Rank 0 → Worker 传递 seqs | 每步 1 次 | 所有 rank |

**BlockManager 本身不触发任何通信**。它运行在 Rank 0 进程中，通过 Shared Memory 将序列元数据（包括 block_table）序列化后传给 Worker 进程。Worker 进程收到 block_table 后独立计算 slot_mapping 和 block_tables 张量。

KV Cache 写入（store_kvcache Triton Kernel）是纯本地 GPU 操作。每个 Rank 的 KV Cache 是独立的（num_kv_heads 已经按 TP size 划分），不需要跨 Rank 同步。

### 7.5 性能取舍总结

| 取舍 | 收益 | 代价 |
|---|---|---|
| 分页分配 vs 预分配 | 消除外部碎片，按需使用显存 | Block 粒度的内部碎片（~5%） |
| block_size=256 vs 更小值 | 更少的元数据开销，更小的 block_table 张量 | 更高的内部碎片，更粗的 Prefix Cache 粒度 |
| Recomputation 抢占 vs Swap | 实现简单，无 CPU 显存管理 | 被抢占序列需要完全重新 prefill |
| FIFO free list vs LRU | O(1) 分配/回收，无额外数据结构 | 不是真正的 LRU，长时间未用的 block 可能不会被优先回收 |
| deque.remove() for 复活 | 实现简单 | O(n) 线性扫描 |

---

## 八、配置项、边界条件与坑点

| 配置项 | 影响的源码路径 | 行为变化 | 风险/坑点 |
|---|---|---|---|
| `kvcache_block_size` (默认 256) | `config.py:22` 校验，`block_manager.py:29` 构造 | 改变分配粒度、碎片率、Prefix Cache 匹配粒度 | 必须是 256 的倍数，否则 assert 失败 |
| `gpu_memory_utilization` (默认 0.9) | `model_runner.py:113` 计算 num_blocks | 利用率越高 → blocks 越多 → 吞吐越高 | 过高可能导致 OOM（峰值激活超出预留空间） |
| `max_num_seqs` (默认 512) | `scheduler.py:30`/`58` 控制 batch 上限 | 限制同时调度的最大序列数 | 越大 → 内部碎片总量越大（每条序列浪费一个 Block） |
| `max_num_batched_tokens` (默认 16384) | `scheduler.py:32` 控制 prefill 批次大小 | 限制单次 prefill 的 token 总数 | 过小 → prefill 被过度分片 → 更多调度开销 |
| `max_model_len` (默认 4096) | `model_runner.py:227` CUDA Graph 的 max_num_blocks | 限制序列的最大长度 | 越大 → CUDA Graph 的 block_tables 预分配越大 |
| `enforce_eager` (默认 False) | `model_runner.py:36-37` | True 时禁用 CUDA Graph | 调试有用，但 decode 性能下降 |

### 边界条件与坑点

1. **Block 数量为 0**：如果 GPU 显存在分配模型后所剩无几，`num_kvcache_blocks` 可能为 0 或负数，触发 `assert config.num_kvcache_blocks > 0`（`model_runner.py:114`）。没有友好的错误提示。

2. **单条序列超过 Block 容量**：如果一条序列的 `num_blocks` 超过 `num_kvcache_blocks`，`can_allocate` 会返回 -1，该序列永远无法被调度。Scheduler 不会报错——它只是无限等待。

3. **抢占死循环风险**：理论上，如果所有 running 序列都自我抢占，`scheduler.py:71` 的 `assert scheduled_seqs` 会触发。这只在 KV Cache 极度不足时发生（连一条序列都装不下新 Block）。

4. **Shared Memory 大小固定 1 MiB**：`model_runner.py:43` 硬编码 `size=2**20`。在极端情况下（大量序列 + 长 prompt），pickle 序列化后的数据可能超过 1 MiB，导致 buffer overflow。

5. **Sequence.block_size 是类变量**：`sequence.py:15` 定义 `block_size = 256`，`llm_engine.py:21` 用 `Sequence.block_size = config.kvcache_block_size` 覆盖。如果在 `LLMEngine.__init__` 之前创建了 Sequence 对象，它会使用错误的 block_size。

---

## 九、测试、示例与覆盖缺口

### 9.1 已覆盖路径

| 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `example.py` | 基本 prefill + decode，enforce_eager=True | 2 条短 prompt，单 GPU |
| `bench.py` | 256 条随机序列，CUDA Graph 模式 | 覆盖了 Block 分配/释放/追加的主路径 |

### 9.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| Prefix Cache 命中与复用 | ❌ 无 | bench.py 用随机 token，无法触发 Prefix Cache |
| 抢占与 re-prefill | ❌ 无（bench.py 的序列数/长度可能不触发抢占） | 抢占逻辑可能存在未发现的 bug |
| Chunked Prefill | ❌ 无显式测试 | 需要 prompt 超过 max_num_batched_tokens 才能触发 |
| 多 GPU Tensor Parallelism | ❌ 无自动化测试 | Shared Memory 序列化可能有边界问题 |
| Block 数量耗尽 | ❌ 无 | 极端场景下的 assert 失败无友好提示 |
| 长序列（接近 max_model_len） | ❌ 无 | CUDA Graph 的 block_tables 预分配是否足够 |
| hash 碰撞场景 | ❌ 无（概率太低，难以构造） | 碰撞时链断开，性能下降但不影响正确性 |
| 抢占后 Prefix Cache 复用 | ❌ 无 | 抢占释放 Block → 重新调度时是否正确命中幽灵缓存 |

**项目没有测试套件**（CLAUDE.md 明确指出 "There is no test suite or linter configured"）。所有行为验证依赖 `example.py` 和 `bench.py` 的手动执行。

---

## 十、局限性与已知优化点

### 10.1 硬约束

- **block_size 必须是 256 的倍数**：无法使用更小的块（如 vLLM 的默认 16），限制了短序列场景的内存效率
- **只支持 Qwen3 架构**：`model_runner.py:31` 硬编码 `Qwen3ForCausalLM`
- **不支持 Beam Search**：没有序列组、没有 fork、没有 copy-on-write Block 共享
- **不支持 Swap-to-CPU**：抢占只能 recompute，没有 CPU Block Allocator
- **TP size 必须整除 num_attention_heads 和 num_key_value_heads**
- **NCCL rendezvous 硬编码 `tcp://localhost:2333`**：不支持多机

### 10.2 维护成本

- **全局变量 `_CONTEXT`**：隐式传递状态，难以追踪数据流
- **`Sequence.block_size` 是类变量**：在 `LLMEngine.__init__` 中被覆盖，如果有多个 Engine 实例会冲突
- **`Sequence.counter = count()` 是类变量**：进程隔离安全，但测试中可能累加
- **无类型注解或接口定义**：BlockManager 和 Scheduler 之间的契约完全隐式

### 10.3 性能瓶颈

| 瓶颈 | 位置 | 严重程度 | 改进方向 |
|---|---|---|---|
| `deque.remove(block_id)` 线性扫描 | `block_manager.py:87` | 低（CPU 侧） | 改用带索引的双向链表 |
| `can_allocate` + `allocate` 重复哈希计算 | `block_manager.py:64` + `80` | 低 | 让 `can_allocate` 返回中间哈希值 |
| `np.array(token_ids).tobytes()` 每次创建临时数组 | `block_manager.py:40` | 低 | 缓存 block 的 bytes 表示 |
| `token_ids` 列表比较 O(256) | `block_manager.py:66` | 低 | 用 numpy 数组存储 + 向量化比较 |
| Prefill 和 Decode 不混合调度 | `scheduler.py:54` | 中 | 混合调度可提高 GPU 利用率 |
| 无 Swap 的抢占 | `scheduler.py:75` | 高（在高竞争场景） | 增加 CPU Block Allocator + swap 路径 |

### 10.4 已知优化方向

1. **更细粒度的 Block Size**：vLLM 默认 block_size=16，大幅减少内部碎片。nano-vllm 的 256 约束可能来自 Flash Attention 的 paged 模式要求

2. **Prefill-Decode 混合调度**：当 Prefill 批次未满时，可以插入 Decode 序列填充 GPU 空闲算力

3. **Swap-to-CPU 抢占**：将被抢占序列的 KV Cache Block 复制到 CPU 内存，恢复时再复制回来，避免重新 prefill

4. **Copy-on-Write**：当多条序列共享 Block 且其中一条要修改时（如 Beam Search 分叉），只复制被修改的 Block

5. **异步 Block Hashing**：将 `hash_blocks` 的哈希计算放到后台线程，不阻塞主调度循环

6. **Free List 索引优化**：将 `free_block_ids` 从 deque 改为 OrderedDict 或自定义索引链表，支持 O(1) 按值删除

---

## 小结与展望

nano-vllm 的 BlockManager 实现可以用几个关键词概括。

**关键词一：分页虚拟内存**

BlockManager 将操作系统的分页内存管理思想搬到了 GPU 的 KV Cache 管理上。block_id 是物理页帧号，block_table 是页表，slot_mapping 是从虚拟地址到物理地址的翻译结果。这套抽象让 KV Cache 的分配完全按需、无外部碎片，是整个系统吞吐量的基础。

**关键词二：幽灵缓存**

被释放的 Block 不会立即丧失身份——它的 hash 和 token_ids 保留在 free list 中，GPU 上的 KV 数据也原封不动。通过 FIFO 的 free list 策略，最近释放的 Block 获得了最长的"幽灵存活期"，等待潜在的 Prefix Cache 命中。这是一种零成本的缓存层——不需要额外的数据结构或 GPU 操作。

**关键词三：链式哈希寻址**

xxhash-64 的链式构造将"相同位置 + 相同内容 + 相同前序"这三个 KV Cache 可复用条件编码进了一个 64 位整数。这使得 Prefix Cache 的查找是 O(1) 的字典操作，验证是 O(block_size) 的列表比较。简洁且正确。

**关键词四：策略与机制分离**

BlockManager（策略：谁分配、谁释放、谁复用）完全运行在 CPU 上的纯 Python 数据结构中。ModelRunner + Triton Kernel（机制：slot 如何映射到 GPU 地址、KV 如何写入和读取）完全运行在 GPU 上。两者通过 Sequence.block_table 这个简单的整数列表桥接。这种分离使得 BlockManager 的逻辑可以独立理解、测试和优化。

**适用场景**：nano-vllm 的 BlockManager 实现适合单机推理、中等并发（~512 序列）、长前缀共享（如 System Prompt、Few-shot）的场景。它的极简设计使得源码可读性极高，非常适合作为 PagedAttention 的学习参考。

**不适合的场景**：高竞争/频繁抢占场景（Recomputation 策略代价高）、短序列密集场景（block_size=256 的碎片率高）、需要 Beam Search 的场景（无 Copy-on-Write）。

**与 vLLM 相比的取舍**：nano-vllm 用极致的简洁性（120 行 BlockManager vs vLLM 数千行的 block manager 体系）换来了功能的精简——没有 Swap、没有 CoW、没有可配置的 eviction policy、没有 GPU/CPU 双 allocator。但核心的分页管理 + Prefix Caching 思路是一致的。

**后续值得继续走读的方向**：
- ModelRunner 的 CUDA Graph 捕获机制（如何在固定形状约束下适配动态 batch）
- Tensor Parallelism 的 Shared Memory 通信协议（pickle 序列化 + Event 同步）
- Attention 层的 Flash Attention 变体选择（`flash_attn_varlen_func` vs `flash_attn_with_kvcache` 的适用条件）
