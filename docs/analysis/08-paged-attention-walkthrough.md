# nano-vllm 源码走读：PagedAttention 实现解析

> 在大语言模型推理场景中，KV Cache 是吞吐量的关键瓶颈——不是因为计算不够快，而是因为显存不够用。一个 4096 token 的序列在 28 层 Transformer 中需要为每个 token 存储 K 和 V 向量，显存消耗随并发请求数线性增长。传统做法是为每个序列预分配一段连续显存，但这会导致严重的内部碎片（预分配但未使用的空间）和外部碎片（请求结束后无法复用的间隙），原始论文（Kwon et al., 2023）的测量显示浪费率可达 60-80%。PagedAttention 正是为了解决这个问题而提出的。
>
> nano-vllm 用约 1200 行 Python 代码重新实现了 vLLM 的核心推理引擎，其中 PagedAttention 是最关键的特性。本文不展开 PagedAttention 的原始论文理论，而是聚焦源码，分析这个轻量框架如何用最小代码量把分页式 KV Cache 管理、前缀缓存、分块预填充、CUDA Graph 和张量并行完整地串联起来——以及这套设计带来了哪些收益和限制。

---

## 前言

### 业务 / 工程背景

PagedAttention 出现在 LLM 推理服务场景。当多个用户请求并发到达时，推理引擎需要为每个请求维护独立的 KV Cache，而显存总量是固定的。核心问题是：**如何在有限显存中服务尽可能多的并发请求，同时避免显存浪费？**

### 核心矛盾

nano-vllm 的 PagedAttention 实现面临三个层次的工程矛盾：

1. **显存效率 vs. 实现复杂度**：分页管理需要 block 分配、释放、引用计数、哈希缓存等完整的"虚拟内存"系统，但框架目标是保持 ~1200 行的极简实现。
2. **前缀复用 vs. 正确性保证**：自动前缀缓存（APC）允许不同请求共享相同前缀的 KV Cache block，但必须保证内容一致性——哈希碰撞会导致注意力计算使用错误的 KV 值，产生无声的推理错误。
3. **调度灵活性 vs. 显存压力**：调度器需要在 prefill（计算密集）和 decode（访存密集）之间做出选择，当显存不足时还需要抢占（preempt）正在运行的序列，但抢占意味着丢弃已计算的 KV Cache。

### 本文主线

本文按机制分章，而非按文件分章。共分为以下几个核心机制：

- **第一章**：分页式 KV Cache 的物理布局——显存是怎么组织的
- **第二章**：Block 生命周期管理——分配、释放与前缀缓存
- **第三章**：调度器与显存的耦合——prefill-first 策略与抢占
- **第四章**：地址翻译——slot_mapping 如何连接逻辑位置与物理缓存
- **第五章**：注意力计算——Triton 写入与 Flash Attention 读取
- **第六章**：CUDA Graph 与分页注意力的协作
- **第七章**：完整主路径串联
- **第八章**：显存、性能与通信分析
- **第九章**：配置项、边界条件与坑点
- **第十章**：局限性与已知优化点

### 不展开的内容

本文不讲 Flash Attention 的内部实现原理，不讲 Transformer 架构基础，不讲 Qwen3 模型结构细节。只讲 nano-vllm 如何把 PagedAttention 集成到推理链路中。

### 核心文件表

| 文件 | 职责 |
|---|---|
| `nanovllm/engine/block_manager.py` | Block 池管理：分配、释放、引用计数、xxhash 前缀缓存 |
| `nanovllm/engine/scheduler.py` | 两阶段调度：prefill/decode 切换、抢占、后处理 |
| `nanovllm/engine/model_runner.py` | KV Cache 物理分配、slot_mapping 计算、CUDA Graph 捕获与回放 |
| `nanovllm/engine/sequence.py` | 序列状态：block_table、缓存进度、序列化优化 |
| `nanovllm/layers/attention.py` | Triton 写缓存内核 + Flash Attention 两路径调用 |
| `nanovllm/utils/context.py` | 全局 Context 传递分页元数据（block_tables、slot_mapping） |
| `nanovllm/engine/llm_engine.py` | 推理主循环：schedule → run → postprocess |
| `nanovllm/config.py` | 配置定义：block_size、gpu_memory_utilization 等 |

---

## 一、KV Cache 物理布局：显存是怎么组织的

### 1.1 设计哲学与核心问题

在没有 PagedAttention 的传统实现中，每个序列的 KV Cache 是一段连续的显存。序列 A 用完释放后留下一个"空洞"，序列 B 如果比 A 长就无法填入——这就是外部碎片。即使序列 A 预分配了 4096 个 token 的空间但只用了 500 个，剩余 3596 个位置的显存也被白白占据——这是内部碎片。

PagedAttention 的核心思想借鉴了操作系统的虚拟内存：**把 KV Cache 切成固定大小的 block（类似内存页），按需分配，逻辑连续但物理不连续。** 这样，序列不需要连续显存，空闲 block 可以被任意序列复用，外部碎片降为零，内部碎片仅存在于每个序列的最后一个 block。

### 1.2 源码入口与关键对象

```
nanovllm/engine/model_runner.py
  - allocate_kv_cache()（第 103-121 行）：计算可用显存、分配物理缓存张量、绑定到注意力层
nanovllm/config.py
  - Config 数据类（第 7-25 行）：kvcache_block_size、num_kvcache_blocks、gpu_memory_utilization
```

### 1.3 主流程拆解

KV Cache 的物理分配发生在 `ModelRunner.__init__()` 中，在模型加载和 warmup 之后调用。让我们跟踪 `allocate_kv_cache()`（`model_runner.py:103-121`）：

**第一步：测量可用显存**（第 106-109 行）

```python
free, total = torch.cuda.mem_get_info()
used = total - free
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
```

这里有一个精巧的设计：`peak` 是 warmup 期间 PyTorch 分配器的峰值（包含了最大 batch 的激活值显存），`current` 是 warmup 后的持久分配（主要是模型权重）。差值 `peak - current` 就是推理过程中临时激活值需要的显存上界。warmup（第 91-101 行）会用最大 batch size 跑一次前向，确保 `peak` 捕获了最坏情况。

**第二步：计算 block 数量**（第 110-113 行）

```python
num_kv_heads = hf_config.num_key_value_heads // self.world_size
head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
```

公式的直觉是：

```
可用显存 = GPU总量 × 利用率 - 已用显存 - (峰值临时分配 - 当前持久分配)
block数 = 可用显存 / 每block字节数
```

注意 `num_kv_heads` 已经除以了 `world_size`——在张量并行场景下，每个 GPU 只存储自己负责的 KV head 分片，这使得每 GPU 的 block 更小，能容纳更多 block。

**第三步：分配物理张量并绑定到注意力层**（第 115-121 行）

```python
self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

整个 KV Cache 是一个 6 维张量，一次 `torch.empty` 调用完成分配，避免了碎片化的小分配。每个注意力层通过 view 获得自己那一"层"的缓存，形状为 `[num_blocks, block_size, num_kv_heads, head_dim]`。

用 ASCII 图来理解物理布局：

```
kv_cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
           │       │          │           │            │            │
           │       │          │           │            │            └── 每个 head 的维度
           │       │          │           │            └── KV head 数（已 TP 分片）
           │       │          │           └── 每个 block 内的 token 槽位数（默认 256）
           │       │          └── 物理 block 池的大小
           │       └── Transformer 层数
           └── 0=K, 1=V

单层单 block 视图:
  ┌─────────────────────────────────────────┐
  │ Block 0:  slot0 slot1 ... slot255       │  ← 256 个 token 位置
  │ Block 1:  slot0 slot1 ... slot255       │  
  │ Block 2:  slot0 slot1 ... slot255       │  
  │   ...                                   │
  │ Block N:  slot0 slot1 ... slot255       │  
  └─────────────────────────────────────────┘
    每个 slot 存储 [num_kv_heads, head_dim] 大小的 K 或 V 向量
```

### 1.4 关键细节与误区澄清

**误区：KV Cache 是按层分别分配的。** 实际上不是——它是一个单一的 6 维张量，各层通过 view（切片）共享同一块连续显存。这避免了多次小分配导致的 CUDA 内存碎片，也简化了显存管理。

**误区：`Attention` 层的 `k_cache`/`v_cache` 初始化为空张量意味着预填充时不写缓存。** 观察 `attention.py:57`，初始值是 `torch.tensor([])`。`attention.py:62` 的写入条件是 `if k_cache.numel() and v_cache.numel()`——只有在 `allocate_kv_cache()` 完成绑定后才会写入。warmup 阶段（发生在 `allocate_kv_cache` 之前）确实不写缓存，这是正确的行为。

**风险点：绑定机制使用 duck typing。** `hasattr(module, "k_cache")` 检查意味着如果模型中有其他模块碰巧具有这两个属性，或者层的遍历顺序不符合预期，缓存会被错误绑定。对于当前仅支持 Qwen3 的实现来说这不是问题，但扩展到其他架构时需要注意。

### 1.5 本章小结

> 💡 小结
> - KV Cache 是一个 `[2, layers, blocks, block_size, kv_heads, head_dim]` 的单一连续张量，一次分配完成
> - 可用 block 数量通过 warmup 测量的峰值显存动态计算，占满 GPU 利用率上限
> - 每层的注意力模块通过 view 引用同一张量的不同切片，零额外显存开销
> - 张量并行下 KV head 数按 world_size 分片，线性减少每 GPU 的 block 体积

---

## 二、Block 生命周期管理：分配、释放与前缀缓存

### 2.1 设计哲学与核心问题

物理缓存已经分配好了，但"哪个 block 给哪个序列用"需要一个管理者。这就是 `BlockManager` 的职责。它解决的核心问题是：

1. **按需分配**：序列增长时才分配新 block，不预分配
2. **及时释放**：序列结束或被抢占时释放 block
3. **前缀复用**：多个请求如果有相同的系统提示（system prompt），它们的 KV Cache 应该共享，而不是各算一份

第三点是最关键的——前缀缓存可以节省大量重复计算。想象 100 个请求共享一个 2048 token 的系统提示：第一个请求需要完整预填充，后续 99 个请求可以直接复用已缓存的 KV block，跳过 2048 个 token 的计算。

### 2.2 源码入口与关键对象

```
nanovllm/engine/block_manager.py
  - Block 类（第 8-23 行）：物理 block 的元数据（block_id、ref_count、hash、token_ids）
  - BlockManager 类（第 26-121 行）：block 池管理，四个核心数据结构：
    · blocks: list[Block] — 物理 block 数组（按 block_id 索引）
    · hash_to_block_id: dict — 哈希到 block_id 的映射（前缀缓存的核心）
    · free_block_ids: deque — 空闲 block 队列（FIFO）
    · used_block_ids: set — 正在使用的 block 集合
```

### 2.3 主流程拆解

#### Block 的操作系统类比

nano-vllm 的 block 管理本质上是一个简化的虚拟内存系统：

| 操作系统概念 | PagedAttention 对应 | 代码位置 |
|---|---|---|
| 物理页帧 | `Block` 对象 | `block_manager.py:8-23` |
| 物理页号 (PPN) | `block.block_id` | `block_manager.py:11` |
| 页表 | `seq.block_table`（block_id 列表） | `sequence.py:28` |
| 虚拟页号 | 逻辑 block 索引 `i` | `sequence.py:63` |
| 物理地址 | `block_id × block_size + offset` | `model_runner.py:154,181` |
| 页帧分配器 | `free_block_ids` 队列 | `block_manager.py:32` |
| TLB/页表查询 | `slot_mapping` 张量构造 | `model_runner.py:150-161` |

#### 前缀缓存：xxhash 链式哈希

前缀缓存的核心是内容可寻址的 block 查找。`compute_hash`（`block_manager.py:36-41`）实现了链式哈希：

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

哈希链的结构类似 Merkle 链：

```
Block 0: H₀ = xxh64(tokens[0:256])
Block 1: H₁ = xxh64(H₀ ‖ tokens[256:512])
Block 2: H₂ = xxh64(H₁ ‖ tokens[512:768])
  ...
Block i: Hᵢ = xxh64(Hᵢ₋₁ ‖ tokens[i×256 : (i+1)×256])
```

每个 block 的哈希包含了从序列开头到该 block 的所有 token 信息。这意味着：

- 两个不同序列在相同位置有相同 token 内容 → 产生相同哈希 → 可以共享 KV Cache
- 任何位置的 token 差异都会"传播"到后续所有 block 的哈希 → 不会错误共享

**为什么用内容寻址而非位置寻址？** 如果按 `(seq_id, block_pos)` 索引，不同序列即使有相同系统提示也无法共享。内容寻址使得共享是"自动"的——不需要显式维护共享树。

#### 分配流程：`can_allocate` → `allocate`

当调度器要为一个新序列分配 block 时，先调用 `can_allocate`（`block_manager.py:58-73`）探测：

```python
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):        # 注意：最后一个 block 不参与缓存检查
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break                               # 第一个 miss 就停止
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1                 # 已在用的缓存 block 不消耗空闲位
    if len(self.free_block_ids) < num_new_blocks:
        return -1                               # 空闲 block 不够
    return num_cached_blocks
```

几个关键设计点：

1. **只缓存完整 block**：循环范围是 `range(seq.num_blocks - 1)`，最后一个（可能不满）block 永远不参与缓存匹配。
2. **前缀连续性**：循环在第一个 miss 就 `break`——缓存命中必须是从头开始的连续前缀，不能跳过中间的 miss。这对自回归模型是正确的，因为任何位置的 KV 值都依赖于所有前置 token。
3. **双重验证**：哈希匹配后还要比较 `token_ids`（第 66 行）。xxh64 的碰撞概率约 1/2⁶⁴，但碰撞一旦发生会导致使用错误 KV 值的无声推理错误，所以必须有这个防线。
4. **返回值双重含义**：返回 -1 表示"无法分配"，返回 ≥0 表示"可以分配，且有 N 个缓存命中"。

然后 `allocate`（`block_manager.py:75-92`）执行实际分配：

```python
def allocate(self, seq: Sequence, num_cached_blocks: int):
    h = -1
    for i in range(num_cached_blocks):
        # ... 复用缓存 block，递增 ref_count
        block.ref_count += 1
        seq.block_table.append(block_id)
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())   # 从空闲池分配新 block
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

**懒惰驱逐（Lazy Eviction）的精妙设计**：当一个 block 被释放（`deallocate`）时，它回到 `free_block_ids` 队列，但 `hash_to_block_id` 中的条目**不会被删除**，block 上的 `hash` 和 `token_ids` 也**不会被清除**。这意味着被释放的 block 仍然可以被未来的请求通过哈希匹配"复活"（`allocate` 第 86-88 行的路径）。只有当这个 block 被重新分配给其他内容（`_allocate_block` 第 47-48 行）时，旧的哈希条目才被清理。

这实际上是一种隐式的 **FIFO 驱逐策略**：`free_block_ids` 是 deque，释放的 block 追加到尾部（第 56 行 `append`），分配时从头部取出（第 44 行 `popleft`）。最早释放的 block 最先被复用——给最近释放的 block 更多时间留在"缓存"中被命中。

#### 释放流程：引用计数

`deallocate`（`block_manager.py:94-101`）在序列结束或被抢占时调用：

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

引用计数使得多个序列可以安全共享同一个 block：只有当所有引用者都释放后，block 才真正回到空闲池。

#### Decode 阶段的增量分配：`can_append` / `may_append`

每个 decode step 生成一个 token。大多数时候，新 token 写入当前 block 的空闲槽位，不需要新 block。只有当序列长度刚好越过 block 边界时（`len(seq) % block_size == 1`），才需要分配一个新 block：

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

这行代码利用了 Python 的 `bool → int` 隐式转换：右侧表达式是 `True`（=1）或 `False`（=0），作为需要的新 block 数量。每 256 个 decode step 中只有 1 个需要分配新 block——分配频率极低。

#### Block 注册：`hash_blocks`

在 `postprocess` 阶段（调度器后处理），新完成的完整 block 被注册到哈希表：

```python
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

这使得后续请求可以发现并复用这些 block。注意：只有**完整**的 block（256 个 token 都已填满）才会被注册——部分填充的 block 不参与缓存。

### 2.4 关键细节与误区澄清

**误区：释放 block 后哈希条目也会被清除。** 实际上不会。`deallocate` 只递减 `ref_count` 并将 block 放回空闲池，但 `hash_to_block_id` 中的条目保留。这是 **lazy eviction** 的有意设计——block 在空闲池中等待被"复活"或被新分配覆盖。

**误区：`can_allocate` 返回 -1 意味着序列永远无法被调度。** 不是的。它只意味着**当前**空闲 block 不足。调度器会通过抢占其他运行中的序列来释放 block，或者等待其他序列完成后自然释放。

**误区：前缀缓存可以跨 block 边界匹配部分内容。** 不行。缓存的粒度是完整 block（256 token）。如果两个请求共享 300 个 token 的前缀，只有前 256 个 token（1 个完整 block）可以被缓存命中，剩余 44 个 token 必须重新计算。

**潜在性能问题**：`allocate` 方法中对被驱逐但仍有哈希的缓存 block 的"复活"路径（第 87 行）调用 `self.free_block_ids.remove(block_id)`，这是 `deque` 的 O(n) 操作。在 block 数量很大（数千个）时可能成为瓶颈。

### 2.5 本章小结

> 💡 小结
> - Block 管理实现了完整的"虚拟内存"语义：分配、释放、引用计数、内容寻址缓存
> - 前缀缓存使用 xxhash 链式哈希，每个 block 的哈希包含完整前缀信息，碰撞概率 ~1/2⁶⁴，外加 token_ids 比较防线
> - 懒惰驱逐策略使释放的 block 保留哈希信息等待复用，隐式实现了 FIFO 缓存驱逐
> - 缓存粒度是 256 token（一个完整 block），部分填充的最后 block 不参与缓存

---

## 三、调度器与显存的耦合：Prefill-First 与抢占

### 3.1 设计哲学与核心问题

调度器是 BlockManager 和 ModelRunner 之间的"桥梁"。它解决的核心问题是：**在有限的显存和计算预算下，先执行哪些序列的什么阶段？**

nano-vllm 选择了最简单的策略——**严格 prefill 优先**：只要有等待预填充的请求，就不会执行 decode。这保证了新请求的 Time-To-First-Token (TTFT) 最优，但牺牲了正在生成中的请求的 Inter-Token Latency (ITL)。

### 3.2 源码入口与关键对象

```
nanovllm/engine/scheduler.py
  - Scheduler 类（第 9-92 行）：
    · schedule()（第 25-73 行）：两阶段调度主逻辑
    · preempt()（第 75-79 行）：序列抢占
    · postprocess()（第 81-92 行）：后处理——哈希注册、token 追加、完成检测
```

### 3.3 主流程拆解

#### 两阶段调度

`schedule()` 方法（`scheduler.py:25-73`）是推理引擎每个 step 的核心决策点。它分为两个阶段，具有严格的优先级：

```
schedule() 调用流程:
  │
  ├─ Phase 1: Prefill（第 29-55 行）
  │   遍历 waiting 队列（FIFO）
  │   对每个序列：检查 block 可分配性 → 计算需处理 token 数 → 分配 block → 设置调度参数
  │   如果调度了任何 prefill 工作 → 立即返回，不进入 decode 阶段
  │
  └─ Phase 2: Decode（第 57-73 行）
      仅在无 prefill 工作时执行
      遍历 running 队列
      对每个序列：检查能否追加 block → 如不能，抢占其他序列 → 调度 decode
```

**Prefill 阶段的关键逻辑**（第 29-53 行）：

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0: break
    if not seq.block_table:                      # 首次调度，需要分配 block
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1: break        # 空闲 block 不足
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
    else:                                        # 分块预填充的后续 chunk
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    if remaining < num_tokens and scheduled_seqs: # 只允许第一个序列被分块
        break
    if not seq.block_table:
        self.block_manager.allocate(seq, num_cached_blocks)
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    # ...
```

几个值得注意的设计：

1. **Block 提前全量分配**：`allocate` 在第 45 行分配了序列所需的**全部** block（不只是当前 chunk 需要的），即使分块预填充只处理部分 token。这意味着一个 32K token 的序列在第一个 chunk 就会分配 128 个 block，其中大部分暂时为空。
2. **分块限制**：第 42 行的条件 `if remaining < num_tokens and scheduled_seqs` 意味着只有 batch 中的第一个序列可以被分块。后续序列如果放不下就直接跳过。这简化了 Flash Attention 的 `cu_seqlens` 处理。
3. **token 预算**：默认 `max_num_batched_tokens=16384`，即每个 prefill step 最多处理 16384 个 token。

**Decode 阶段与抢占**（第 57-73 行）：

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())     # LIFO 驱逐：抢占最后加入的序列
        else:
            self.preempt(seq)                     # 自身也无法服务，抢占自己
            break
    else:
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
```

抢占逻辑的设计：

- **LIFO 驱逐**：`self.running.pop()` 从队尾取——最后加入的序列最先被牺牲。直觉是：最新到达的序列已完成的 decode 最少，抢占它浪费的计算量最小。
- **完全释放**：`preempt()` 会调用 `deallocate()` 释放序列的所有 block。没有 swap-to-CPU 机制——抢占后的序列必须从头重新预填充（除非前缀缓存能恢复部分 block）。
- **优先重调度**：被抢占的序列被 `appendleft` 到 waiting 队列最前面，下次 prefill 时最先被调度。

#### 后处理

`postprocess()`（`scheduler.py:81-92`）在每个 step 的模型执行之后调用：

```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)       # 注册新完成的完整 block 到哈希表
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue                               # 分块预填充未完成，不追加 token
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)               # O(n) 操作
```

### 3.4 关键细节与误区澄清

**误区：decode 和 prefill 可以在同一个 step 中混合执行。** 不行。`schedule()` 在 prefill 阶段如果调度了任何工作（第 54-55 行 `if scheduled_seqs: return scheduled_seqs, True`），会立即返回而不进入 decode 阶段。这是与 vLLM 的重要差异——vLLM 支持 mixed batching。

**误区：抢占后序列可以从中断点继续 decode。** 不行。`preempt()` 调用 `deallocate()` 释放所有 block，序列的 `is_prefill` 被重置为 `True`。它必须完全重新预填充。唯一的缓解是前缀缓存——如果之前的 block 还留在空闲池中（尚未被其他分配覆盖），它们的哈希条目仍然有效，`can_allocate` 可以发现并复用它们。

**潜在的死锁场景**：当只剩一个 running 序列且它需要新 block 但没有空闲 block 时（decode 阶段第 63-65 行），它会抢占自己。然后 `break` 退出内层循环，外层循环的 `for-else` 中 `else` 分支不执行，`scheduled_seqs` 为空，触发第 71 行的 `assert scheduled_seqs` 断言失败，引擎崩溃。这是一个已知的边界情况 bug。

### 3.5 本章小结

> 💡 小结
> - 严格 prefill-first 策略：有 prefill 就不 decode，保证 TTFT 但可能饿死 decode
> - 分块预填充允许长序列分多步处理，但 block 在首 chunk 就全量分配
> - 抢占采用 LIFO + 完全释放策略，没有 CPU swap，依赖前缀缓存缓解重计算开销
> - 后处理阶段注册新的完整 block 到哈希表，使后续请求可以利用前缀缓存

---

## 四、地址翻译：slot_mapping 如何连接逻辑位置与物理缓存

### 4.1 设计哲学与核心问题

BlockManager 管理的是 block 级别的逻辑-物理映射（block_table），但 GPU 上的 Triton 内核和 Flash Attention 需要**token 级别**的物理地址。`slot_mapping` 就是这个翻译层——它把每个 token 在序列中的逻辑位置转换为 KV Cache 张量中的物理槽位索引。

这一步是纯 CPU 操作，发生在每个 step 的 GPU 计算之前（`prepare_prefill` 或 `prepare_decode`），计算出的张量随后被传到 GPU。

### 4.2 源码入口与关键对象

```
nanovllm/engine/model_runner.py
  - prepare_prefill()（第 129-170 行）：预填充阶段的地址翻译
  - prepare_decode()（第 172-188 行）：解码阶段的地址翻译
  - prepare_block_tables()（第 123-127 行）：将 block_table 列表转为 GPU 张量
```

### 4.3 主流程拆解

#### 地址翻译公式

对于序列 S 中逻辑位置 P 的 token：

```
block_index = P ÷ block_size              （逻辑 block 索引）
offset      = P mod block_size            （block 内偏移）
physical_block_id = S.block_table[block_index]
physical_slot = physical_block_id × block_size + offset
```

这个 physical_slot 就是 KV Cache 张量第 3、4 维（block 和 slot）的线性化索引。

#### Prefill 路径（`prepare_prefill`，第 129-170 行）

Prefill 可能涉及多个 block 的多个 token（可能还有前缀缓存跳过部分 token）。关键变量：

```python
start = seq.num_cached_tokens          # 从哪个 token 开始（前缀缓存跳过的部分）
seqlen_q = seq.num_scheduled_tokens    # 本次要处理多少个 token
end = start + seqlen_q                 # 结束位置
```

slot_mapping 的构造遍历涉及的 block：

```python
start_block = start // self.block_size
end_block = (end + self.block_size - 1) // self.block_size
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * self.block_size
    if i == start_block:
        slot_start += start % self.block_size    # 第一个 block 可能从中间开始
    if i != end_block - 1:
        slot_end = seq.block_table[i] * self.block_size + self.block_size
    else:
        slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
    slot_mapping.extend(range(slot_start, slot_end))
```

一个具体例子：假设 `block_size=256`，序列有 1024 个 token（4 个 block），前缀缓存命中 512 个 token（2 个 block）：

```
block_table = [5, 12, 3, 8]     （物理 block ID）
start = 512, end = 1024
start_block = 2, end_block = 4

Block 2（物理 block 3）: slots 768..1023  → [3×256+0 .. 3×256+255]
Block 3（物理 block 8）: slots 2048..2303 → [8×256+0 .. 8×256+255]
```

slot_mapping = [768, 769, ..., 1023, 2048, 2049, ..., 2303]，共 512 个值。

#### Prefill 中的前缀缓存检测

第 162-163 行有一个关键判断：

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
    block_tables = self.prepare_block_tables(seqs)
```

当 `cu_seqlens_k`（K 维度的总长度）大于 `cu_seqlens_q`（Q 维度的总长度）时，意味着有些 K/V 来自缓存（已写入但不在本次计算范围内），需要通过 `block_tables` 告诉 Flash Attention 去分页缓存中读取。

#### Decode 路径（`prepare_decode`，第 172-188 行）

Decode 时每个序列只有 1 个新 token，地址翻译极其简单：

```python
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

最后一个 block 的最后一个已填充位置。`last_block_num_tokens`（`sequence.py:60-61`）返回 `num_tokens - (num_blocks - 1) * block_size`，值域为 `[1, block_size]`。

### 4.4 关键细节与误区澄清

**误区：`prepare_prefill` 为序列的所有 token 生成 slot_mapping。** 实际上只为**本次调度的** token 生成。`start = seq.num_cached_tokens` 跳过了前缀缓存已覆盖的 token，`seqlen_q = seq.num_scheduled_tokens` 限制了分块预填充的范围。

**误区：Decode 时的 `last_block_num_tokens - 1` 可能为负。** 不会。`last_block_num_tokens` 的最小值是 1（序列至少有 1 个 token，且 block 至少包含 1 个 token），所以 `- 1` 后最小为 0——对应 block 内的第一个位置。

### 4.5 本章小结

> 💡 小结
> - slot_mapping 是 block 级页表到 token 级物理地址的翻译层，纯 CPU 计算
> - Prefill 路径需处理 block 边界、前缀缓存偏移、分块预填充等复杂情况
> - Decode 路径极简：每序列 1 个 slot，指向最后 block 的最后已填充位置
> - 所有张量使用 pin_memory + non_blocking 传输，最小化 CPU-GPU 数据拷贝延迟

---

## 五、注意力计算：Triton 写入与 Flash Attention 读取

### 5.1 设计哲学与核心问题

地址翻译完成后，GPU 需要完成两件事：

1. **把新计算的 K/V 写入分页缓存的正确位置**（scatter write）
2. **从分页缓存中读取所有相关 K/V 计算注意力**（paged read）

第一件事由自定义 Triton 内核完成，第二件事委托给 Flash Attention 库。两者通过 `Context` 全局对象获取 `slot_mapping` 和 `block_tables`。

### 5.2 源码入口与关键对象

```
nanovllm/layers/attention.py
  - store_kvcache_kernel（第 10-30 行）：Triton JIT 内核，将 K/V 散写入分页缓存
  - store_kvcache()（第 33-40 行）：Python 包装器，校验 shape 并启动内核
  - Attention.forward()（第 59-75 行）：调用写入 + Flash Attention 的两路径入口
nanovllm/utils/context.py
  - Context 数据类（第 5-14 行）：全局状态，携带 is_prefill、slot_mapping、block_tables 等
  - get_context/set_context/reset_context：全局 getter/setter
```

### 5.3 主流程拆解

#### Triton 内核：scatter write

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                         k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)                        # 每个 token 一个 program
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return                          # 跳过填充位置（CUDA Graph 场景）
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

设计要点：

- **一个 token 一个 Triton program**：`tl.program_id(0)` 索引 batch 内的 token 位置
- **D = num_kv_heads × head_dim**：每个 token 的 K/V 向量被作为一段连续数据一次性读写
- **slot = -1 哨兵值**：CUDA Graph 回放时 batch size 可能小于捕获时的 batch size，多余位置的 slot_mapping 被填充为 -1（`model_runner.py:206`），内核跳过这些位置
- **scatter 写入模式**：不同 token 的 slot 可能指向物理缓存中不连续的位置——这正是分页的核心，打破了物理连续性的需求

#### Flash Attention 的两路径

`Attention.forward()`（`attention.py:59-75`）根据 `context.is_prefill` 分为两条路径：

**Prefill 路径**（第 64-70 行）：

```python
if context.is_prefill:
    if context.block_tables is not None:    # 前缀缓存：K/V 部分在缓存中
        k, v = k_cache, v_cache             # 用缓存替代当前计算的 k, v
    o = flash_attn_varlen_func(q, k, v,
                               cu_seqlens_q=..., cu_seqlens_k=...,
                               block_table=context.block_tables, causal=True)
```

关键点：

- **无前缀缓存时**（`block_tables is None`）：新计算的 `k, v` 直接传给 Flash Attention，不通过分页缓存读取。K/V 同时被 Triton 内核写入缓存（为后续 decode 准备）并直接用于本次注意力计算——避免了一次冗余的缓存读取。
- **有前缀缓存时**（`block_tables is not None`）：用 `k_cache, v_cache` 替代 `k, v`，Flash Attention 通过 `block_table` 参数从分页缓存中读取所有 K/V。

**Decode 路径**（第 72-74 行）：

```python
else:
    o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                cache_seqlens=context.context_lens,
                                block_table=context.block_tables, causal=True)
```

`flash_attn_with_kvcache` 原生支持分页 KV Cache：

- `q.unsqueeze(1)`：添加 seqlen 维度（=1），形状从 `[batch, heads, dim]` 变为 `[batch, 1, heads, dim]`
- `block_table`：形状 `[batch, max_num_blocks]`，每个元素是物理 block_id
- `cache_seqlens`：每个序列的总长度，Flash Attention 只读取有效范围内的 K/V

#### 全局 Context 的生命周期

```
prepare_prefill/prepare_decode → set_context(...)     写入全局状态
          ↓
    model.forward()
          ↓
    每个注意力层 → Attention.forward() → get_context()  读取全局状态
          ↓
    ParallelLMHead.forward() → get_context()            读取 cu_seqlens_q 取最后 token
          ↓
    ModelRunner.run() → reset_context()                  清理全局状态
```

### 5.4 关键细节与误区澄清

**误区：Prefill 时 K/V 从缓存中读取，所以 Triton 写入内核是必须先执行完再做注意力。** 实际上：在无前缀缓存的常规 prefill 中（`block_tables is None`），Flash Attention 直接使用刚计算的 `k, v` 张量，根本不从缓存读取。Triton 写入内核是为**后续 decode** 做准备，与本次注意力计算并行"准备"数据。只有存在前缀缓存时（`block_tables is not None`），Flash Attention 才从分页缓存读取。

**误区：全局 `_CONTEXT` 是线程安全的。** 不是。它是一个裸的模块级全局变量，没有锁。但在当前设计中这不是问题——推理引擎是单线程的（一次只执行一个 step）。然而，这个设计使得在同一进程中运行多个推理 pipeline（如投机解码需要的 draft model + target model）成为不可能。

**全局 Context 的脆弱性**：如果 `set_context` 和 `reset_context` 之间发生异常，全局状态会残留。没有 `with` 语句或 context manager 保护清理。

### 5.5 本章小结

> 💡 小结
> - KV Cache 写入由 Triton scatter-write 内核完成，slot=-1 哨兵值适配 CUDA Graph 填充
> - Prefill 无前缀缓存时直接使用当前 K/V（跳过缓存读取），有前缀缓存时通过 block_table 从缓存读取
> - Decode 使用 `flash_attn_with_kvcache` 原生支持分页读取
> - 全局 Context 模式简单但脆弱——不支持并发 pipeline，异常时可能残留状态

---

## 六、CUDA Graph 与分页注意力的协作

### 6.1 设计哲学与核心问题

CUDA Graph 通过一次性捕获 GPU 操作序列来消除逐 kernel 的 CPU 启动开销。对于 decode 阶段（每个序列只 1 个 token，GPU 计算量小，CPU 启动开销占比大），CUDA Graph 可以带来 20-40% 的性能提升。

但 PagedAttention 引入了一个冲突：**CUDA Graph 要求固定的张量地址，而分页注意力的 `slot_mapping`、`block_tables` 每个 step 都不同。** 解决方案是预分配最大尺寸的固定缓冲区，每次回放前把实际数据拷入。

### 6.2 源码入口与关键对象

```
nanovllm/engine/model_runner.py
  - capture_cudagraph()（第 222-257 行）：捕获多个 batch size 的 CUDA Graph
  - run_model()（第 196-212 行）：选择并回放 CUDA Graph 或走 eager 路径
```

### 6.3 主流程拆解

#### 捕获

`capture_cudagraph()`（第 222-257 行）在初始化时执行：

```python
max_bs = min(self.config.max_num_seqs, 512)
# 捕获的 batch size: [1, 2, 4, 8, 16, 32, ..., max_bs]
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

# 预分配固定缓冲区——最大 batch size
input_ids = torch.zeros(max_bs, dtype=torch.int64)
positions = torch.zeros(max_bs, dtype=torch.int64)
slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
context_lens = torch.zeros(max_bs, dtype=torch.int32)
block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
outputs = torch.zeros(max_bs, hf_config.hidden_size)

for bs in reversed(self.graph_bs):      # 从大到小捕获
    graph = torch.cuda.CUDAGraph()
    set_context(False, slot_mapping=slot_mapping[:bs], ...)
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # 捕获
    if self.graph_pool is None:
        self.graph_pool = graph.pool()   # 第一个（最大的）Graph 创建内存池
    self.graphs[bs] = graph
```

设计要点：

- **从大到小捕获**：第一个捕获的是最大 batch size，它创建的 graph pool 容量最大。后续更小的 graph 复用同一个 pool——类似预分配大缓冲区再子分配。
- **batch size 粒度**：小 batch 用 2 的幂次（1,2,4,8），大 batch 用 16 的步长（16,32,48,...）。对齐 GPU warp size，默认最多 36 个 graph。
- **固定缓冲区**：`slot_mapping`、`block_tables` 等是预分配的固定张量，CUDA Graph 记录的是对这些张量地址的操作。回放时修改这些张量的内容（不改地址），Graph 就能用新数据执行旧的操作序列。

#### 回放

`run_model()`（第 196-212 行）的 CUDA Graph 路径：

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))  # eager
else:
    bs = input_ids.size(0)
    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]    # 向上取整
    graph_vars["input_ids"][:bs] = input_ids
    graph_vars["positions"][:bs] = positions
    graph_vars["slot_mapping"].fill_(-1)           # 全部填 -1
    graph_vars["slot_mapping"][:bs] = context.slot_mapping   # 覆盖有效部分
    graph_vars["context_lens"].zero_()
    graph_vars["context_lens"][:bs] = context.context_lens
    graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
    graph.replay()
    return self.model.compute_logits(graph_vars["outputs"][:bs])
```

**填充处理**：batch size 5 使用 graph_bs=8 的 graph。多余的 3 个位置：

- `slot_mapping` 填为 -1 → Triton 内核跳过（`attention.py:23`）
- `context_lens` 填为 0 → Flash Attention 不计算这些位置的注意力
- `block_tables` 只写前 `bs` 行 → 多余行包含上次的残留数据

**Prefill 不使用 CUDA Graph**：第 197 行的条件排除了 `is_prefill`。原因是 prefill 的 batch 形状（变长序列打包）与 CUDA Graph 的固定形状不兼容。

### 6.4 关键细节与误区澄清

**误区：CUDA Graph 回放时 `block_tables` 的未覆盖行是安全的。** 这是一个值得关注的潜在问题：第 210 行只写了 `[:bs, :context.block_tables.size(1)]`，未被覆盖的列和行可能包含上次回放的残留数据。对于行（位置 `bs` 到 `graph_bs`），`context_lens` 为 0 保证 Flash Attention 不会访问这些行的 block_tables。但理论上，如果 Flash Attention 的内部实现不检查 `context_lens`，就可能读到脏数据。在实践中，Flash Attention 确实会用 `cache_seqlens` 做边界检查，所以目前是安全的。

**误区：CUDA Graph 消耗的额外显存是 `graph 数量 × 单 graph 显存`。** 实际上不是——所有 graph 共享同一个 `graph_pool`，最大 graph 的分配覆盖了所有更小 graph 的需求。额外显存主要来自预分配的固定缓冲区（约 1 MiB）和 graph pool 的 kernel workspace。

### 6.5 本章小结

> 💡 小结
> - CUDA Graph 仅用于 decode 阶段（batch size ≤ 512），prefill 始终 eager 执行
> - 通过预分配固定缓冲区 + 每次回放前数据拷贝，绕过了 CUDA Graph 的固定地址限制
> - slot_mapping 的 -1 哨兵值和 context_lens 的 0 填充共同保证了填充位置的安全性
> - 从大到小捕获 + graph pool 共享，最小化 CUDA Graph 的额外显存开销

---

## 七、完整主路径串联

### 7.1 完整调用栈

```
User: llm.generate(prompts, sampling_params)
  │
  ├─ Step 1: 请求入队
  │     LLMEngine.generate()                      [llm_engine.py:60-90]
  │       └─ for prompt, sp in zip(prompts, sampling_params):
  │            self.add_request(prompt, sp)        [llm_engine.py:43-47]
  │              └─ tokenize → Sequence(tokens, sp) → scheduler.add(seq)
  │
  ├─ Step 2: 主循环 (每次迭代 = 一个 step)
  │     while not self.is_finished():
  │       │
  │       ├─ Step 2a: 调度
  │       │     seqs, is_prefill = scheduler.schedule()   [scheduler.py:25-73]
  │       │       ├─ Phase 1: Prefill                     
  │       │       │   can_allocate → allocate → 设置 num_scheduled_tokens
  │       │       └─ Phase 2: Decode（仅在无 prefill 时）
  │       │           can_append → may_append → preempt（如需要）
  │       │
  │       ├─ Step 2b: 模型执行
  │       │     token_ids = model_runner.call("run", seqs, is_prefill)  [model_runner.py:85-89]
  │       │       ├─ prepare_prefill/prepare_decode         [model_runner.py:129-188]
  │       │       │   计算 slot_mapping, block_tables → set_context()
  │       │       ├─ run_model(input_ids, positions, is_prefill)  [model_runner.py:196-212]
  │       │       │   ├─ Eager: model.forward() → compute_logits()
  │       │       │   └─ CUDA Graph: 数据拷入固定缓冲区 → graph.replay()
  │       │       │       每层 Attention.forward():            [attention.py:59-75]
  │       │       │         ├─ store_kvcache(k, v, k_cache, v_cache, slot_mapping)
  │       │       │         └─ flash_attn_varlen_func / flash_attn_with_kvcache
  │       │       ├─ sampler(logits, temperatures)            [sampler.py:8-12]
  │       │       └─ reset_context()
  │       │
  │       └─ Step 2c: 后处理
  │             scheduler.postprocess(seqs, token_ids, is_prefill)  [scheduler.py:81-92]
  │               ├─ block_manager.hash_blocks(seq)  — 注册新完成的 block
  │               ├─ seq.num_cached_tokens += seq.num_scheduled_tokens
  │               ├─ seq.append_token(token_id)      — 追加新 token
  │               └─ 检查 EOS / max_tokens → deallocate + FINISHED
  │
  └─ Step 3: 结果收集
        outputs = {seq_id: completion_token_ids}
        tokenizer.decode → return [{"text": ..., "token_ids": ...}]
```

### 7.2 每一层做了什么

| 阶段 | 输入 | 输出 | 状态变化 | 通信 | 显存影响 | 频率 |
|---|---|---|---|---|---|---|
| add_request | 文本/token_ids | Sequence 对象 | waiting 队列增长 | 无 | 无 | 每请求一次 |
| schedule (prefill) | waiting 队列 | 序列列表 + is_prefill | block 分配, waiting→running | 无 | block 分配 | 有 waiting 时每 step |
| schedule (decode) | running 队列 | 序列列表 + is_prefill | 可能 may_append/preempt | 无 | 可能分配/释放 block | 无 waiting 时每 step |
| prepare_prefill/decode | 序列列表 | GPU 张量 | set_context 写入全局状态 | 无 | pin_memory 临时分配 | 每 step |
| run_model | input_ids, positions | logits | KV Cache 写入 | all_reduce (TP) | 激活值临时分配 | 每 step |
| sampler | logits, temperatures | token_ids | 无 | 无 | 临时分配 | 每 step |
| postprocess | token_ids | 完成检测 | hash_blocks, token 追加 | 无 | 完成时释放 block | 每 step |

### 7.3 哪些逻辑不在主路径

| 容易误解的函数/逻辑 | 为什么看似在主路径 | 实际情况 |
|---|---|---|
| `warmup_model()`（`model_runner.py:91-101`） | 名字暗示可能每次执行 | 仅初始化时执行一次，用于测量峰值显存 |
| `capture_cudagraph()`（`model_runner.py:222-257`） | CUDA Graph 是性能关键 | 仅初始化时执行一次，运行时只做 replay |
| `loop()`（`model_runner.py:61-66`） | 看起来是主循环 | 只在 TP rank > 0 的 worker 进程中运行，主进程不走这条路径 |
| `Sequence.__getstate__`/`__setstate__` | 每次 step 都会调用 | 只在 TP > 1 时通过 SharedMemory 序列化时使用 |
| `Block.update()` / `Block.reset()` | 看起来是 block 生命周期的核心 | 只是简单的字段赋值，真正的逻辑在 BlockManager 的方法中 |

---

## 八、显存、性能与通信分析

### 8.1 显存收益范围

| 内容 | 是否因 PagedAttention 节省 | 原因 |
|---|---|---|
| 模型参数 | ❌ | 参数存储与 KV Cache 管理无关 |
| KV Cache 内部碎片 | ✅ 大幅减少 | 仅最后一个 block 有碎片（平均浪费 128 token/序列） |
| KV Cache 外部碎片 | ✅ 消除 | 非连续分配，block 可被任意序列复用 |
| 重复前缀的 KV Cache | ✅ 前缀缓存 | 共享系统提示的请求复用相同 block |
| 激活值 | ❌ | 激活值是临时分配，与 KV Cache 分页无关 |
| Logits | ❌ | Logits 在 sampler 中临时计算 |
| Optimizer state | N/A | 推理场景不涉及 optimizer |
| CUDA Graph 缓冲区 | ❌ 小量增加 | 预分配的固定缓冲区 ~1 MiB |

**真正的显存大头是 KV Cache 本身。** 以 Qwen3-0.6B 为例计算单 block 显存：

```
每 block 字节 = 2(K+V) × 28(层) × 256(token/block) × 2(KV head) × 64(head_dim) × 2(bf16)
             = 3,670,016 字节 ≈ 3.5 MiB

每 token 字节 = 3.5 MiB / 256 ≈ 14 KiB
```

在 8GB GPU（如 RTX 4070 Laptop）上的典型分配：

```
可用 = 8GB × 0.9 - ~2GB(模型+开销) ≈ 5.2 GB
block 数 = 5.2 GB / 3.5 MiB ≈ 1,485 blocks
token 容量 = 1,485 × 256 ≈ 380,160 tokens
```

使用 TP=2 时，每 GPU 的 KV head 减半 → 每 block 缩小到 1.75 MiB → 同样显存能容纳双倍 block。

### 8.2 通信开销

#### 每层通信（TP > 1 时）

每个 `Qwen3DecoderLayer` 包含：

| 位置 | 操作 | 通信量 |
|---|---|---|
| `o_proj`（`RowParallelLinear`） | 1 次 all_reduce | `batch × hidden_size × dtype_size` |
| `down_proj`（`RowParallelLinear`） | 1 次 all_reduce | `batch × hidden_size × dtype_size` |

额外的全局通信：

| 位置 | 操作 | 通信量 |
|---|---|---|
| `VocabParallelEmbedding` | 1 次 all_reduce | `batch × hidden_size × dtype_size` |
| `ParallelLMHead` | 1 次 gather（到 rank 0） | `batch × vocab_size × dtype_size` |

**总计**：每次前向传播 `2 × num_layers + 2` 次集合通信。Qwen3-0.6B (28 层) = 58 次。

#### SharedMemory IPC（TP > 1 时）

每个 step 一次 pickle 序列化 + 共享内存写入。Decode 阶段 `__getstate__` 只发送 `last_token`（不发全量 `token_ids`），序列化开销极小（~0.1-0.5ms）。

### 8.3 性能取舍

| 收益 | 代价 |
|---|---|
| 显存碎片消除 → 更多并发序列 | block 级寻址增加 GPU 内核复杂度（Triton scatter write） |
| 前缀缓存 → 跳过重复计算 | 每序列的 `can_allocate` 需要 O(blocks) 的哈希链计算 |
| CUDA Graph → 减少 CPU 启动开销 | 预分配固定缓冲区 + 每次回放前的数据拷贝 |
| block_size=256 → 对齐 Flash Attention 最优粒度 | 内部碎片平均 128 token/序列，短序列浪费显著 |
| 抢占机制 → 防止 OOM | 无 swap，抢占后必须完全重新预填充 |
| Prefill-first → 最小化 TTFT | Decode 延迟在持续负载下可能无限增长 |

---

## 九、配置项、边界条件与坑点

### 关键配置与行为影响

| 配置项 | 默认值 | 源码路径 | 行为影响 | 风险/坑点 |
|---|---|---|---|---|
| `kvcache_block_size` | 256 | `config.py:17` | KV Cache 分页粒度 | 必须是 256 的倍数（flash_attn 约束），短序列浪费大 |
| `gpu_memory_utilization` | 0.9 | `config.py:12` | 决定 KV Cache block 数量 | 设太高可能导致 OOM（CUDA Graph 捕获需要额外显存） |
| `max_num_batched_tokens` | 16384 | `config.py:9` | 单 step 最大 prefill token 数 | 影响分块预填充粒度和 decode 延迟 |
| `max_num_seqs` | 512 | `config.py:10` | 最大并发序列数 | 同时也是 CUDA Graph 的最大 batch size |
| `enforce_eager` | False | `config.py:14` | 禁用 CUDA Graph | 调试时有用，生产环境不建议开启 |
| `tensor_parallel_size` | 1 | `config.py:13` | TP 切分数 | 必须整除 num_attention_heads 和 num_kv_heads |
| `temperature` | 1.0 | `sampling_params.py:6` | 采样温度 | 必须 > 1e-10，不支持 greedy（temperature=0） |

### 硬编码约束

| 硬编码值 | 位置 | 影响 |
|---|---|---|
| `"tcp://localhost:2333"` | `model_runner.py:26` | NCCL 端口固定，同机多实例冲突 |
| `name="nanovllm"` | `model_runner.py:43` | SharedMemory 名称固定，同机多实例冲突 |
| `size=2**20`（1MB） | `model_runner.py:43` | SharedMemory 大小限制，大 batch 长序列 prefill 可能溢出 |
| `512`（CUDA Graph 上限） | `model_runner.py:197,226` | decode batch > 512 退回 eager 模式 |
| `Qwen3ForCausalLM` | `model_runner.py:31` | 仅支持 Qwen3 模型 |
| `block_size % 256 == 0` | `config.py:22` | block size 最小 256，无法更细粒度 |
| `1 ≤ TP ≤ 8` | `config.py:23` | 最多 8 卡并行 |

### 静默失效条件

1. **`max_model_len < block_size`**：序列只有 1 个 block，前缀缓存完全失效（`can_allocate` 的循环 `range(num_blocks - 1) = range(0)` 为空）
2. **相同系统提示但长度不是 block_size 的整数倍**：末尾的不完整 block 不被缓存，每次都需要重算
3. **SharedMemory 1MB 溢出**：TP > 1 时大量长序列 prefill 可能超出 pickle 序列化缓冲区

---

## 十、局限性与已知优化点

### 10.1 硬约束

- **仅支持 Qwen3**：模型类硬编码在 `model_runner.py:31`
- **不支持 greedy 采样**：`sampling_params.py:11` 的断言 `temperature > 1e-10`
- **TP size 必须整除 `num_attention_heads` 和 `num_key_value_heads`**：否则 `linear.py:9` 的 `divide()` 会报 AssertionError
- **block_size 必须是 256 的倍数**：由 flash_attn 的分页注意力内核决定
- **不支持 beam search**：ref_count 机制存在但从未用于 beam 分叉
- **不支持 speculative decoding**：全局 Context 模式排除了多模型并行

### 10.2 维护成本

- **全局 Context 模式**：任何需要多 pipeline 的扩展（投机解码、pipeline 并行）都需要重构
- **duck-typing 的 KV Cache 绑定**：`hasattr(module, "k_cache")` 在新模型架构下可能匹配到错误的模块
- **SharedMemory 名称/端口硬编码**：无法同机多实例部署
- **无错误处理**：全链路使用 assert，CUDA OOM 或 Flash Attention 维度不匹配会直接崩溃

### 10.3 性能瓶颈

- **prefill-first 调度**：连续到达的长 prefill 请求会无限饿死 decode
- **无 swap 的抢占**：长序列被抢占后完全重新预填充，计算浪费严重
- **O(n) 操作**：`deque.remove()` 在 `allocate`（缓存 block 复活路径）和 `postprocess`（序列完成移除）中出现
- **`can_allocate` 每次重算全部哈希链**：即使序列未变化，也不缓存之前计算过的哈希
- **block_size=256 的内部碎片**：对于短序列（< 256 token），每个序列浪费一个完整 block

### 10.4 已知优化方向

1. **CPU Swap**：被抢占的序列将 KV Cache 换出到 CPU 而非丢弃，恢复时换入而非重新计算
2. **混合 Batching**：允许同一 step 中同时调度 prefill 和 decode，减少 decode 延迟
3. **更小的 block size**：如果 Flash Attention 未来支持更细粒度的 block table，可以减少内部碎片
4. **LRU 驱逐**：当前 FIFO 驱逐可替换为带时间戳的 LRU，优先保留高命中率的缓存 block
5. **异步 SharedMemory IPC**：当前同步 pickle 序列化可以优化为异步或使用更高效的序列化格式
6. **哈希缓存复用**：在 `can_allocate` 中缓存已计算的哈希链前缀，避免重复计算
7. **动态 SharedMemory 大小**：根据 `max_num_seqs × max_model_len` 动态计算缓冲区大小

---

## 小结与展望

nano-vllm 的 PagedAttention 实现可以用几个关键词概括。

**关键词一：虚拟内存抽象**

整个实现的核心是把操作系统的页式虚拟内存搬到了 GPU 显存管理上。`Block` = 页帧，`block_table` = 页表，`slot_mapping` = 物理地址翻译。这个类比不是表面的——nano-vllm 实现了完整的分配/释放/引用计数/懒惰驱逐语义，只差 swap space 就是一个完整的虚拟内存系统。

**关键词二：内容可寻址的前缀缓存**

xxhash 链式哈希使得 block 的查找基于内容而非身份。不需要维护显式的共享树或前缀 trie——只要 token 内容相同，hash 就相同，block 就可以共享。双重验证（hash + token_ids 比较）在保持 O(1) 查找速度的同时消除了碰撞风险。这是一个优雅的工程设计。

**关键词三：slot_mapping 桥接逻辑与物理**

slot_mapping 是整个系统中最"不起眼"但最关键的数据结构。它在每个 step 的 CPU 端被计算出来，桥接了调度器的 block 级视图和 GPU 内核的 token 级视图。slot=-1 的哨兵值更是一箭双雕地解决了 CUDA Graph 填充位置的正确性问题。

**关键词四：极简中的取舍**

~1200 行代码实现了 vLLM 的核心功能（PagedAttention + 前缀缓存 + 分块预填充 + CUDA Graph + 张量并行），但每个简化都有代价：无 swap 意味着抢占代价高昂，prefill-first 意味着 decode 可能饿死，block_size=256 意味着短序列浪费显存，全局 Context 意味着无法支持多 pipeline。

**适用场景**：

- 学习 vLLM 内部机制的教学材料
- 单模型、单实例的离线推理（batch inference）
- 在此基础上做 PagedAttention 相关的研究实验

**不适用场景**：

- 生产级在线服务（缺少混合 batching、swap、错误恢复）
- 多模型/多实例部署（硬编码端口和共享内存名称）
- 非 Qwen3 模型（硬编码模型类）
- 需要 greedy 采样或 beam search 的场景

**后续值得继续走读的方向**：

- 完整 vLLM 的 `Scheduler` 实现——如何做到 prefill/decode 混合调度
- vLLM 的 `SwapManager`——如何实现 KV Cache 的 CPU-GPU 换入换出
- vLLM 的 `Evictor`——如何用 LRU/LFU 策略优化前缀缓存的驱逐
- Flash Attention 内部的 block table 处理——分页读取在 CUDA kernel 层面如何实现
