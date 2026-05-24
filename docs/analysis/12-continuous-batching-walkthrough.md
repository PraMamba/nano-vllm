# nano-vllm 源码走读：Continuous Batching 实现解析

> 在 LLM 推理场景中，GPU 的利用率瓶颈往往不在单次前向计算本身，而在于**请求之间的调度效率**。传统的 static batching 要求一个 batch 内所有序列同时完成才能接入新请求，导致短序列完成后 GPU 空转等待长序列。Continuous Batching（也称 iteration-level scheduling）正是为了解决这个问题而提出的：让序列可以在 **每一个 decode step** 独立进出 batch，从而最大化 GPU 吞吐。本文不展开 PagedAttention 或 Flash Attention 的原理（这些在前序文章中已有覆盖），而是聚焦 nano-vllm 的源码，分析这个仅 ~1200 行的轻量框架如何把 Continuous Batching 从论文概念变成可运行的工程系统，以及它做了哪些取舍。

---

# 前言

## 业务与工程背景

LLM 推理的核心矛盾在于：**prefill 阶段是计算密集型（一次处理大量 prompt token），decode 阶段是访存密集型（每个序列每步只生成 1 个 token）**。两个阶段的计算特征截然不同，但它们共享同一块 GPU 和同一片 KV cache 显存。

在 static batching 下，一个 batch 被绑定为一个整体：所有序列必须等最长的那个完成，才能释放资源、接入新请求。这意味着一个 batch 中如果有一条 1024-token 的长序列和一条 32-token 的短序列，短序列完成后 GPU 会空转 992 步。

## 核心矛盾

Continuous Batching 要解决的工程矛盾是三方面的：

1. **调度灵活性 vs 执行效率**：每步都重新组 batch 带来调度开销，但能最大化 GPU 利用率。
2. **prefill 优先级 vs decode 延迟**：新请求需要 prefill 才能开始生成，但 prefill 期间已在 decode 的序列会被阻塞。
3. **KV cache 容量 vs 并发度**：更多并发序列意味着更高吞吐，但每个序列都需要 KV cache 显存，总量有限。

## 本文主线

本文按以下主线展开：

- **第一章**：Scheduler 的两阶段调度算法——Continuous Batching 的核心实现
- **第二章**：BlockManager 的分页内存管理——KV cache 的按需分配与回收
- **第三章**：Sequence 状态机与 Engine 主循环——请求的生命周期管理
- **第四章**：ModelRunner 的 prefill/decode 双路径——调度决策如何落地为 GPU 计算
- **第五章**：完整主路径串联——一次 `generate()` 调用的端到端数据流
- **第六章**：核心机制深挖——抢占、分块 prefill、前缀缓存
- **第七章**：显存、性能与通信分析
- **第八章**：配置项、边界条件与坑点
- **第九章**：测试、示例与覆盖缺口
- **第十章**：局限性与已知优化点
- **小结与展望**

## 不展开的内容

本文不讲 PagedAttention 的原理（见 `08-paged-attention-walkthrough.md`），不讲 BlockManager 的哈希缓存细节（见 `10-prefix-cache-walkthrough.md`），不讲 KV cache 容量计算（见 `11-kv-cache-capacity-walkthrough.md`）。只讲 **Continuous Batching 的调度机制**以及它如何与这些子系统协作。

## 核心文件表

| 文件 | 职责 |
|------|------|
| `nanovllm/engine/scheduler.py` | Continuous Batching 调度器：两阶段调度、抢占、后处理 |
| `nanovllm/engine/llm_engine.py` | 引擎主循环：编排 scheduler → model_runner → postprocess |
| `nanovllm/engine/block_manager.py` | 分页 KV cache 管理：分配、回收、前缀缓存 |
| `nanovllm/engine/sequence.py` | 序列状态机：WAITING → RUNNING → FINISHED |
| `nanovllm/engine/model_runner.py` | 模型执行器：输入组装、CUDA graph、前向计算 |
| `nanovllm/utils/context.py` | 全局 Context 单例：调度元数据到注意力层的隐式传递 |
| `nanovllm/layers/attention.py` | 注意力实现：prefill/decode 双路径 + KV cache 写入 |
| `nanovllm/config.py` | 配置：`max_num_seqs`、`max_num_batched_tokens` 等调度参数 |

---

# 一、两阶段调度算法：Continuous Batching 的核心

## 1.1 设计哲学与核心问题

nano-vllm 的 Continuous Batching 实现围绕一个关键设计决策：**prefill 和 decode 互斥**。每个调度步（step）要么是纯 prefill，要么是纯 decode，绝不混合。

为什么做出这个选择？因为 prefill 和 decode 的计算模式完全不同：

- **Prefill**：一次处理大量 token，使用 `flash_attn_varlen_func`（变长序列的打包注意力），输入 shape 是 `[total_tokens]` 级别。
- **Decode**：每个序列只处理 1 个 token，使用 `flash_attn_with_kvcache`（分页 KV cache 查询），输入 shape 是 `[batch_size]` 级别。

两种模式使用不同的 Flash Attention 入口函数、不同的输入张量组装逻辑、不同的 CUDA graph 策略（prefill 始终 eager，decode 可以 graph）。将它们混合在一个 step 中需要复杂的张量拼接和 kernel 路由——vLLM 做了，但在 1200 行代码的约束下，互斥是合理的简化。

代价是：**decode 序列在每次 prefill step 期间被阻塞**。如果持续有新请求到达，已在 decode 的序列的 time-to-next-token 会产生毛刺。

## 1.2 源码入口与关键对象

```
nanovllm/engine/scheduler.py
  - Scheduler.__init__：初始化 waiting/running 双队列、BlockManager
  - Scheduler.schedule()：核心调度函数，93 行中最重要的 ~50 行
  - Scheduler.preempt()：抢占（回收 KV cache、回退到 waiting）
  - Scheduler.postprocess()：后处理（token 追加、终止判定、block 哈希）
```

## 1.3 主流程拆解

`schedule()` 方法（`scheduler.py:25-73`）返回 `(list[Sequence], bool)`，其中 `bool` 表示是否为 prefill step。

### 阶段一：Prefill 调度（行 29-55）

```python
# scheduler.py:29-55
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]                  # FIFO：总是取队首
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0:
        break
    if not seq.block_table:                # 首次调度：检查 block 分配
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1:        # 显存不足
            break
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
    else:                                  # 分块续传：已有 block_table
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    if remaining < num_tokens and scheduled_seqs:  # ⚠️ 分块 prefill 仅允许首条序列
        break
    if not seq.block_table:
        self.block_manager.allocate(seq, num_cached_blocks)
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    num_batched_tokens += seq.num_scheduled_tokens
    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
        seq.status = SequenceStatus.RUNNING   # 完整 prefill 完成
        self.waiting.popleft()
        self.running.append(seq)
    scheduled_seqs.append(seq)

if scheduled_seqs:
    return scheduled_seqs, True   # ⚠️ 有 prefill 就立即返回，不做 decode
```

这段代码的核心逻辑用自然语言描述：

1. 按 FIFO 顺序从 waiting 队列取序列。
2. 对每个序列，先检查有没有前缀缓存命中（`can_allocate` 返回命中的 block 数量），再检查 free block 是否足够。
3. 如果这个序列不能完整放入 token 预算内，且它不是 batch 中的第一条——**打断**。只有第一条序列才能做分块 prefill。
4. 分配 KV cache block，设置本步要处理的 token 数。
5. 如果整个 prompt 都被调度完了（`num_cached_tokens + num_scheduled_tokens == num_tokens`），把序列从 WAITING 提升为 RUNNING。
6. **只要有任何 prefill 被调度，立即返回**，不给 decode 机会。

### 阶段二：Decode 调度（行 57-73）

只有当 prefill 阶段没有调度到任何序列时（waiting 队列为空或所有 waiting 序列因显存不足无法分配），才进入 decode 阶段：

```python
# scheduler.py:57-73
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):    # 需要新 block 吗？
        if self.running:
            self.preempt(self.running.pop())          # 抢占最年轻的序列
        else:
            self.preempt(seq)                         # 自我抢占
            break
    else:                                             # while-else：正常路径
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
assert scheduled_seqs                                 # ⚠️ 断言：至少要有一个
self.running.extendleft(reversed(scheduled_seqs))     # 放回 running 队列
return scheduled_seqs, False
```

decode 调度的逻辑：

1. 按 FIFO 从 running 队列 pop 序列。
2. 检查是否需要新 block（当序列长度 `% block_size == 1` 时，意味着上一步的 token 刚跨入新 block 边界）。
3. 如果 free block 不够：**抢占**——从 running 队列尾部（最晚加入的序列）开始驱逐，释放它的全部 block。
4. 每个序列调度 1 个 token（decode 固定为 1 token/step）。
5. 调度完成后，把所有已调度的序列放回 running 队列头部。

这里有一个关键的 Python 语法细节：**`while/else`**。Python 的 `while` 语句带 `else` 子句时，`else` 只在 while 条件变为 False 时执行，**不在** `break` 退出时执行。因此，当序列自我抢占后 `break`，它不会被加入 `scheduled_seqs`。

## 1.4 关键细节与误区澄清

**误区一：以为 prefill 和 decode 可以在同一个 step 中混合。**

从调用链看，`schedule()` 在行 54-55 有一个 early return：`if scheduled_seqs: return scheduled_seqs, True`。一旦有任何 prefill 调度成功，方法直接返回，根本不会进入行 57 的 decode 循环。prefill 和 decode 是**严格互斥**的。

**误区二：以为所有 waiting 序列都有机会分块 prefill。**

行 42 的条件 `if remaining < num_tokens and scheduled_seqs` 意味着只有 `scheduled_seqs` 为空时（即第一条序列）才允许 token 数超出剩余预算。一旦有了第一条序列，后续的序列必须完整放入，否则被跳过。换言之，一个 prefill batch 最多只有一条分块序列，而且必须是第一条。

**误区三：以为 `running.extendleft(reversed(scheduled_seqs))` 是一个可有可无的操作。**

行 72 至关重要。decode 循环中，每个序列都是 `popleft()` 从 running 中取出的。如果不放回去，它们就永远离开了 running 队列。`extendleft(reversed(...))` 把它们按原始顺序放回队列头部，保持 FIFO 语义。

## 1.5 本章小结

💡 **小结**
- nano-vllm 的 Continuous Batching 采用 **两阶段互斥调度**：一个 step 要么全部 prefill，要么全部 decode。
- 调度策略是 **严格 FIFO**，无优先级、无公平权重。
- 分块 prefill 仅允许 batch 中的第一条序列，后续序列必须完整放入 token 预算。
- 抢占采用 **最年轻优先**（LIFO 驱逐），被抢占的序列全部 KV cache 被释放，需要完整 re-prefill。

---

# 二、分页内存管理：KV cache 的按需分配与回收

## 2.1 设计哲学与核心问题

Continuous Batching 的一个隐含前提是：**每个序列的 KV cache 必须能独立分配和释放**。如果 KV cache 是按最大长度预分配的连续内存，序列结束后的内存碎片无法被新序列利用。

BlockManager 解决的核心问题是：把 KV cache 切成固定大小的 **block**（默认 256 tokens/block），按需分配，按引用计数回收。这使得：
- 序列不需要预分配最大长度的 KV cache。
- 完成的序列的 block 可以立即被新序列复用。
- 共享前缀的多个序列可以引用同一个物理 block，节省显存。

## 2.2 源码入口与关键对象

```
nanovllm/engine/block_manager.py
  - Block：物理 block 元数据（ref_count, hash, token_ids）
  - BlockManager.can_allocate()：检查能否分配 + 前缀缓存命中数
  - BlockManager.allocate()：实际分配 block（含前缀缓存复用）
  - BlockManager.can_append() / may_append()：decode 时检查/追加新 block
  - BlockManager.deallocate()：释放序列的全部 block
  - BlockManager.hash_blocks()：对已填满的 block 计算内容哈希
```

## 2.3 主流程拆解

### 分配检查：`can_allocate()`（行 58-73）

这个方法做两件事：检查有没有前缀缓存命中，以及检查 free block 是否足够。

```python
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks        # 序列总共需要多少 block
    for i in range(seq.num_blocks - 1):    # ⚠️ 不检查最后一个 block（可能不满）
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)          # 链式哈希
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break                                     # 缓存未命中，停止
        num_cached_blocks += 1
        if block_id in self.used_block_ids:           # 已在使用：不需要新 block
            num_new_blocks -= 1
    if len(self.free_block_ids) < num_new_blocks:
        return -1                                     # 显存不足
    return num_cached_blocks
```

关键设计：
- **链式哈希**：每个 block 的哈希依赖前一个 block 的哈希，形成 Merkle 链。相同 token 在不同位置会产生不同哈希。
- **只检查 `num_blocks - 1` 个 block**：最后一个 block 可能未填满，不参与缓存。
- **二次验证**：哈希命中后还比对 `token_ids`，防止哈希碰撞导致错误的缓存复用。
- **已用 block 不占新配额**：如果缓存的 block 已经被其他序列引用（在 `used_block_ids` 中），引用计数 +1 即可，不需要从 free list 分配。

### Decode 时追加：`can_append()` 与 `may_append()`（行 103-108）

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

这是一个精巧的单行表达式。`len(seq) % self.block_size == 1` 是一个 `bool`，在 Python 中 `True == 1`、`False == 0`。所以：
- 当序列长度刚好跨入新 block 边界时（`% block_size == 1`），需要 ≥ 1 个 free block。
- 否则，当前 block 还有空间，不需要新 block。

### 释放：`deallocate()`（行 94-101）

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)    # 移入 free list，但保留 hash
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

**关键设计：惰性驱逐。** `_deallocate_block()` 只是把 block_id 从 `used_block_ids` 移到 `free_block_ids` 的尾部，**不清除** block 的 `hash` 和 `token_ids`。这意味着被驱逐的 block 仍然可以被未来的 `can_allocate` 通过哈希查找重新命中。只有当 block 被真正重新分配给其他序列时（`_allocate_block()`），旧的哈希映射才被删除。

`free_block_ids` 使用 FIFO（`append` 入队、`popleft` 出队），这意味着最近释放的 block 会在队列尾部，被最晚重分配，给前缀缓存留出最大的生存窗口。

## 2.4 关键细节与误区澄清

**误区：以为 `free_block_ids.remove(block_id)` 在 `allocate()` 中是 O(1)。**

`allocate()` 行 87 的 `self.free_block_ids.remove(block_id)` 是在 `deque` 上做线性搜索，复杂度 O(N)，其中 N 是 free list 的长度。当 block 总数很大时（比如数千个 block），这可能成为调度开销的来源。用 `set` + 有序迭代器可以优化到 O(1)，但当前实现选择了简单性。

## 2.5 本章小结

💡 **小结**
- BlockManager 把 KV cache 切成 256-token 的 block，按需分配，按引用计数回收。
- 前缀缓存通过链式 xxhash + token 内容二次验证实现，避免哈希碰撞。
- 惰性驱逐使被释放的 block 保留哈希信息，最大化未来的缓存命中率。
- `free_block_ids` 的 FIFO 顺序天然优先复用最老的 block，保护最近释放的缓存 block。

---

# 三、Sequence 状态机与 Engine 主循环

## 3.1 设计哲学与核心问题

Continuous Batching 要求每个请求有独立的生命周期。传统 static batching 中，一个 batch 的所有请求作为整体管理。而在 iteration-level scheduling 中，每个请求必须独立跟踪自己的状态：正在等待 prefill？正在 decode？已经完成？

`Sequence` 类（`sequence.py:14-83`）就是这个独立生命周期的载体。

## 3.2 源码入口与关键对象

```
nanovllm/engine/sequence.py
  - SequenceStatus：三态枚举（WAITING / RUNNING / FINISHED）
  - Sequence：请求的完整状态容器
  - Sequence.__getstate__ / __setstate__：张量并行 IPC 的序列化优化
```

## 3.3 状态机拆解

```
                    add_request()
                         │
                         ▼
              ┌──────────────────┐
              │     WAITING      │◄─────────────────┐
              │  (is_prefill=T)  │                   │
              └────────┬─────────┘                   │
                       │ prefill 完成                │ preempt()
                       │ (cached + scheduled         │ - 释放全部 block
                       │  == num_tokens)             │ - is_prefill = True
                       ▼                             │
              ┌──────────────────┐                   │
              │     RUNNING      │───────────────────┘
              │  (is_prefill=F)  │
              └────────┬─────────┘
                       │ postprocess()
                       │ token == EOS 或
                       │ num_completion == max_tokens
                       ▼
              ┌──────────────────┐
              │    FINISHED      │
              └──────────────────┘
```

三个状态转换：
1. **WAITING → RUNNING**：`scheduler.schedule()` 行 49，当整个 prompt prefill 完成。
2. **RUNNING → FINISHED**：`scheduler.postprocess()` 行 90，当生成 EOS 或达到 `max_tokens`。
3. **RUNNING → WAITING**：`scheduler.preempt()` 行 76，抢占时回退。

还有一个 **不改变状态但改变内部数据** 的自循环：**WAITING → WAITING（分块 prefill）**。序列的 `block_table` 已分配，`num_cached_tokens` 增加，但 `status` 保持 WAITING。

### IPC 序列化优化（行 72-83）

当使用张量并行时，`Sequence` 对象需要通过 `SharedMemory` 序列化到其他 rank。`__getstate__` 做了一个精巧的优化：

```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)
```

- **Decode 模式**：只发送 `last_token`（一个 int），不发送整个 token 列表。因为 decode 只需要最后一个 token 作为输入。
- **Prefill 模式**：必须发送完整 `token_ids`，因为 worker 需要全部 prompt token。
- **不发送** `status`、`temperature`、`max_tokens`、`ignore_eos`、`seq_id`——这些只在 rank 0 使用。

## 3.4 Engine 主循环（`llm_engine.py`）

`LLMEngine.step()` 是 Continuous Batching 的编排层：

```python
# llm_engine.py:49-55
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

这是一个**完全同步的单线程流水线**：
1. CPU 调度 → 2. GPU 前向 → 3. CPU 后处理

没有 CPU-GPU 重叠：第 N+1 步的调度必须等第 N 步的 GPU 计算完成。这是相对于 vLLM async engine 的一个重要简化。

`generate()` 方法（行 60-90）则是对 `step()` 的循环封装，直到所有序列完成。

## 3.5 本章小结

💡 **小结**
- 每个 `Sequence` 是一个独立的生命周期单元，有自己的状态、block table、token 列表。
- 状态转换由 `Scheduler` 驱动，不存在 `WAITING → FINISHED` 的直接跳转。
- IPC 序列化针对 decode 模式做了极致优化（只发 1 个 int），将数据量从 O(seq_len) 降到 O(1)。
- Engine 主循环是同步的 `schedule → run → postprocess` 流水线，无 CPU-GPU 重叠。

---

# 四、ModelRunner 的双路径：调度决策如何落地为 GPU 计算

## 4.1 设计哲学与核心问题

Scheduler 决定了哪些序列参与当前 step，以及是 prefill 还是 decode。但这些 **CPU 侧的调度决策** 需要转化为 **GPU 侧的张量**，才能驱动模型前向计算。ModelRunner 的职责就是完成这个转换。

核心挑战：prefill 和 decode 的张量 layout 完全不同。Prefill 使用变长打包格式（cumulative sequence lengths），decode 使用固定的 1-token-per-sequence 格式。两种格式对应不同的 Flash Attention kernel。

## 4.2 源码入口与关键对象

```
nanovllm/engine/model_runner.py
  - prepare_prefill()：组装 prefill 输入张量 + 设置全局 Context
  - prepare_decode()：组装 decode 输入张量 + 设置全局 Context
  - run_model()：选择 eager/CUDA graph 执行路径
  - run()：完整的 prepare → forward → sample 流程
```

## 4.3 主流程拆解

### Prefill 输入组装（行 129-170）

```
对于 N 个 prefill 序列，每个有不同的 scheduled token 数：

序列 A: 512 tokens (全新)
序列 B: 256 tokens (分块续传，前 256 已缓存)
序列 C: 128 tokens (前缀缓存命中了 384 tokens)

组装后的张量:
  input_ids:    [A的512个token, B的256个token, C的128个token] → shape [896]
  positions:    [0..511,        256..511,       384..511      ] → shape [896]
  cu_seqlens_q: [0, 512, 768, 896]  → 每个序列的 query 长度的前缀和
  cu_seqlens_k: [0, 512, 512, 512]  → 每个序列的 key 长度的前缀和
  slot_mapping: [A的物理slot×512, B的物理slot×256, C的物理slot×128]
```

当 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 时（说明有序列的 key 长度超过了 query 长度，即有前缀缓存），还会构建 `block_tables` 用于分页注意力从 KV cache 读取缓存的 K/V。

### Decode 输入组装（行 172-188）

```
对于 N 个 decode 序列：

序列 A: 长度 600，最后一个 token 是 42
序列 B: 长度 300，最后一个 token 是 17

组装后的张量:
  input_ids:    [42, 17]       → shape [2]
  positions:    [599, 299]     → shape [2]
  slot_mapping: [A的最新slot, B的最新slot]  → shape [2]
  context_lens: [600, 300]     → shape [2]
  block_tables: [[A的block列表], [B的block列表]]  → shape [2, max_blocks]
```

### 全局 Context：隐式参数传递

两种路径的最后一步都是调用 `set_context()` 将元数据写入全局单例 `_CONTEXT`：

```python
# context.py:16
_CONTEXT = Context()
```

然后在 `Attention.forward()`（`attention.py:59-75`）中通过 `get_context()` 读取，决定使用哪个 Flash Attention kernel 以及传什么参数。

这是一种 **隐式参数传递** 模式——模型的 `forward(input_ids, positions)` 签名不含注意力元数据，元数据通过全局变量"隧穿"到注意力层。好处是避免了在 Qwen3 每一层之间传递调度元数据；代价是不可重入、不可并发、不可独立测试。

### CUDA Graph 与 Eager 的分流（行 196-212）

```python
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))  # eager
    else:
        # CUDA graph 重放
        bs = input_ids.size(0)
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]  # 最近的 ≥ bs
        # ... 填充预分配的 graph 输入张量 ...
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

三条规则：
1. **Prefill 始终 eager**：batch size 不固定，无法用固定 shape 的 CUDA graph。
2. **Decode bs > 512 也 eager**：graph 只在 `[1, 2, 4, 8, 16, 32, ..., 512]` 这些 batch size 上捕获。
3. **Decode bs ≤ 512 用 CUDA graph**：选择最近的 ≥ 实际 bs 的 graph size。padding 部分通过 `slot_mapping.fill_(-1)` 确保 Triton kernel 不写入 KV cache。

CUDA graph 的 padding 会浪费计算：对于 bs=9 使用 graph-16，有 43.75% 的 FLOPS 花在 dummy 位置上。但 graph 避免了 kernel launch 开销，对小 batch 的净效果通常是正面的。

### Prefill 时的 logits 提取优化

在 prefill 时，模型前向计算所有 scheduled token 的 hidden states，但只有每个序列的**最后一个 token** 需要计算 logits（用于预测下一个 token）。`ParallelLMHead.forward()` 使用 `cu_seqlens_q[1:] - 1` 提取每个序列的最后位置：

```python
# embed_head.py:56-61
def forward(self, x):
    context = get_context()
    if context.is_prefill:
        last_indices = context.cu_seqlens_q[1:] - 1
        x = x[last_indices].contiguous()   # 只取每个序列的最后一个 hidden state
    logits = F.linear(x, self.weight)
```

这避免了对所有 prefill token 计算 `vocab_size` 维度的 logits，节省大量计算。

## 4.4 关键细节与误区澄清

**误区：以为 CUDA graph 的 `block_tables` 在每次重放前会被完全清零。**

`model_runner.py:210` 只覆盖了 `[:bs, :context.block_tables.size(1)]`，padding 行（bs 之后的行）保留了上一次的数据。这依赖于 `context_lens` 为 0 使得 Flash Attention 不会读取这些行的 block_tables。如果 `flash_attn_with_kvcache` 的实现发生变化，可能导致读取过期的 block ID。这不是 bug，但是一个隐含的正确性假设。

## 4.5 本章小结

💡 **小结**
- ModelRunner 为 prefill 和 decode 提供完全独立的输入组装路径，不存在混合 batch。
- 全局 `Context` 单例是调度层到注意力层的 "隧道"，避免了元数据穿透模型栈。
- CUDA graph 只用于 decode，prefill 始终 eager。Graph padding 的正确性靠 `slot_mapping=-1` 保证。
- Prefill 时只计算每个序列最后一个 token 的 logits，避免 O(total_tokens × vocab_size) 的计算浪费。

---

# 五、完整主路径串联

## 5.1 完整调用栈

以一次 `llm.generate(["Hello", "World"], SamplingParams(max_tokens=64))` 为例：

```
User: llm.generate(prompts, sampling_params)
  │
  ├─ Step 1: 请求注册
  │     └─ LLMEngine.add_request() × 2
  │           └─ tokenizer.encode() → Sequence() → scheduler.add()
  │           └─ 两个 Sequence 进入 scheduler.waiting 队列
  │
  ├─ Step 2: Prefill 步（可能多步，取决于 token 总量）
  │     └─ LLMEngine.step()
  │           ├─ scheduler.schedule()        → (seqs, is_prefill=True)
  │           │     ├─ block_manager.can_allocate() → 检查前缀缓存 + free block
  │           │     ├─ block_manager.allocate()      → 分配 block，建立 block_table
  │           │     └─ 状态：WAITING → RUNNING
  │           ├─ model_runner.call("run", seqs, True)
  │           │     ├─ prepare_prefill()     → 组装变长打包张量 + set_context()
  │           │     ├─ run_model() [eager]   → model.forward() + compute_logits()
  │           │     │     └─ Attention.forward()
  │           │     │           ├─ store_kvcache()        → 写入 KV cache
  │           │     │           └─ flash_attn_varlen_func → 变长注意力
  │           │     └─ sampler()             → 采样 token_id（仅 rank 0）
  │           └─ scheduler.postprocess()
  │                 ├─ block_manager.hash_blocks() → 哈希已满的 block
  │                 ├─ seq.num_cached_tokens += num_scheduled_tokens
  │                 └─ seq.append_token(token_id)
  │
  ├─ Step 3..N: Decode 步（循环，直到所有序列完成）
  │     └─ LLMEngine.step()
  │           ├─ scheduler.schedule()        → (seqs, is_prefill=False)
  │           │     ├─ block_manager.can_append() → 检查是否需要新 block
  │           │     ├─ block_manager.may_append() → 按需分配新 block
  │           │     └─ seq.num_scheduled_tokens = 1
  │           ├─ model_runner.call("run", seqs, False)
  │           │     ├─ prepare_decode()      → 组装 1-token-per-seq 张量 + set_context()
  │           │     ├─ run_model() [CUDA graph]
  │           │     │     └─ Attention.forward()
  │           │     │           ├─ store_kvcache()          → 写入 1 个 KV entry
  │           │     │           └─ flash_attn_with_kvcache  → 分页查询注意力
  │           │     └─ sampler()
  │           └─ scheduler.postprocess()
  │                 ├─ seq.append_token(token_id)
  │                 └─ if EOS or max_tokens → FINISHED → deallocate → remove from running
  │
  └─ Step N+1: 收集输出
        └─ 按 seq_id 排序 → tokenizer.decode() → 返回文本
```

## 5.2 每一层做了什么

| 层 | 输入 | 输出 | 状态变化 | 通信 | 每步执行？ |
|----|------|------|----------|------|-----------|
| `scheduler.schedule()` | waiting/running 队列 | `(seqs, is_prefill)` | 序列状态转换、block 分配 | 无 | 是 |
| `prepare_prefill/decode()` | `seqs` 列表 | GPU 张量 | 全局 `_CONTEXT` 被设置 | 无 | 是 |
| `model.forward()` | `input_ids, positions` | hidden states | KV cache 被写入 | NCCL all_reduce（TP>1） | 是 |
| `compute_logits()` | hidden states | logits | 无 | NCCL gather（TP>1） | 是 |
| `sampler()` | logits, temperatures | token_ids | 无 | 无（仅 rank 0） | 是 |
| `postprocess()` | seqs, token_ids | 无（就地修改） | token 追加、block 哈希、终止判定 | 无 | 是 |

## 5.3 哪些逻辑不在主路径

| 函数/文件 | 看似相关 | 实际角色 |
|-----------|----------|----------|
| `warmup_model()` | 像是功能预热 | **初始化时执行一次**，用于测量 GPU 显存峰值以计算 KV cache 容量 |
| `capture_cudagraph()` | 像是每步都需要 | **初始化时执行一次**，预捕获所有 batch size 的 CUDA graph |
| `Sequence.__getstate__` | 像是持久化 | **仅在 TP>1 时** 用于共享内存 IPC，单卡不触发 |
| `hash_blocks()` | 每步都调用 | 大多数 decode 步中是 **no-op**（因为 1 个 token 几乎不会恰好填满一个 256-token block） |
| `allocate_kv_cache()` | 像是动态分配 | **初始化时执行一次**，分配一个巨大的连续张量 |

---

# 六、核心机制深挖

## 6.1 抢占机制：粗暴但有效的 recompute 策略

### 问题：KV cache 耗尽时怎么办？

当所有 free block 被用完，但 decode 序列跨入新 block 需要更多空间时，必须有人让步。nano-vllm 的方案：驱逐最年轻（最晚加入 running 队列）的序列，**完全释放它的所有 block**，让它重新 prefill。

```python
# scheduler.py:75-79
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)      # 释放全部 block
    self.waiting.appendleft(seq)            # 插到 waiting 队首
```

### 为什么不是局部释放？

考虑一个序列有 10 个 block，只需要释放 1 个就够了。为什么要全部释放？

因为 KV cache 是一个**因果链**：block 0 的 K/V 影响 block 1 的注意力输出，以此类推。如果只释放 block 10 但保留 block 0-9，序列在 re-prefill 时仍然需要重新计算 block 10 的 KV——但此时 block 0-9 的 KV 数据是"正确的"（它们的内容没变），所以理论上可以跳过。然而，nano-vllm 没有实现这种"部分 re-prefill"逻辑。全部释放 + 全部 re-prefill 是最简单的正确实现。

### 前缀缓存对抢占的缓解

被驱逐的 block 虽然回到了 free list，但它们的**哈希和 token_ids 被保留**。当被抢占的序列重新被 prefill 时，`can_allocate()` 会检查这些 block 的哈希——如果它们还没有被重新分配给其他序列，就可以直接复用，跳过 KV 重计算。

这意味着在低内存压力下（free list 还没有被其他序列消耗），抢占的代价接近于零——序列只需要重新计算最后一个未被缓存的 block。

### 自我抢占与断言崩溃风险

`scheduler.py:60-65` 有一个边界情况：当只有一个 running 序列且它无法 append 时，它会自我抢占：

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())
    else:
        self.preempt(seq)    # 自我抢占
        break                # 跳出 while，不进入 else 分支
```

`break` 导致 `while/else` 的 `else` 不执行，所以 `seq` 不会被加入 `scheduled_seqs`。随后行 71 的 `assert scheduled_seqs` 会**崩溃**。

这在实践中何时发生？当一个序列的总 block 需求恰好等于 `num_kvcache_blocks`（所有 block 都分给了它），它生成到刚好要跨入第 `num_kvcache_blocks + 1` 个 block 时，free list 为空，它是唯一的 running 序列，自我抢占后没有序列可调度——断言失败。

这是一个**真实的 bug**，虽然触发条件苛刻（需要序列长度恰好用满所有 block）。

## 6.2 分块 Prefill：长 prompt 的渐进处理

### 问题：16K token 的 prompt 如何处理？

如果 `max_num_batched_tokens = 16384`，一个 20000-token 的 prompt 无法在一步内完成 prefill。分块 prefill 允许将它分成多步：

- Step 1：处理 token [0, 16384)，`num_scheduled_tokens = 16384`
- Step 2：处理 token [16384, 20000)，`num_scheduled_tokens = 3616`

### 追踪状态

分块的关键在于两个字段：
- `seq.num_cached_tokens`：已经 prefill 到 KV cache 中的 token 数。每步 `postprocess()` 后更新。
- `seq.block_table`：首次 `schedule()` 时分配好**全部**所需的 block，后续步骤不需要重新分配。

`seq.block_table` 是否存在被用作首次 vs 续传的判断依据：
```python
if not seq.block_table:        # 首次调度
    ...
else:                          # 分块续传
    num_tokens = seq.num_tokens - seq.num_cached_tokens
```

### 限制：仅允许 batch 中的第一条

行 42 的限制意味着一个 prefill batch 最多有一条分块序列。如果第一条序列用了 10000 tokens 的预算（被分块到 16384），剩余的 6384 token 预算可以分配给其他更短的序列（只要它们不需要分块）。但如果第二条序列需要 8000 tokens 且剩余预算只有 6384，它不会被分块——直接跳过，等下一步。

### 分块 prefill 的 token 浪费

分块序列虽然在 prefill 步中不产生输出 token，但模型仍然会通过 `compute_logits` 和 `sampler` 对它产生一个 token_id。这个 token_id 在 `postprocess()` 行 86-87 被静默丢弃：

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue    # 跳过 append_token
```

## 6.3 Context 单例：零侵入但脆弱的参数传递

### 问题：调度元数据如何传到注意力层？

模型的 `forward(input_ids, positions)` 签名不包含 `slot_mapping`、`block_tables` 等注意力层需要的元数据。如果要显式传递，需要修改 `Qwen3ForCausalLM.forward()`、`Qwen3DecoderLayer.forward()`、`Qwen3Attention.forward()` 的签名——这对于一个"轻量实现"来说过于侵入。

nano-vllm 的解决方案是全局 `_CONTEXT` 单例：

```python
# context.py
_CONTEXT = Context()    # 全局单例

def set_context(...):   # ModelRunner 写入
    global _CONTEXT
    _CONTEXT = Context(...)

def get_context():      # Attention 读取
    return _CONTEXT
```

### 隐含假设

- **单线程**：`set_context` → `model.forward` → `reset_context` 必须串行执行，不能并发。
- **不可重入**：如果在 `model.forward` 内部再次调用 `set_context`，外层的 context 会被覆盖。
- **每步必须 reset**：`reset_context()` 在 `ModelRunner.run()` 的末尾调用（行 219），确保不会污染下一步。

### 读取点

`get_context()` 被两处读取：
1. `Attention.forward()`（`attention.py:60`）：决定 prefill/decode 路径和参数。
2. `ParallelLMHead.forward()`（`embed_head.py:57-59`）：prefill 时提取最后一个 token 的 hidden state。

---

# 七、显存、性能与通信分析

## 7.1 显存收益范围

| 内容 | 是否因 Continuous Batching 节省 | 原因 |
|------|-------------------------------|------|
| 模型参数 | ❌ | 参数显存与调度无关 |
| KV cache | ✅ 显著 | 分页分配避免了按 max_len 预留；完成的序列立即释放 block |
| 输入 token 张量 | ✅ 轻微 | 变长打包避免了 padding 到 max_seq_len |
| Logits | ✅ 轻微 | Prefill 只计算最后一个 token 的 logits |
| 激活值 | ❌ | 仍然是 batch 中所有 token 的激活值 |
| Optimizer 状态 | N/A | 推理场景无 optimizer |

**真正的显存大头是 KV cache**。分页管理使得 KV cache 利用率从 static batching 的 ~50%（因为要按 max_len 预分配）提升到接近 100%（只分配实际需要的 block）。

但 256-token block size 带来的**内部碎片**不可忽略：每个序列的最后一个 block 平均浪费 128 个 token 的空间。以 Qwen3-0.6B（28 层，16 KV 头，head_dim=64，BF16）为例：

```
每个 token 的 KV 显存 = 2 × 28 × 16 × 64 × 2 = 229,376 bytes ≈ 224 KB
每个序列的平均碎片 = 128 × 224 KB ≈ 28 MB
512 个并发序列的总碎片 ≈ 14 GB
```

14 GB 在 80 GB A100 上占 ~17.5%——这是 block size = 256 的代价。vLLM 默认 block size = 16，碎片只有 ~1/16。

## 7.2 通信开销

### NCCL 通信（仅 TP > 1）

每步前向计算中的 NCCL 操作：
- `VocabParallelEmbedding.forward`：1 次 `all_reduce`
- 每个 Decoder Layer（×28）：2 次 `all_reduce`（`o_proj` 和 `down_proj` 的 `RowParallelLinear`）
- `ParallelLMHead.forward`：1 次 `gather`

**每步总计：1 + 2×28 + 1 = 58 次 NCCL 操作。**

Decode 时（bs=512，hidden=896，BF16）每次 `all_reduce` 传输 512×896×2 = 896 KB。58 次总计 ~50 MB/step。

Prefill 时（16384 tokens）每次 `all_reduce` 传输 16384×896×2 = 28.6 MB。58 次总计 ~1.66 GB/step。

### SharedMemory IPC（仅 TP > 1）

rank 0 通过 1 MB 的共享内存缓冲区向其他 rank 发送 pickle 序列化的 `Sequence` 列表。Decode 时每个序列只发 ~200-300 bytes（因为 `__getstate__` 只发 `last_token`），512 个序列约 150 KB，远小于 1 MB 限制。

## 7.3 性能取舍

| 取舍 | 得到了什么 | 牺牲了什么 |
|------|-----------|-----------|
| Prefill/decode 互斥 | 实现简单，两条独立的输入组装路径 | Decode 延迟在 prefill 期间不受控 |
| Recompute 抢占 | 无需 CPU 端 KV cache 缓冲 | 被抢占的序列必须完整 re-prefill |
| Block size 256 | 减少 block table 大小和管理操作数 | 更大的内部碎片 |
| 同步主循环 | 无并发 bug 风险 | 无 CPU-GPU 重叠，调度/后处理占用 GPU 空闲时间 |
| 全局 Context 单例 | 无需修改模型签名 | 不可并发、不可重入、隐式依赖 |

---

# 八、配置项、边界条件与坑点

| 配置项 | 默认值 | 影响的源码路径 | 行为变化 | 风险/坑点 |
|--------|--------|---------------|----------|-----------|
| `max_num_batched_tokens` | 16384 | `scheduler.schedule()` 行 32 | 控制单步 prefill 的 token 总量上限 | 设太小会导致长 prompt 被分成很多块，增加 prefill 步数 |
| `max_num_seqs` | 512 | `scheduler.schedule()` 行 30/58 | 控制最大 batch size | 设太大可能导致 CUDA graph padding 浪费 |
| `kvcache_block_size` | 256 | `block_manager.py` 全局 | block 粒度 | 必须是 256 的倍数；越大碎片越严重，越小管理开销越大 |
| `gpu_memory_utilization` | 0.9 | `model_runner.py:113` | 决定 KV cache 可用显存 | 设到 0.95+ 可能导致 CUDA OOM（未留足激活值空间） |
| `max_model_len` | 4096 | `config.py:25` | 截断到 `hf_config.max_position_embeddings` | 序列超过此长度时行为未定义（RoPE 越界） |
| `enforce_eager` | False | `model_runner.py:197` | 禁用 CUDA graph | 调试用；正常推理应保持 False |

### 静默失效条件

- **`temperature ≤ 1e-10`**：`Sampler` 内部使用 `logits / temperature` 后取 `exponential_().argmax()`。temperature 接近 0 会导致数值溢出，但 `SamplingParams` 没有断言阻止（只有 `temperature > 0` 的隐含假设）。
- **`kvcache_block_size` 不是 256 的倍数**：`config.py:22` 有断言 `assert self.kvcache_block_size % 256 == 0`，但报错信息不明确。
- **多实例运行**：NCCL rendezvous 硬编码为 `tcp://localhost:2333`（`model_runner.py:26`），共享内存名称硬编码为 `"nanovllm"`（`model_runner.py:43`）。同一台机器不能运行两个 nano-vllm 实例。

### 不兼容组合

- `tensor_parallel_size > 1` + 非 Qwen3 模型：只实现了 Qwen3。
- `max_num_seqs > 512` + `enforce_eager=False`：CUDA graph 的 batch size 上限是 512（`model_runner.py:226`），超过部分会 fallback 到 eager。

---

# 九、测试、示例与覆盖缺口

## 9.1 已覆盖路径

| 示例/脚本 | 覆盖的行为 | 说明 |
|-----------|-----------|------|
| `example.py` | 基本推理 | 2 条 prompt，`enforce_eager=True`，TP=1，`max_tokens=256` |
| `bench.py` | 吞吐测试 | 256 条随机长度序列，`enforce_eager=False`，`ignore_eos=True` |

## 9.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|--------|---------------|----------|
| 抢占路径 | ❌ | 自我抢占断言崩溃（行 71）在极端显存压力下会触发 |
| 分块 prefill | ❌ | 单条超长 prompt 的多步 prefill 未验证 |
| 前缀缓存 | ❌ | 大量共享前缀的序列是否正确复用 block 未验证 |
| 多卡 TP | ❌ | 共享内存 IPC 的序列化边界（1 MB 限制）未测试 |
| KV cache 耗尽 | ❌ | 极端情况下的抢占级联和调度行为未验证 |
| `max_num_seqs` 边界 | ❌ | 恰好 512 条序列同时 decode 时的 CUDA graph 选择 |
| 序列长度达到 `max_model_len` | ❌ | 超长序列是否正确终止 |
| 保存/加载/resume | N/A | 不支持 |
| 性能回归 | ❌ | 无性能基准测试的 CI/CD |

**项目没有 test suite 和 linter。** 所有验证都依赖手动运行 `example.py` 和 `bench.py`。

---

# 十、局限性与已知优化点

## 10.1 硬约束

- **仅支持 Qwen3**：`model_runner.py:31` 硬编码 `Qwen3ForCausalLM`。
- **Block size 必须是 256 的倍数**：`config.py:22`。
- **不支持 greedy decoding**：`temperature > 0` 是隐含假设（`Sampler` 使用 exponential distribution trick）。
- **TP size ≤ 8**：`config.py:23`。
- **单机部署**：NCCL 地址硬编码 `localhost`。
- **无 beam search**：只有采样解码，没有 `SequenceGroup` 概念。

## 10.2 维护成本

- **全局 Context 单例**：任何尝试并发化或多实例化的改动都需要先解决 `_CONTEXT` 的全局可变状态问题。
- **`Sequence.block_size` 类级别变量**：由 `LLMEngine.__init__` 外部设置（`llm_engine.py:21`），如果创建两个不同 block_size 的 Engine 实例，后者会覆盖前者。
- **`Sequence.counter` 类级别计数器**：跨实例递增，不会重置。
- **SharedMemory 名称冲突**：硬编码 `"nanovllm"` 使得同一台机器不能运行多个实例。

## 10.3 性能瓶颈

- **Prefill/decode 互斥**：这是最大的性能瓶颈。在高 QPS 场景下，频繁的 prefill step 会让 decode 序列反复等待，降低平均 time-to-next-token。
- **同步主循环**：每步的 CPU 开销（调度 + 后处理 + 数据组装）不与 GPU 重叠。对于小模型（GPU 前向 ~1ms），CPU 开销可能占到 step 时间的 25-75%。
- **`running.remove(seq)` O(n)**：`scheduler.py:92` 对每个 finished 序列做线性扫描。512 个序列时虽然不是瓶颈，但在高吞吐场景下会累积。
- **`free_block_ids.remove(block_id)` O(n)**：`block_manager.py:87` 在 allocate 路径中的线性搜索。
- **Block size 256 碎片**：每个序列平均浪费 128 个 token 的 KV cache 空间。
- **ParallelLMHead.gather**：TP>1 时，全词表 logits 的 gather 操作数据量大（vocab=151936, bs=512 时 ~297 MB）。

## 10.4 已知优化点

1. **混合 prefill/decode batch**：允许在同一个 step 中同时处理 prefill token 和 decode token。需要统一 `prepare_prefill` 和 `prepare_decode` 为一个路径，并使用支持混合 batch 的 Flash Attention 接口。这是 vLLM 的做法。
2. **CPU-GPU 重叠**：在 GPU 执行 step N 时，CPU 可以提前执行 step N+1 的调度和数据组装。需要引入 async engine。
3. **Swap 抢占**：将被抢占序列的 KV cache swap 到 CPU 内存，避免 re-prefill 的计算浪费。
4. **更小的 block size**：16 或 64 token 的 block 可以大幅降低碎片，代价是 block table 更大和管理操作更频繁。
5. **异步 IPC**：用 CUDA IPC 或 tensor-based 共享替代 pickle + SharedMemory，减少序列化开销。
6. **预算最优化**：SJF（最短作业优先）替代 FIFO，减少不必要的抢占。
7. **断言修复**：行 71 的 `assert scheduled_seqs` 应改为优雅降级（重新进入 prefill 路径）。

---

# 小结与展望

nano-vllm 的 Continuous Batching 实现可以用几个关键词概括。

**关键词一：两阶段互斥调度**

每个 step 要么纯 prefill、要么纯 decode，通过 early return 实现互斥。这是对 vLLM 混合 batch 的极端简化，将 ~1200 行框架中最复杂的调度逻辑压缩到了 50 行 Python。代价是 decode 延迟在 prefill 期间不受控。

**关键词二：iteration-level 的 batch 重组**

每一步都根据当前 waiting/running 队列的状态重新组 batch。序列独立进出，完成即释放。这是 Continuous Batching 区别于 static batching 的本质——batch 不再是一个固定生命周期的整体。

**关键词三：分页 + 惰性驱逐 + 链式哈希**

KV cache 的管理融合了三个机制：分页避免预分配浪费，惰性驱逐让被释放的 block 保留哈希以最大化前缀缓存命中率，链式哈希确保位置敏感的缓存正确性。三者协同，使得抢占后的 re-prefill 代价可以被前缀缓存大幅缓解。

**关键词四：隐式 Context 隧穿**

调度元数据通过全局 `_CONTEXT` 单例从 ModelRunner 传递到 Attention 层，避免了在模型栈中逐层传递的签名污染。这是一个在 ~1200 行约束下的务实选择，但也是阻碍未来并发化或多实例化的技术债。

**关键词五：recompute 换简单性**

抢占策略选择了最简单的 recompute：释放全部 block、重新 prefill。没有 CPU swap、没有部分 recompute、没有优先级。在低 QPS 场景下这足够好，在高 QPS 场景下这是性能杀手。

---

**适合场景**：学习 vLLM 核心机制、单机少量并发推理、快速原型验证、教学演示。

**不适合场景**：生产级推理服务（缺少混合 batch、异步 engine、swap 抢占）、高 QPS 在线服务（decode 延迟不可控）、多模型并行（全局可变状态冲突）。

**与 vLLM 的取舍**：nano-vllm 用 ~1200 行 Python 实现了 vLLM 的核心调度语义（iteration-level scheduling + paged KV cache + prefix caching），但省略了所有生产级特性（混合 batch、swap、async engine、speculative decoding、beam search、多架构支持）。它不是 vLLM 的替代品，而是理解 vLLM 设计哲学的最佳入口。

**后续值得继续走读的方向**：
- Qwen3 模型层的 `packed_modules_mapping` 和 `weight_loader` 如何实现权重加载时的自动融合。
- CUDA graph 的 capture/replay 机制与 `graph_pool` 共享的实现细节。
- Flash Attention 的 `block_table` 参数在 paged prefill 和 paged decode 中的不同语义。
