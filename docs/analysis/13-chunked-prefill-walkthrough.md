# Nano-vLLM 源码走读：Chunked Prefill 实现解析

> 在上一篇 Prefix Cache 的走读中，我们看到了 Nano-vLLM 如何通过内容哈希实现 KV Cache 的跨请求复用。但 Prefix Cache 解决的是"重复前缀"的问题——如果一个请求本身就很长（比如 32K token 的文档理解任务），即使没有重复前缀，它的 prefill 阶段也会一次性吃掉巨量的激活显存，并且长时间阻塞其他序列的 decode 推理。Chunked Prefill 正是为了解决这个问题而引入的：将一个长序列的 prefill 阶段拆成多个较小的 chunk，分多步完成。本文不展开 Chunked Prefill 的学术原理（参见 Sarathi-Serve 论文），而是聚焦源码，分析 Nano-vLLM 如何把这个特性接入现有调度链路，以及它带来了哪些收益和限制。

---

## 前言

### 业务 / 工程背景

在 LLM 推理的 continuous batching 场景中，一个新请求到达后必须先完成 prefill（将全部 prompt token 的 KV 值写入 cache），然后才能进入逐 token 的 decode 阶段。对于短 prompt（几百 token），prefill 只需几十毫秒，对系统影响不大。但当 prompt 长度达到数千甚至数万 token 时，一次 prefill 可能耗时数百毫秒到数秒，期间：

1. **激活显存暴涨**：前向计算的中间张量（Q/K/V 投影、MLP 中间态等）与 token 数成正比，可能触发 OOM。
2. **decode 被阻塞**：在 Nano-vLLM 的两阶段调度设计中，只要有 prefill 需要做，所有 decode 序列都必须等待。一个长 prefill 意味着所有正在生成的序列都"卡住"了。

### 核心矛盾

**一句话概括**：调度器希望一次处理尽可能多的 token 以提升 GPU 利用率，但单步处理过多 token 会撑爆激活显存并阻塞其他序列。

Chunked Prefill 的核心思想是引入 **token 预算（token budget）** 机制：每步最多处理 `max_num_batched_tokens` 个 token。如果一个序列的 prompt 超过这个预算，就把它拆成多个 chunk，分多步完成 prefill。这把一个 O(L) 的显存峰值问题降为 O(B) 的常量问题（B 为预算）。

### 本文主线

本文按照 Chunked Prefill 的**生命周期**组织，分为以下几章：

1. **配置与入口**：用户如何开启这个特性，它的控制旋钮是什么。
2. **调度层：chunk 的诞生**：调度器如何决定 chunk 边界、如何管理部分 prefill 的序列状态。
3. **执行层：chunk 的消化**：ModelRunner 如何为 chunk 准备输入，Attention 如何正确处理不对称的 Q/K 长度。
4. **后处理与状态推进**：一个 chunk 完成后系统状态如何更新，最终如何过渡到 decode。
5. **完整主路径串联**：用一个具体例子端到端走通全部流程。
6. **核心机制深挖**：prefix cache 复用、block 分配策略、preemption 交互。
7. **显存、性能与通信分析**。
8. **配置项、边界条件与坑点**。
9. **bug 演进史**：从初版到现版的 4 次修复。
10. **局限性与优化方向**。

### 不展开的内容

本文不讲 Flash Attention 的 IO 复杂度推导、不讲 Sarathi-Serve 的 stall-free scheduling 理论、不讲 PagedAttention 的学术原理（前文已覆盖）。只讲 Nano-vLLM 的源码实现。

### 核心文件表

| 文件 | 职责 |
|---|---|
| `nanovllm/config.py` | `max_num_batched_tokens` 的定义与默认值 |
| `nanovllm/engine/scheduler.py` | chunk 的诞生地：预算管理、状态转换、postprocess |
| `nanovllm/engine/sequence.py` | `num_cached_tokens` / `num_scheduled_tokens` 追踪 chunk 进度 |
| `nanovllm/engine/block_manager.py` | 全量 block 预分配、增量 hash |
| `nanovllm/engine/model_runner.py` | `prepare_prefill` 的 chunk 偏移与 slot mapping 计算 |
| `nanovllm/utils/context.py` | `cu_seqlens_q` / `cu_seqlens_k` 的不对称传递 |
| `nanovllm/layers/attention.py` | Flash Attention 的 varlen 调用与 block_table 切换 |
| `nanovllm/layers/embed_head.py` | `ParallelLMHead` 的 last-token 提取逻辑 |

---

## 一、配置与入口：一个"不需要开关"的特性

### 1.1 设计哲学与核心问题

Chunked Prefill 最常见的实现方式是提供一个显式的 `enable_chunked_prefill` 开关。但 Nano-vLLM 采用了一种更简洁的设计：**没有开关**。Chunked Prefill 始终隐式生效，其行为完全由 `max_num_batched_tokens` 这一个旋钮控制。

如果一个序列的待处理 token 数超过了当前步的剩余 token 预算，它就会被自动 chunk——不需要用户做任何额外配置。这意味着 Chunked Prefill 不是一个"可选特性"，而是调度器的**内在行为**。

### 1.2 源码入口与关键对象

```text
nanovllm/config.py
  - Config.max_num_batched_tokens = 16384  (line 9)：token 预算，唯一控制旋钮
  - Config.max_model_len = 4096            (line 11)：最大序列长度

nanovllm/engine/llm_engine.py
  - LLMEngine.__init__                     (line 17-20)：将 kwargs 转发到 Config
```

### 1.3 关键细节与误区澄清

> **误区 #1**："`max_num_batched_tokens` 只控制 prefill batch 的大小"。
>
> 错误。这个参数还间接决定了 KV Cache 的可用容量。在 `ModelRunner.warmup_model()` 中（`model_runner.py:91-101`），warmup 的序列长度取 `min(max_num_batched_tokens, max_model_len)`。warmup 的峰值显存直接影响 `allocate_kv_cache()` 能分配多少 block：
>
> ```python
> config.num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
> ```
>
> 因此，调小 `max_num_batched_tokens` 不仅会触发更多 chunk，还会因为 warmup 峰值降低而让出更多显存给 KV Cache，从而支撑更高的 decode 并发。

> **误区 #2**："`max_num_batched_tokens` 设成 `max_model_len` 就等于关闭 Chunked Prefill"。
>
> 正确——但有一个细节。默认配置下 `max_num_batched_tokens=16384` 而 `max_model_len=4096`，预算远大于最大序列长度，所以 **Chunked Prefill 在默认配置下永远不会触发**。只有当用户显式设置 `max_model_len` 超过 `max_num_batched_tokens`（例如 `max_model_len=32768`），或者调低 `max_num_batched_tokens`（例如 `4096`），chunk 才会真正发生。

### 1.4 💡 小结

- Chunked Prefill 没有独立的开关，完全由 `max_num_batched_tokens` 隐式控制。
- 默认配置下 Chunked Prefill 不会触发（预算 16384 > 最大序列长度 4096）。
- `max_num_batched_tokens` 具有双重效应：控制 chunk 粒度 + 影响 KV Cache 容量。

---

## 二、调度层：Chunk 的诞生

### 2.1 设计哲学与核心问题

调度器是 Chunked Prefill 的决策中心。它要回答一个看似简单但实际很微妙的问题：**这一步应该处理哪些序列的哪些 token？**

具体来说，它需要在以下约束之间做平衡：
- **token 预算**：每步最多处理 `max_num_batched_tokens` 个 token。
- **序列数上限**：每步最多调度 `max_num_seqs` 个序列。
- **block 可用性**：必须能为新序列分配 KV Cache block。
- **部分 prefill 延续**：已经 chunk 过一半的序列需要继续完成。

### 2.2 源码入口与关键对象

```text
nanovllm/engine/scheduler.py
  - Scheduler.schedule()        (line 25-73)：核心调度逻辑，prefill + decode 两阶段
  - Scheduler.postprocess()     (line 81-92)：更新 num_cached_tokens，决定是否 append token
  - Scheduler.preempt()         (line 75-79)：preemption 时的状态重置
```

### 2.3 主流程拆解

`schedule()` 方法的 prefill 阶段（`scheduler.py:29-55`）是一个 `while` 循环，逐一处理 `waiting` 队列中的序列。其核心逻辑可以用伪代码表示为：

```
remaining_budget = max_num_batched_tokens

WHILE waiting 不为空 AND 已调度序列数 < max_num_seqs:
    seq = waiting 队头

    IF remaining_budget == 0:
        BREAK

    IF seq 没有 block_table（首次调度）:
        num_cached_blocks = block_manager.can_allocate(seq)
        IF 分配失败:
            BREAK
        num_tokens = seq.num_tokens - num_cached_blocks × block_size
    ELSE（已部分 prefill，有 block_table）:
        num_tokens = seq.num_tokens - seq.num_cached_tokens

    IF remaining_budget < num_tokens AND 已有其他序列被调度:
        BREAK    // ← 关键：只允许第一个序列被 chunk

    IF seq 没有 block_table:
        block_manager.allocate(seq, num_cached_blocks)

    seq.num_scheduled_tokens = min(num_tokens, remaining_budget)
    remaining_budget -= seq.num_scheduled_tokens

    IF num_cached_tokens + num_scheduled_tokens == num_tokens:
        seq 状态 → RUNNING，从 waiting 移到 running
    // 否则 seq 留在 waiting，等待下一步继续 chunk

    加入 scheduled_seqs

IF 有任何 prefill 序列被调度:
    RETURN (scheduled_seqs, is_prefill=True)   // ← decode 不执行

// 否则进入 decode 阶段...
```

这段伪代码揭示了几个关键设计决策：

**决策 1：只允许第一个序列被 chunk**（`scheduler.py:42`）

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

这个约束意味着：如果 `waiting` 队列中有多个序列，只有排在最前面的那个可以被部分 prefill。后面的序列要么完整塞进剩余预算，要么不被调度。这大大简化了状态管理——在任何时刻，最多只有一个序列处于"部分 prefill"状态。

**决策 2：部分 prefill 的序列留在 waiting 队列**（`scheduler.py:48-52`）

当一个序列的 prefill 没有在本步完成时，它**不会**从 `waiting` 移到 `running`。下一次 `schedule()` 调用时，它仍然是 `waiting[0]`，但此时 `seq.block_table` 非空（首次调度时已分配 block），所以走 `else` 分支（`scheduler.py:40-41`），用 `num_cached_tokens` 计算剩余 token 数。

**决策 3：prefill 优先，decode 完全被阻塞**（`scheduler.py:54-55`）

```python
if scheduled_seqs:
    return scheduled_seqs, True
```

只要有任何 prefill 序列被调度（哪怕只是一个 chunk），decode 阶段就完全不执行。这意味着 **Chunked Prefill 在本实现中不会与 decode 交替执行**——一个多 chunk 的序列会独占调度器直到 prefill 完成。这是与 vLLM 的关键区别。

### 2.4 关键细节与误区澄清

> **误区 #3**："Chunked Prefill 允许 prefill 和 decode 交替进行，降低 decode 延迟。"
>
> 在 vLLM 中是这样的——vLLM 可以在同一个 batch 中混合 prefill chunk 和 decode token。但在 Nano-vLLM 中，**prefill 和 decode 永远不在同一步执行**。一个需要 4 个 chunk 的长序列会连续独占调度器 4 步，期间所有 decode 序列都被阻塞。Chunked Prefill 在这里的主要收益是**控制激活显存峰值**，而非降低 decode 延迟。

> **误区 #4**："部分 prefill 的序列在 waiting 和 scheduled_seqs 中各有一份拷贝。"
>
> 不是拷贝，是同一个对象的引用。序列留在 `self.waiting`（未被 `popleft()`），同时被 `append` 到返回的 `scheduled_seqs` 列表。这意味着 ModelRunner 处理的序列和 waiting 队列中的序列是同一个 Python 对象。`postprocess()` 对它的修改（比如递增 `num_cached_tokens`）会立即反映在队列中。

### 2.5 💡 小结

- 调度器通过 token 预算和 `min(num_tokens, remaining)` 实现自动 chunk。
- 每步最多只有一个序列被 chunk（`waiting[0]`），其余序列必须完整塞入预算。
- 部分 prefill 的序列留在 `waiting` 队列，靠 `block_table` 是否为空来区分首次 vs. 延续。
- prefill 和 decode 严格互斥——Chunked Prefill 不降低 decode 延迟，只控制显存峰值。

---

## 三、执行层：Chunk 的消化

### 3.1 设计哲学与核心问题

调度器只决定"处理哪些 token"，真正的执行在 ModelRunner 和 Attention 层。Chunked Prefill 对执行层提出了一个非平凡的问题：**如何让模型正确地对一个"不从 position 0 开始"的 token 子序列做 attention？**

核心困难在于：
- **位置编码**：chunk 2 的 token 位置是 `[512, 513, ..., 1023]`，不是 `[0, 1, ..., 511]`。
- **KV Cache 读写**：chunk 2 需要把自己的 K/V 写入 cache 的正确位置，同时读取 chunk 1 已经写好的 K/V。
- **Attention 掩码**：chunk 2 的 Q 只有 512 个 token，但 K 有 1024 个 token（chunk 1 的 512 + chunk 2 的 512）。`cu_seqlens_q ≠ cu_seqlens_k`。

### 3.2 源码入口与关键对象

```text
nanovllm/engine/model_runner.py
  - ModelRunner.prepare_prefill()  (line 129-170)：chunk 偏移、slot mapping、cu_seqlens 构建
  - ModelRunner.run_model()        (line 195-212)：prefill 始终 eager mode

nanovllm/layers/attention.py
  - Attention.forward()            (line 59-75)：prefill 的 varlen attention 调用
  - store_kvcache()                (line 33-40)：Triton 写 KV Cache

nanovllm/layers/embed_head.py
  - ParallelLMHead.forward()       (line 56-66)：用 cu_seqlens_q 提取 last-token logits
```

### 3.3 主流程拆解

`prepare_prefill()` 是理解 Chunked Prefill 执行逻辑的关键函数。对于每个被调度的序列，它做如下处理（`model_runner.py:138-161`）：

```python
start = seq.num_cached_tokens         # chunk 的起始位置（绝对位置）
seqlen_q = seq.num_scheduled_tokens   # 本 chunk 的 Q 长度
end = start + seqlen_q                # chunk 的结束位置
seqlen_k = end                        # K 的长度 = 已 cache + 本 chunk
```

这四行代码是整个 Chunked Prefill 执行层的核心。让我们逐一拆解它们的含义：

**`start = seq.num_cached_tokens`**：对于 chunk 1（首次 prefill），`start=0`。对于 chunk 2，`start=512`（假设 chunk 1 处理了 512 个 token）。这个值由 `postprocess()` 在上一步递增。

**`seqlen_q = seq.num_scheduled_tokens`**：本 chunk 实际要计算的 token 数，即 Q（query）的长度。

**`seqlen_k = end = start + seqlen_q`**：K（key）的长度不是 `seqlen_q`，而是从位置 0 到当前 chunk 末尾的所有 token。对于 chunk 2（start=512, seqlen_q=512），`seqlen_k=1024`。这意味着 attention 需要让 chunk 2 的 512 个 Q token attend 到 1024 个 K token（包括 chunk 1 已经在 cache 中的 512 个）。

这种 **Q/K 不对称**通过 `cu_seqlens_q` 和 `cu_seqlens_k` 两个独立数组传递给 Flash Attention：

```python
cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)   # 按 chunk 大小累积
cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)   # 按完整上下文累积
```

当 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 时（即存在已 cache 的 token），`prepare_prefill` 会构建 `block_tables`（`model_runner.py:162-163`）。这个条件触发 Attention 层的一个关键分支。

**进入 Attention 层**（`attention.py:59-75`）：

```python
if context.is_prefill:
    if context.block_tables is not None:    # 有已 cache 的 token
        k, v = k_cache, v_cache             # K/V 从 paged cache 读
    o = flash_attn_varlen_func(q, k, v, ..., block_table=context.block_tables)
```

关键理解：当 `block_tables` 不为 `None` 时，K 和 V 被替换为整个 KV Cache tensor（`k_cache, v_cache`），Flash Attention 通过 `block_table` 参数实现 paged attention——从物理 block 中读取所有已 cache 的 KV 条目（包括之前 chunk 写入的和本 chunk 刚刚由 `store_kvcache` 写入的）。

这段代码的精妙之处在于：**它与 Prefix Cache 共享完全相同的代码路径**。无论是因为 Prefix Cache 命中导致部分 token 已在 cache 中，还是因为 Chunked Prefill 导致前一个 chunk 的 token 已在 cache 中，触发条件相同（`cu_seqlens_k > cu_seqlens_q`），处理方式相同（切换到 paged KV cache 读取）。

**Slot mapping 的 chunk 偏移**（`model_runner.py:151-161`）：

slot mapping 只覆盖当前 chunk 的 token 位置，不包括之前已 cache 的位置：

```python
start_block = start // self.block_size
end_block = (end + self.block_size - 1) // self.block_size
```

对于 chunk 2（start=512, end=1024, block_size=256）：`start_block=2, end_block=4`。只生成 block 2 和 block 3 的 slot mapping，跳过已由 chunk 1 填充的 block 0 和 block 1。

**位置编码**（`model_runner.py:144`）：

```python
positions.extend(range(start, end))   # 绝对位置，不是 chunk 内相对位置
```

chunk 2 的位置是 `[512, 513, ..., 1023]`。这保证了 Rotary Embedding 使用正确的绝对位置频率。

**LM Head 的 last-token 提取**（`embed_head.py:58-60`）：

```python
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

注意这里用的是 `cu_seqlens_q`（chunk 大小）而非 `cu_seqlens_k`（完整上下文）。这是正确的——模型前向计算的输出 tensor 长度等于 Q 的总长度（即 chunk 内的 token 数），last-token 提取应该用 Q 的累积长度。

### 3.4 关键细节与误区澄清

> **误区 #5**："chunk 1（首次 prefill，start=0）也需要 block_table 来做 paged attention。"
>
> 错误。chunk 1 的 `seqlen_q == seqlen_k`（都等于 chunk 大小），所以 `cu_seqlens_k[-1] == cu_seqlens_q[-1]`，`block_tables` 为 `None`。此时 Flash Attention 使用直接传入的 Q/K/V tensor，不走 paged attention。只有 chunk 2 开始才需要 block_table——因为需要读取 chunk 1 已写入 cache 的 KV。

> **误区 #6**："不完整的 chunk（没有完成全部 prefill）不会产生 logits / token。"
>
> 错误。模型前向计算和 LM Head **总是**执行的，包括不完整的 chunk。`run()` 方法（`model_runner.py:214-220`）无条件地调用 `run_model` → `compute_logits` → `sampler`。对于不完整的 chunk，sampler 确实产生了一个 token——但这个 token 在 `postprocess()` 中被丢弃（`scheduler.py:86-87`）。这意味着 LM Head 和 sampler 的计算是浪费的。

### 3.5 💡 小结

- `prepare_prefill` 通过 `start = num_cached_tokens` 实现 chunk 偏移，正确设置绝对位置、slot mapping 和 Q/K 不对称的 cu_seqlens。
- chunk 2+ 走 paged attention 路径，与 Prefix Cache 共享同一条代码路径。
- chunk 1 走标准 self-attention，不需要 block_table。
- 位置编码使用绝对位置，保证 Rotary Embedding 正确。
- 不完整 chunk 的 LM Head / sampler 计算是浪费的——一个可以优化的点。

---

## 四、后处理与状态推进

### 4.1 设计哲学与核心问题

每个 chunk 完成后，系统需要回答两个问题：
1. 这个序列的 prefill 完成了吗？
2. 如果没完成，下一步应该从哪里继续？

这些问题由 `postprocess()` 方法回答。

### 4.2 源码入口与关键对象

```text
nanovllm/engine/scheduler.py
  - Scheduler.postprocess()     (line 81-92)：状态推进的核心
  - BlockManager.hash_blocks()  (line 110-120)：增量 block 哈希
```

### 4.3 主流程拆解

`postprocess()` 对每个已调度的序列执行以下操作（`scheduler.py:81-92`）：

```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)       # Step 1: hash 已完成的 block
        seq.num_cached_tokens += seq.num_scheduled_tokens  # Step 2: 推进水位线
        seq.num_scheduled_tokens = 0              # Step 3: 重置本步计划
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue                              # Step 4: 未完成 → 跳过 token append
        seq.append_token(token_id)                # Step 5: 完成 → append 生成的 token
        # Step 6: 检查是否结束（EOS / max_tokens）
```

**Step 1：增量 block 哈希**

`hash_blocks()`（`block_manager.py:110-120`）只哈希在本 chunk 中被**完整填充**的 block：

```python
start = seq.num_cached_tokens // self.block_size
end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
```

注意这里用的是整除，所以只有完整的 block 才会被计入 `[start, end)` 范围。跨 chunk 边界的不完整 block 要等到下一个 chunk 填满它后才会被哈希。

**Step 4：chunk 延续的关键守卫**

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
```

当本步是 prefill 且 `num_cached_tokens` 仍小于 `num_tokens` 时，说明 prefill 未完成。此时 `continue` 跳过 `append_token`——sampler 产生的 token 被丢弃。序列留在 `waiting` 队列中（因为 `schedule()` 没有把它移到 `running`），下一步继续处理。

### 4.4 关键细节与误区澄清

> **误区 #7**："`hash_blocks` 在 `postprocess` 中先于 `num_cached_tokens` 更新调用，这是否意味着它使用了错误的范围？"
>
> 仔细看代码：`hash_blocks` 用 `num_cached_tokens`（还未更新，是上一步的值）和 `num_scheduled_tokens`（本步计划值）计算范围。`start = num_cached_tokens // block_size` 是上一步结束时的 block 边界，`end = (num_cached_tokens + num_scheduled_tokens) // block_size` 是本步结束时的 block 边界。这是正确的——它精确覆盖了本 chunk 期间被完整填充的 block。
>
> 但这里存在一个隐蔽的 bug（稍后在 "bug 演进史" 中详述）：在 **decode 阶段**，`hash_blocks` 在 `append_token` 之前调用，而 `seq.block(i)` 读取的是 `seq.token_ids` 的内容——此时新生成的 token 还没被 append。如果 decode 恰好填满了一个 block 的最后一个 slot，hash 会基于缺少最后一个 token 的内容计算。这会降低 Prefix Cache 命中率，但不会导致错误计算。

### 4.5 💡 小结

- `postprocess` 通过递增 `num_cached_tokens` 推进 chunk 水位线。
- 未完成的 prefill 通过 `continue` 跳过 token append——sampler 的输出被丢弃。
- `hash_blocks` 只哈希完整填充的 block，跨 chunk 边界的 block 延迟哈希。
- 存在一个 decode 阶段的 hash timing bug，影响 Prefix Cache 命中率但不影响正确性。

---

## 五、完整主路径串联

本章用一个具体例子串联前面拆散的机制。

### 5.1 场景设定

假设有一个 2048 token 的 prompt，配置如下：
- `max_num_batched_tokens = 512`
- `block_size = 256`
- 无 Prefix Cache 命中
- 设 block_table = `[B0, B1, B2, B3, B4, B5, B6, B7]`

这个序列需要 `ceil(2048 / 512) = 4` 个 chunk。

### 5.2 完整调用栈

```
User: LLM.generate(prompt_of_2048_tokens, ...)
  │
  ├─ LLMEngine.add_request() → Sequence 创建，加入 waiting 队列
  │
  ├─ Round 1 (Chunk 1)
  │   ├─ scheduler.schedule()
  │   │   ├─ seq = waiting[0], block_table 为空 → 首次调度
  │   │   ├─ can_allocate() → num_cached_blocks=0
  │   │   ├─ num_tokens = 2048 - 0 = 2048
  │   │   ├─ remaining = 512 < 2048, 但 scheduled_seqs 为空 → 允许 chunk
  │   │   ├─ allocate(seq, 0) → 分配全部 8 个 block
  │   │   ├─ seq.num_scheduled_tokens = min(2048, 512) = 512
  │   │   ├─ 0 + 512 ≠ 2048 → seq 留在 waiting
  │   │   └─ return ([seq], is_prefill=True)
  │   │
  │   ├─ model_runner.run(seqs, True)
  │   │   ├─ prepare_prefill: start=0, seqlen_q=512, seqlen_k=512
  │   │   │   ├─ input_ids: seq[0:512], shape [512]
  │   │   │   ├─ positions: [0..511]
  │   │   │   ├─ cu_seqlens_q = [0, 512], cu_seqlens_k = [0, 512]
  │   │   │   ├─ slot_mapping: block B0 和 B1 的全部 slot
  │   │   │   └─ block_tables: None (cu_seqlens_k == cu_seqlens_q)
  │   │   ├─ Attention: flash_attn_varlen_func(q, k, v, ...) ← 标准 self-attention
  │   │   └─ store_kvcache: 写入 block B0, B1
  │   │
  │   └─ scheduler.postprocess()
  │       ├─ hash_blocks: block 0,1 被哈希
  │       ├─ num_cached_tokens: 0 → 512
  │       └─ 512 < 2048 → continue, token 被丢弃
  │
  ├─ Round 2 (Chunk 2)
  │   ├─ scheduler.schedule()
  │   │   ├─ seq = waiting[0], block_table 非空 → 延续调度
  │   │   ├─ num_tokens = 2048 - 512 = 1536
  │   │   ├─ seq.num_scheduled_tokens = min(1536, 512) = 512
  │   │   └─ 512 + 512 = 1024 ≠ 2048 → 留在 waiting
  │   │
  │   ├─ model_runner.run(seqs, True)
  │   │   ├─ prepare_prefill: start=512, seqlen_q=512, seqlen_k=1024
  │   │   │   ├─ input_ids: seq[512:1024], shape [512]
  │   │   │   ├─ positions: [512..1023]
  │   │   │   ├─ cu_seqlens_q = [0, 512], cu_seqlens_k = [0, 1024]  ← Q/K 不对称！
  │   │   │   ├─ slot_mapping: block B2 和 B3 的 slot
  │   │   │   └─ block_tables: [[B0..B7]] ← 触发 paged attention
  │   │   ├─ store_kvcache: 写入 block B2, B3
  │   │   └─ Attention: flash_attn_varlen_func(q, k_cache, v_cache, block_table=...)
  │   │       └─ Q[512 tokens] attend to K[1024 tokens via paged cache]
  │   │
  │   └─ scheduler.postprocess()
  │       ├─ hash_blocks: block 2,3 被哈希
  │       ├─ num_cached_tokens: 512 → 1024
  │       └─ 1024 < 2048 → continue, token 被丢弃
  │
  ├─ Round 3 (Chunk 3)
  │   │  （结构与 Round 2 相同，start=1024, end=1536）
  │   └─ num_cached_tokens: 1024 → 1536, 仍然 < 2048
  │
  ├─ Round 4 (Chunk 4, 最终 chunk)
  │   ├─ scheduler.schedule()
  │   │   ├─ num_tokens = 2048 - 1536 = 512
  │   │   ├─ seq.num_scheduled_tokens = min(512, 512) = 512
  │   │   ├─ 1536 + 512 = 2048 == 2048 → ✅ prefill 完成！
  │   │   ├─ seq.status → RUNNING
  │   │   └─ seq 从 waiting 移到 running
  │   │
  │   ├─ model_runner.run: start=1536, seqlen_q=512, seqlen_k=2048
  │   │   └─ Attention: Q[512] attend to K[2048 via paged cache]
  │   │
  │   └─ scheduler.postprocess()
  │       ├─ num_cached_tokens: 1536 → 2048
  │       ├─ 2048 ≥ 2048 → 不 continue
  │       └─ append_token(token_id) → 序列进入 decode 就绪状态
  │
  └─ Round 5+ (Decode)
      ├─ scheduler.schedule() → waiting 为空，进入 decode 阶段
      ├─ seq.num_scheduled_tokens = 1, seq.is_prefill = False
      └─ prepare_decode: 1 token, CUDA graph 可用
```

### 5.3 状态变化总结表

| Round | 阶段 | num_cached | num_scheduled | seqlen_q | seqlen_k | 队列 | block_tables |
|-------|------|-----------|---------------|----------|----------|------|-------------|
| 1 | Chunk 1 | 0→512 | 512→0 | 512 | 512 | waiting | None |
| 2 | Chunk 2 | 512→1024 | 512→0 | 512 | 1024 | waiting | ✅ |
| 3 | Chunk 3 | 1024→1536 | 512→0 | 512 | 1536 | waiting | ✅ |
| 4 | Chunk 4 | 1536→2048 | 512→0 | 512 | 2048 | →running | ✅ |
| 5 | Decode | 2048→2049 | 1→0 | 1 | 2049 | running | ✅ |

### 5.4 哪些逻辑不在主路径

| 函数/路径 | 为什么容易误解 | 实际是否在主流程 |
|-----------|--------------|----------------|
| `prepare_decode()` | 名字暗示与 decode 有关 | ❌ Chunked Prefill 期间不执行 |
| `capture_cudagraph()` | 初始化时执行 | ❌ 只用于 decode，prefill 始终 eager |
| `preempt()` | 看似与 chunk 状态有关 | ❌ 只在 decode 阶段的 block 不足时触发 |
| `can_append()` / `may_append()` | block 追加逻辑 | ❌ 只用于 decode 产生新 token 时 |
| `Sampler.__call__()` | 每步都执行 | ⚠️ 在未完成的 chunk 中执行了但结果被丢弃 |

### 5.5 💡 小结

- 一个 2048-token 序列在 `max_num_batched_tokens=512` 下需要 4 轮才能完成 prefill。
- chunk 1 走标准 self-attention，chunk 2-4 走 paged attention（读已 cache 的 KV）。
- 序列在 4 轮 prefill 期间一直留在 `waiting` 队列，第 4 轮结束才移到 `running`。
- 每轮 decode 都被阻塞——这 4 轮中没有 decode 能执行。

---

## 六、核心机制深挖

### 6.1 cu_seqlens 的不对称：Chunked Prefill 和 Prefix Cache 的统一抽象

这是 Nano-vLLM 中最优雅的设计之一。不管 token 已经在 cache 中的原因是什么——Prefix Cache 命中、还是前一个 chunk 已处理——在 Attention 层看来，它们的表现是完全相同的：`cu_seqlens_k > cu_seqlens_q`。

这个统一抽象的好处是代码路径极简：
- `model_runner.py:162`：一个 `if` 判断触发 block_table 准备
- `attention.py:65`：一个 `if` 判断切换 K/V 来源

代价是：如果 Prefix Cache 和 Chunked Prefill 同时生效（一个序列既有 prefix 命中又需要 chunk），两者的效果叠加——`start = num_cached_tokens` 已经包含了 prefix 命中的 token 数（由 `allocate()` 设置），chunk 逻辑直接在此基础上继续。不需要任何特殊组合逻辑。

### 6.2 Block 分配策略：全量预分配 vs. 按需分配

Nano-vLLM 选择了**全量预分配**策略（`block_manager.py:75-92`）：当一个序列首次被调度时，为它的**所有** block 一次性分配物理 block，无论这些 block 需要多少个 chunk 才能被填满。

```python
def allocate(self, seq, num_cached_blocks):
    for i in range(num_cached_blocks, seq.num_blocks):  # 分配所有 block
        seq.block_table.append(self._allocate_block())
```

对于一个 2048-token 的序列（8 个 block），即使 chunk 1 只会填满 2 个 block，另外 6 个 block 也已经被分配。

**为什么这样设计？** 主要原因是简化状态管理。如果采用按需分配（每个 chunk 只分配需要的 block），需要在调度器中处理"中间某个 chunk 分配失败"的情况——此时序列已经部分 prefill，既不能继续也不能轻易回退。全量预分配把分配失败的风险集中在第一次调度时，`can_allocate()` 返回 `-1` 就直接跳过这个序列。

**代价是什么？** 显存浪费。一个需要 4 个 chunk 的序列，在 chunk 1 阶段有 6 个 block 被分配但未使用。这些 block 占用了 KV Cache 容量，可能导致其他 decode 序列因为 block 不足而被 preempt。

### 6.3 Preemption 与 Chunked Prefill 的交互

一个自然的问题是：部分 prefill 的序列会被 preempt 吗？

答案是**不会直接被 preempt**。原因是 preemption 只发生在 decode 阶段（`scheduler.py:60-65`），而部分 prefill 的序列留在 `waiting` 队列中，不在 `running` 队列中。decode 阶段只从 `running` 队列选择序列，所以部分 prefill 的序列不会被触及。

但有一个间接影响：一个序列完成 prefill 后移到 `running`，如果此时另一个长序列正在全量预分配 block 导致 cache 紧张，这个刚完成 prefill 的序列可能在下一轮 decode 中被 preempt。Preemption 会清空它的所有 block（`deallocate()`），重置 `num_cached_tokens=0`，将其放回 `waiting` 队首。下次调度时它需要重新 prefill——但此时 `hash_blocks()` 之前已经为它的 block 注册了 hash，如果这些 block 还没被覆盖，`can_allocate()` 可能找到 prefix cache 命中，从而跳过部分重计算。

### 6.4 💡 小结

- Q/K 不对称的 cu_seqlens 是 Chunked Prefill 和 Prefix Cache 的统一抽象，极其简洁。
- block 全量预分配简化了状态管理，但增加了显存浪费。
- 部分 prefill 的序列不会被 preempt（不在 running 队列），但完成 prefill 后可能被 preempt。

---

## 七、显存、性能与通信分析

### 7.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|------|---------|------|
| 模型参数 | ❌ | 与 chunk 无关 |
| KV Cache 总量 | ❌ | 全量预分配，不减少 |
| 激活值 | ✅ | chunk 大小决定前向峰值，从 O(L) 降到 O(B) |
| Logits | ✅ | 每步只产生 chunk 大小的 logits |
| Optimizer state | N/A | 纯推理，不涉及 |
| 输入 batch | ✅ | 每步只有 chunk 大小的 input_ids |
| 未用 block | ❌ 额外开销 | 全量预分配的未用 block 占用 cache 容量 |

**真正的显存大头**是激活值。在 Transformer 的前向计算中，每层的中间张量（Q/K/V 投影、Attention 输出、MLP 中间态）都与 token 数成正比。对于一个 32768-token 的序列和 hidden_size=1024 的模型：

- 非 chunk：每层主要激活约 `32768 × 1024 × 4` 字节（Q 投影）= ~128 MB / 层
- chunk（B=4096）：每层约 `4096 × 1024 × 4` 字节 = ~16 MB / 层

8 倍的激活显存减少。

但注意：KV Cache 的"省"是一个幻觉。全量预分配意味着不管你分几个 chunk，block 在第一个 chunk 就已经全部分配了。Chunked Prefill 不减少 KV Cache 用量。

### 7.2 通信开销

在 Tensor Parallelism 场景下，每步的通信量取决于参与计算的 token 数：

| 通信点 | 每步发生次数 | 类型 | 受 chunk 影响 |
|--------|------------|------|-------------|
| `RowParallelLinear.forward()` | `2 × num_layers` | all_reduce | ✅ 每步 token 少则通信量少 |
| `ParallelLMHead.forward()` | 1 | gather | ✅ 但只涉及 last-token |
| `VocabParallelEmbedding` | 1 | all_reduce | ✅ token 数决定通信量 |
| 序列序列化（SharedMemory） | 1 | pickle via shm | ⚠️ 传输全量 token_ids |

注意最后一项：在 TP 模式下，`Sequence.__getstate__()` 对于 prefill 序列传输完整的 `token_ids`（`sequence.py:73`）：

```python
last_state = self.last_token if not self.is_prefill else self.token_ids
```

这意味着一个 32K-token 的序列，每个 chunk 步都通过 1MB 的 SharedMemory buffer 传输整个 32K 的 token_ids 列表。如果序列更长，可能会超过 buffer 限制（`model_runner.py:43`：`size=2**20`，即 1MB）。

### 7.3 性能取舍

Chunked Prefill 在 Nano-vLLM 中的核心取舍可以概括为：

**获得**：
- 激活显存峰值可控（从 O(L) 降到 O(B)）
- warmup 阶段峰值更低 → 更多 KV Cache block → 更高 decode 并发

**牺牲**：
- 每个 chunk 都有一次完整的调度 + 上下文准备 + kernel launch 开销
- 不完整 chunk 的 LM Head / sampler 计算被浪费
- chunk 2+ 的 attention 计算量随上下文增长：chunk k 的 Q 只有 B 个 token，但 K 有 k×B 个 token
- decode 延迟增加：4 个 chunk 意味着 4 步 decode 被阻塞

**没有牺牲**：
- 总 attention FLOPs 不变。对于长度 L 的序列，无论是否 chunk，总 attention FLOPs 为 `L×(L+1)/2`。chunk 只是把同样的计算分到了多步中。

### 7.4 💡 小结

- 主要显存收益来自激活值的峰值控制，KV Cache 用量不减少。
- TP 模式下每个 chunk 步都序列化完整 token_ids，长序列可能超出 1MB buffer。
- 总 FLOPs 不变，开销来自调度和无用的 sampler 计算。

---

## 八、配置项、边界条件与坑点

### 8.1 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险/坑点 |
|--------|------------|---------|----------|
| `max_num_batched_tokens=16384` (默认) | `scheduler.py:32` | 预算 16384，`max_model_len=4096` 时永远不触发 chunk | 用户以为启用了 Chunked Prefill，实际从未触发 |
| `max_num_batched_tokens=512` | `scheduler.py:46` | 每步最多 512 token，长序列必然被 chunk | warmup 峰值低，KV Cache block 多；但 prefill 需要很多步 |
| `max_model_len=32768` | `config.py:25` 取 min | 序列可达 32K，大概率触发 chunk | 全量预分配 128 个 block，可能挤占其他序列 |
| `enforce_eager=True` | `model_runner.py:197` | 所有推理 eager mode | 对 chunk 无直接影响（prefill 本来就是 eager） |
| `kvcache_block_size=256` (默认) | `block_manager.py:111-112` | chunk 边界可能不对齐 block 边界 | 跨 block 的 chunk 导致 hash 延迟 |

### 8.2 边界条件

**边界 1：chunk 大小等于序列长度**
- 效果：与不 chunk 完全相同。`min(num_tokens, remaining) = num_tokens`，`num_cached + num_scheduled == num_tokens`，序列直接移到 running。

**边界 2：chunk 大小为 0**
- 被 `remaining == 0` 守卫拦住（`scheduler.py:33`），不会调度任何序列。

**边界 3：极短序列（token 数 < block_size）**
- 只分配 1 个 block，hash 永远不触发（`hash_blocks` 的 `end = (tokens) // 256 = 0 = start`），prefix cache 无效。

**边界 4：preemption 后重新 prefill**
- `deallocate()` 清空 block_table → 重新进入 `not seq.block_table` 分支 → 重新 `can_allocate()` → 可能命中之前 `hash_blocks` 注册的 hash → 部分 token 免重算。

### 8.3 💡 小结

- 默认配置下 Chunked Prefill 不会触发。
- 全量预分配在长序列场景下是双刃剑：避免中途分配失败，但挤占 cache 容量。
- 极短序列无法享受 Prefix Cache——block 粒度太大。

---

## 九、Bug 演进史：4 次修复的教训

Chunked Prefill 从初版实现到当前版本，经历了 4 次 bug 修复。这些 bug 的分布揭示了这个特性的复杂性所在。

### 9.1 初版实现（commit `8d63a98`）

**作者**：GeekExplorer (xingkai@deepseek.com)  
**日期**：2026-04-14  
**内容**：引入 `num_scheduled_tokens`，修改 scheduler 和 model_runner 支持部分 prefill。

初版的 scheduler 在 `postprocess` 中将 prefill 和 decode 的 `num_cached_tokens` 更新逻辑分开处理，较为复杂。

### 9.2 Bug #1：IndexError（commit `77dd709`）

**问题**：`num_tokens` 在 `allocate()` 之前计算。但 `allocate()` 会因为 Prefix Cache 更新 `num_cached_tokens`。结果 `num_scheduled_tokens` 过大，`prepare_prefill` 中 `end_block` 越界。

**修复**：将 `num_tokens` 的计算移到 `allocate()` 之后。

**教训**：`can_allocate()` 和 `allocate()` 之间有状态耦合——`allocate` 会修改 `num_cached_tokens`。

### 9.3 Bug #2：seqlen_k 错误（commit `25794a1`）

**问题**：`seqlen_k` 被设为 `len(seq)`（完整序列长度），而非 `end`（当前 chunk 的结束位置）。结果 attention 读取了尚未写入的 KV Cache slot——这些 slot 包含垃圾数据。

**修复**：`seqlen_k = end`，只 attend 到已有 KV 的位置。

**教训**：Chunked Prefill 的 K 长度不是"序列有多长"，而是"已经处理了多少"。

### 9.4 Bug #3：cache hit 逻辑（commit `9fa256a`）

**问题**：`_allocate_block` 使用 `self.free_block_ids.remove(block_id)` 删除特定 block，这是 O(n) 操作且语义不对。

**修复**：改为 `popleft()` 从队首取 block。

### 9.5 Bug #4：大重构（commit `f64d821`）

这次重构一次性解决了多个问题：

1. **分离 `can_allocate` 和 `allocate`**：`can_allocate` 只返回 cached block 数量（或 -1），`allocate` 接受 `num_cached_blocks` 参数。消除了之前的状态耦合。
2. **延迟分配**：`allocate` 移到预算检查之后（`scheduler.py:44`），避免为不会被调度的序列分配 block。
3. **引入 `is_prefill` 标志**：在 Sequence 上添加 `is_prefill` 字段，用于序列化时决定传输全量 token_ids 还是只传 last_token。
4. **引入 `hash_blocks`**：将 block 哈希从 allocate 时移到 postprocess 时，支持增量哈希。
5. **简化 postprocess**：统一 prefill 和 decode 的 `num_cached_tokens` 更新逻辑。

### 9.6 教训总结

4 次修复中：
- 2 次涉及 **scheduler 和 block_manager 之间的状态一致性**（Bug #1, #4）
- 1 次涉及 **Q/K 长度语义**（Bug #2）
- 1 次涉及 **block 分配实现**（Bug #3）

Chunked Prefill 的复杂性不在算法本身，而在**跨组件的状态协调**。没有测试套件使得每个 bug 都是在实际运行中发现的。

### 9.7 💡 小结

- 4 次修复暴露了 Chunked Prefill 的核心难点：多组件间的状态一致性。
- 重构（`f64d821`）通过分离 `can_allocate`/`allocate` 和延迟分配显著改善了代码质量。
- 缺少测试套件是这些 bug 反复出现的根本原因。

---

## 十、测试、示例与覆盖缺口

### 10.1 已覆盖路径

| 测试/示例 | 覆盖的行为 | 说明 |
|-----------|----------|------|
| `bench.py` | 256 序列，input 100-1024 tok | `max_num_batched_tokens=16384` → chunk 不会触发 |
| `example.py` | 短 prompt，enforce_eager | 不触发 chunk |

### 10.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|--------|-------------|---------|
| 长序列触发 chunk（>16K tok） | ❌ | 无法验证多 chunk 的 Q/K 不对称是否正确 |
| Prefix Cache + Chunked Prefill 同时生效 | ❌ | 两者叠加的 num_cached_tokens 语义可能有 edge case |
| Preemption 后重新 chunk | ❌ | deallocate/reallocate/re-chunk 的状态转换 |
| TP 模式下长序列的 SharedMemory 溢出 | ❌ | >1MB 的 pickle 数据导致 buffer 溢出 |
| chunk 边界不对齐 block 边界时的 hash | ❌ | prefix cache 命中率可能低于预期 |
| decode 阶段 hash_blocks 的 timing bug | ❌ | prefix cache 命中率降低 |
| chunk 大小 < block_size | ❌ | 一个 block 跨多个 chunk 的 hash 行为 |

### 10.3 💡 小结

- 现有测试和示例均未覆盖 Chunked Prefill 的实际触发路径。
- 最关键的缺口是多 chunk + prefix cache + preemption 的交叉测试。

---

## 十一、局限性与已知优化点

### 11.1 硬约束

1. **只支持单序列 chunk**：每步最多一个序列被 chunk（`scheduler.py:42`），其余序列必须完整塞入预算。
2. **prefill 和 decode 互斥**：一步只做 prefill 或 decode，不混合（`scheduler.py:54-55`）。
3. **全量 block 预分配**：即使 chunk 1 只用 2 个 block，也要分配全部 8 个（`block_manager.py:90-91`）。
4. **SharedMemory 1MB 限制**：长序列 + TP 模式可能溢出（`model_runner.py:43`）。
5. **block_size 必须是 256 的倍数**：大 block 导致极短序列无法 prefix cache（`config.py:22`）。

### 11.2 维护成本

1. **`num_cached_tokens` 语义过载**：同时代表 prefix cache 命中数和 chunk 进度，在 `allocate`/`postprocess`/`prepare_prefill` 三个地方被读写。
2. **全局可变 Context**：`set_context` / `reset_context` 没有 `try/finally` 保护，异常可能留下脏状态。
3. **postprocess 中的 hash timing**：`hash_blocks` 在 `append_token` 之前调用，decode 阶段可能 hash 错误内容。

### 11.3 性能瓶颈

1. **不完整 chunk 的 LM Head / Sampler 浪费**：每个非最终 chunk 都白做了一次 vocab projection（模型最大的线性层）。
2. **TP 模式每步序列化全量 token_ids**：每个 chunk 步都通过 SharedMemory pickle 传输完整 prompt。
3. **单序列 chunk 限制导致预算浪费**：如果第一个序列只需 100 token，第二个序列需要 50000 token，后者不能用剩余的 16284 预算做 chunk——必须等到它排到队首。

### 11.4 已知优化点

1. **混合 prefill-decode batching**（参考 vLLM）：在同一步中同时处理 chunk prefill 和 decode token，消除 decode 阻塞。这需要在 `run_model` 中分别处理 prefill 和 decode 子 batch，是高价值但高复杂度的改动。

2. **按需 block 分配**：每个 chunk 只分配当前需要的 block，减少 cache 占用。需要处理中途分配失败的情况。

3. **跳过不完整 chunk 的 LM Head**：在 `run()` 中检查是否有未完成的 chunk，跳过 `compute_logits` 和 sampler。需要让 model_runner 感知 chunk 完成状态。

4. **只传 chunk token_ids**：`__getstate__` 中对 prefill 序列只传 `token_ids[start:end]`，而非全量 `token_ids`。需要 worker 端相应调整 `prepare_prefill` 的索引逻辑。

5. **多序列 chunk**：允许多个序列在同一步被 chunk。Flash Attention 的 varlen 接口天然支持每个序列不同的 Q/K 长度。主要挑战在调度器的状态管理。

---

## 小结与展望

Nano-vLLM 的 Chunked Prefill 实现可以用几个关键词概括。

**关键词一：隐式触发**

没有显式开关，Chunked Prefill 作为调度器的内在行为存在——`max_num_batched_tokens` 既是 token 预算也是 chunk 粒度。这种设计避免了配置爆炸，但也让用户难以意识到这个特性何时生效。

**关键词二：cu_seqlens 不对称**

`cu_seqlens_q ≠ cu_seqlens_k` 是 Chunked Prefill 传递给 Flash Attention 的核心信号，且与 Prefix Cache 完全复用同一条件和同一代码路径。这是本实现中最优雅的设计——一个抽象同时覆盖两个特性。

**关键词三：全量预分配**

block 在首次调度时全量分配，简化了中途失败的处理，但以 cache 占用为代价。这是简洁性与效率的典型工程权衡。

**关键词四：prefill-decode 互斥**

与 vLLM 不同，Nano-vLLM 的 Chunked Prefill 不降低 decode 延迟——它只控制激活显存峰值。这是本实现最大的局限，也是最有价值的优化方向。

**适用场景**：
- 长 prompt（>16K）的推理，需要控制激活显存防止 OOM
- 中小模型在单 GPU 上的推理，KV Cache 容量有限
- 对 decode 延迟不敏感的离线批处理场景

**不适用场景**：
- 需要低延迟流式 decode 的在线服务（decode 会被多 chunk 阻塞）
- 需要在同一 batch 中混合 prefill 和 decode 以最大化 GPU 利用率的高吞吐场景
- TP 模式下处理极长序列（>100K，SharedMemory buffer 限制）

**后续值得继续走读的方向**：
- vLLM 中 Chunked Prefill + Decode 混合调度的实现（Sarathi-Serve 风格）
- 按需 block 分配如何与 prefix cache 协同工作
- CUDA Graph 如何适应动态 batch 组成（decode + prefill 混合）
