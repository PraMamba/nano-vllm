# nano-vllm 源码走读：抢占机制实现解析

> 在前几篇文章中，我们分析了 nano-vllm 的 Paged Attention、Block Manager、Prefix Cache、KV Cache 容量规划、Continuous Batching 以及 Chunked Prefill。这些机制共同构建了一个高效的推理引擎——但有一个关键问题始终悬而未决：**当 KV Cache 被完全占满、新的 decode step 无法分配 block 时，系统该怎么办？** 这正是抢占机制（Preemption）要回答的问题。本文不展开抢占的理论背景（如 vLLM 论文中 recompute vs. swap 的讨论），而是聚焦源码，分析 nano-vllm 如何在 92 行调度器代码中实现一套完整的抢占-恢复链路，以及它带来了哪些收益和局限。

---

## 前言

### 业务 / 工程背景

在 LLM 推理的 continuous batching 场景下，多条 sequence 并发运行，共享有限的 GPU 显存。每条 sequence 在 decode 阶段每一步只生成一个 token，但可能需要分配新的 KV cache block（当 token 数量跨越 block 边界时）。当并发 sequence 数量足够多、生成长度足够长时，KV cache 的 block 池会被耗尽。

此时系统面临一个选择：要么拒绝服务（OOM 或停止调度），要么主动"牺牲"某些 sequence 来为其他 sequence 腾出空间。

### 核心矛盾

**抢占机制的本质是一个时空权衡：用重复计算换取显存空间。** 被抢占的 sequence 丢失所有 KV cache（显存释放），必须在未来重新 prefill（计算代价），但系统整体避免了因显存耗尽而停滞。

这个矛盾引出了几个关键工程问题：

1. **谁被牺牲？** victim 选择策略决定了浪费的计算量和公平性。
2. **牺牲后还能恢复多少？** Prefix cache 的存在使得被抢占的 sequence 有机会跳过部分重复计算——但这取决于 freed block 是否在 re-prefill 之前被其他 sequence 消费。
3. **如何避免活锁？** 如果两条 sequence 反复互相抢占，系统可能永远无法推进。

### 本文主线

本文按以下主线展开：

- **第一章**：Sequence 的状态机与抢占引发的状态回退
- **第二章**：decode 调度循环——抢占的触发与 victim 选择
- **第三章**：BlockManager 的释放与回收——抢占如何与 prefix cache 交互
- **第四章**：重新调度——被抢占的 sequence 如何回到主流程
- **第五章**：完整主路径串联
- **第六章**：关键机制深挖——FIFO free list、自我抢占、recompute vs. swap
- **第七章**：显存、性能与通信分析
- **第八章**：配置项、边界条件与坑点
- **第九章**：局限性与已知优化点

### 不展开的内容

本文不讲 vLLM 完整版的 swap-to-CPU 抢占策略，不讲 PagedAttention 原理（已在 08 篇覆盖），不讲 prefix cache 的哈希算法细节（已在 10 篇覆盖）。只聚焦 nano-vllm 中抢占机制的源码实现。

### 核心文件表

| 文件 | 职责 |
|---|---|
| `nanovllm/engine/scheduler.py` | 调度核心：两阶段调度、抢占触发、victim 选择、postprocess |
| `nanovllm/engine/block_manager.py` | Block 生命周期：分配、释放、prefix cache hash、FIFO free list |
| `nanovllm/engine/sequence.py` | Sequence 状态机：WAITING/RUNNING/FINISHED、序列化 |
| `nanovllm/engine/llm_engine.py` | 引擎主循环：schedule → run → postprocess |
| `nanovllm/engine/model_runner.py` | 模型执行：prepare_prefill/decode、CUDA graph |
| `nanovllm/utils/context.py` | 全局上下文：is_prefill、slot_mapping、block_tables |

---

## 一、Sequence 状态机：抢占是一种可逆的状态回退

### 1.1 设计哲学与核心问题

在大多数系统中，状态机的转换是单向的——任务从"等待"到"运行"再到"完成"。但在显存受限的 LLM 推理中，**一条 sequence 可能在运行到一半时被强制退回到等待状态**。这不是异常处理，而是正常调度策略的一部分。

如果没有这种"状态回退"能力，系统面对显存不足时只有两个选择：等待其他 sequence 自然结束（可能很慢），或者直接 OOM 崩溃。抢占提供了第三条路——主动释放资源，代价是未来需要重新计算。

### 1.2 源码入口与关键对象

```
nanovllm/engine/sequence.py
  - SequenceStatus 枚举：WAITING / RUNNING / FINISHED（第 8-11 行）
  - Sequence 类：管理 token_ids、block_table、状态标志（第 14-83 行）

nanovllm/engine/scheduler.py
  - Scheduler.preempt()：执行状态回退（第 75-79 行）
```

### 1.3 状态转换全景

```
                    T1: prefill 完成
         ┌──────────────────────────────┐
         │                              │
         ▼                              │
    +-----------+   T3: 抢占      +-----------+
    |  WAITING  |◄────────────────|  RUNNING  |
    |           |                 |           |
    +-----------+                 +-----------+
         │                              │
         │  T4: chunked prefill         │ T2: EOS / max_tokens
         └──(自循环，状态不变)           │
                                        ▼
                                  +-----------+
                                  | FINISHED  |
                                  +-----------+
```

| 转换 | 触发条件 | 状态变更 | 所在模块 | 源码位置 |
|---|---|---|---|---|
| T1: WAITING → RUNNING | prefill 调度完成，`num_cached_tokens + num_scheduled_tokens == num_tokens` | `status=RUNNING`，从 `waiting` 移到 `running` 队列 | `Scheduler.schedule()` | `scheduler.py:48-51` |
| T2: RUNNING → FINISHED | 命中 EOS 或达到 `max_tokens` | `status=FINISHED`，释放 block，从 `running` 移除 | `Scheduler.postprocess()` | `scheduler.py:89-92` |
| T3: RUNNING → WAITING | decode 阶段 `can_append` 失败，被选为 victim | `status=WAITING`，`is_prefill=True`，block 全部释放，插入 `waiting` 队首 | `Scheduler.preempt()` | `scheduler.py:75-79` |
| T4: WAITING → WAITING | chunked prefill 未完成 | `num_cached_tokens` 递增，但状态不变 | `Scheduler.postprocess()` | `scheduler.py:86-87` |

### 1.4 关键细节与误区澄清

**误区一：`preempt()` 是否需要重置 `num_scheduled_tokens`？**

`preempt()` 方法（`scheduler.py:75-79`）重置了 `status`、`is_prefill`，并通过 `deallocate()` 清除了 `block_table` 和 `num_cached_tokens`，但**没有显式重置 `num_scheduled_tokens`**：

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

这看起来像一个 bug，但实际上**当前不会引发问题**。原因是：被抢占的 sequence 在上一轮 `postprocess()` 中已经将 `num_scheduled_tokens` 置零（`scheduler.py:85`），而在当前 decode 轮次中还没来得及被赋新值。当 sequence 重新进入 prefill 调度时，`num_scheduled_tokens` 会被无条件覆写（`scheduler.py:46`）。但这属于**脆弱的正确性**——如果未来有代码在 preempt 和 re-schedule 之间读取该字段，就会拿到过时的值。

**误区二：抢占后 sequence 的 `token_ids` 是否保留？**

是的，完全保留。`preempt()` 不会修改 `token_ids`。一条已经生成了 200 个 token 的 sequence 被抢占后，它的 `token_ids` 仍然包含原始 prompt + 200 个已生成的 token。这些 token 不会丢失，只是对应的 KV cache 被释放了，需要在 re-prefill 时重新计算。

### 1.5 本章小结

> 💡 **小结**
> - Sequence 状态机支持 RUNNING → WAITING 的可逆回退，这是抢占机制的基础。
> - `preempt()` 方法重置状态标志并释放所有 block，但保留 `token_ids`（包括已生成的 token）。
> - `num_scheduled_tokens` 未被显式重置，虽然当前安全，但属于脆弱设计。
> - 被抢占的 sequence 必须从头 re-prefill，但 prefix cache 可能帮助跳过部分计算。

---

## 二、Decode 调度循环：抢占的触发与 Victim 选择

### 2.1 设计哲学与核心问题

抢占发生在 decode 阶段，而非 prefill 阶段。这是因为 **prefill 阶段通过 `can_allocate` 的返回值控制准入**——如果 block 不够，新 sequence 根本不会被调度，也就不需要抢占。而在 decode 阶段，所有 running sequence 已经分配了 block，但每次 decode 可能需要分配新 block（当 token 数量跨越 block 边界时）。如果此时没有 free block，就必须从其他 running sequence 那里"抢"。

核心问题是：**从谁那里抢？抢了之后怎么办？**

### 2.2 源码入口与关键对象

```
nanovllm/engine/scheduler.py
  - Scheduler.schedule()，decode 部分：第 57-73 行
  - Scheduler.preempt()：第 75-79 行

nanovllm/engine/block_manager.py
  - BlockManager.can_append()：第 103-104 行
  - BlockManager.may_append()：第 106-108 行
```

### 2.3 主流程拆解

decode 调度循环是整个抢占机制中最精妙（也最微妙）的代码段。让我们逐行拆解：

```python
# scheduler.py:57-73
# decode
while self.running and len(scheduled_seqs) < self.max_num_seqs:     # [A]
    seq = self.running.popleft()                                      # [B]
    while not self.block_manager.can_append(seq):                     # [C]
        if self.running:                                              # [D]
            self.preempt(self.running.pop())                          # [E]
        else:                                                         # [F]
            self.preempt(seq)                                         # [G]
            break                                                     # [H]
    else:                                                             # [I]
        seq.num_scheduled_tokens = 1                                  # [J]
        seq.is_prefill = False                                        # [K]
        self.block_manager.may_append(seq)                            # [L]
        scheduled_seqs.append(seq)                                    # [M]
assert scheduled_seqs                                                 # [N]
self.running.extendleft(reversed(scheduled_seqs))                     # [O]
```

**三条代码路径：**

**路径 1：正常 decode（最常见）**

`can_append` 返回 True → 内层 `while` 不执行 → `else` 块执行（Python 的 `while...else` 语义：循环正常结束时执行 `else`）→ 设置 `num_scheduled_tokens=1`，分配新 block（如果需要），加入 `scheduled_seqs`。

**路径 2：抢占其他 sequence**

`can_append` 返回 False → 进入内层 `while` → `self.running` 非空 → 从 `running` **右端**弹出一条 sequence 作为 victim（`self.running.pop()`，LIFO 策略）→ 调用 `preempt(victim)` 释放其 block → 回到内层 `while` 重新检查 `can_append` → 如果仍然不够，继续抢占下一个 victim → 直到 `can_append` 成功。

**路径 3：自我抢占（边界情况）**

`can_append` 返回 False → `self.running` 已空（所有其他 sequence 都已被抢占或已调度）→ 当前 seq 抢占自己 → `break` 退出内层 `while` → `else` 不执行 → seq 不加入 `scheduled_seqs` → 外层 `while` 检查 `self.running` 为空，退出 → **`assert scheduled_seqs` 可能触发崩溃**。

### 2.4 Victim 选择策略：LIFO 的得与失

`self.running.pop()` 弹出的是 `running` 队列的**右端**（最后加入的 sequence）。由于 sequence 通过 `self.running.append(seq)`（`scheduler.py:51`）从右端加入，LIFO 策略实际上是**驱逐最晚开始 decode 的 sequence**。

这意味着已经 decode 了较多 token 的 sequence（在队列前部）受到保护，而新近进入 decode 阶段的 sequence 更容易被牺牲。

```
running 队列：[seq_A(老) ... seq_D(新)]
                 ↑                ↑
             popleft()          pop() ← victim 从这里选
             (当前调度)        (被抢占)
```

**LIFO 的优点：**
- 被抢占的 sequence 通常生成的 token 较少，re-prefill 的额外计算量相对较小
- 保护了已经投入大量计算的 sequence，避免功亏一篑

**LIFO 的缺点：**
- **victim 锁定效应**：被抢占的 sequence 通过 `appendleft` 进入 `waiting` 队首，在下一轮 prefill 后重新进入 `running` 的右端——又变成了下一次抢占的首选 victim。这可能导致同一条 sequence 反复被抢占，而其他 sequence 一路畅通完成。
- 没有考虑"释放 block 数量"——一条占用 2 个 block 的短 sequence 和一条占用 10 个 block 的长 sequence 被同等对待，但驱逐后者能释放更多空间

### 2.5 `can_append` 的布尔运算技巧

```python
# block_manager.py:103-104
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

这行代码利用了 Python 中 `bool` 是 `int` 子类的特性：`True == 1`，`False == 0`。

其语义是：
- 当 `len(seq) % block_size == 1` 时（sequence 长度刚好跨入新 block 的第一个 token），需要 1 个 free block → 检查 `free_block_ids >= 1`
- 否则不需要新 block → 检查 `free_block_ids >= 0`，恒为真

为什么判定条件是 `% block_size == 1` 而不是 `== 0`？因为在 decode 调度时，`len(seq)` 反映的是当前 sequence 的总 token 数（包括上一步刚生成的 token）。当 `len(seq) % 256 == 1` 时，意味着上一步的 token 恰好填满了前一个 block，当前 token 需要写入新 block。

### 2.6 关键细节与误区澄清

**误区三：`while...else` 的语义**

Python 的 `while...else` 是一个容易误解的语法结构。`else` 块只在 `while` 条件变为 `False`（正常退出）时执行，如果通过 `break` 退出则**不执行**。这意味着：

- 正常 decode（`can_append` 成功）：`else` 执行，sequence 被调度
- 自我抢占（`break`退出）：`else` 不执行，sequence **不被调度**

这是整个抢占机制正确性的关键控制流。

**误区四：已调度的 sequence 是否会被抢占？**

不会。内层循环中抢占的 victim 来自 `self.running.pop()`——即尚未被当前轮次处理过的 sequence。已经加入 `scheduled_seqs` 的 sequence 已从 `running` 中移除（通过 `popleft`），不在 victim 候选池中。这保证了一旦 sequence 在当前轮次被调度，其 block 分配是安全的。

### 2.7 本章小结

> 💡 **小结**
> - 抢占只发生在 decode 阶段，由 `can_append` 返回 False 触发。
> - Victim 选择使用 LIFO 策略（`running.pop()`），优先驱逐最晚加入 decode 的 sequence。
> - `while...else` 语法保证了被自我抢占的 sequence 不会被调度。
> - LIFO 策略保护了投入最多计算的 sequence，但可能导致同一条 sequence 反复被牺牲。

---

## 三、BlockManager 的释放与回收：抢占如何与 Prefix Cache 交互

### 3.1 设计哲学与核心问题

当一条 sequence 被抢占时，它的所有 KV cache block 被释放。但"释放"在 nano-vllm 中是一个**带有缓存语义的操作**：block 的物理空间被释放回 free list，但 block 上的**哈希元数据被保留**。这意味着被释放的 block 仍然可以作为 prefix cache 的候选，在被抢占的 sequence 重新 prefill 时被"认领"回来，跳过对应 token 的重复计算。

这是 nano-vllm 抢占机制最精巧的设计之一：**释放不等于遗忘**。

### 3.2 源码入口与关键对象

```
nanovllm/engine/block_manager.py
  - BlockManager.deallocate()：释放 sequence 的所有 block（第 94-101 行）
  - BlockManager._deallocate_block()：单个 block 的释放（第 53-56 行）
  - BlockManager._allocate_block()：从 free list 分配 block（第 43-51 行）
  - BlockManager.can_allocate()：检查是否能分配并计算 prefix cache 命中（第 58-73 行）
```

### 3.3 主流程拆解：释放 Block 的完整路径

```python
# block_manager.py:94-101
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):     # [1] 逆序遍历
        block = self.blocks[block_id]
        block.ref_count -= 1                        # [2] 引用计数递减
        if block.ref_count == 0:
            self._deallocate_block(block_id)        # [3] 引用归零才真正释放
    seq.num_cached_tokens = 0                       # [4] 重置缓存 token 计数
    seq.block_table.clear()                         # [5] 清空 block 表
```

关键观察：

**[1] 逆序释放**：`reversed(seq.block_table)` 意味着最后一个 block（sequence 尾部）先被释放，第一个 block（prompt 前缀）最后被释放。由于 `_deallocate_block` 将 block 追加到 `free_block_ids` 的**尾部**（`append`），而 `_allocate_block` 从**头部**取（`popleft`），这意味着先释放的尾部 block 在 FIFO 队列中排在前面，更早被回收消费；后释放的前缀 block 排在后面，存活时间更长。

**这对 prefix cache 的意义是深远的**：prompt 前缀 block 是最有可能被其他 sequence 共享的（比如相同 system prompt），也是 re-prefill 时最有价值的缓存目标。逆序释放 + FIFO 回收确保了前缀 block 的存活概率最高。

```
释放顺序：block_N → block_N-1 → ... → block_1 → block_0
FIFO free list: [block_N, block_N-1, ..., block_1, block_0]
                 ↑                                        ↑
            先被回收消费                             最后被回收消费
            (sequence 尾部)                         (prompt 前缀)
```

**[2]-[3] 引用计数**：如果一个 block 被多条 sequence 共享（通过 prefix cache），`ref_count > 1`，此时 `deallocate` 只递减计数而不真正释放。只有最后一个引用者释放时，block 才回到 free list。

**[4]-[5] 清理 Sequence 状态**：`num_cached_tokens = 0` 和 `block_table.clear()` 将 sequence 恢复到"从未分配过 block"的状态，为后续 re-prefill 做准备。

### 3.4 哈希元数据的保留机制

`_deallocate_block` 的实现只做两件事：

```python
# block_manager.py:53-56
def _deallocate_block(self, block_id: int):
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)
```

注意它**没有**清除 `block.hash`、`block.token_ids` 或 `hash_to_block_id` 映射。block 上的哈希指纹和 token 内容完整保留。这些元数据只有在 block 被**重新分配**时才被清除：

```python
# block_manager.py:43-51
def _allocate_block(self) -> int:
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    assert block.ref_count == 0
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]    # 此时才清除哈希映射
    block.reset()                                 # 此时才清除 block 元数据
    self.used_block_ids.add(block_id)
    return block_id
```

这构成了一个**惰性缓存失效（lazy invalidation）** 策略：block 在 free list 中时仍然"记得"自己存储的内容，直到被分配给新用户时才真正"失忆"。

### 3.5 关键细节与误区澄清

**误区五：被抢占的 sequence 重新 prefill 时，prefix cache 命中率一定很高吗？**

不一定。虽然哈希元数据被保留，但 prefix cache 命中依赖于一个关键前提：**被释放的 block 在 re-prefill 之前没有被 `_allocate_block` 回收**。

而抢占恰恰发生在显存紧张时——free list 为空或接近空。被抢占 sequence 释放的 block 是 free list 中仅有的 block。decode 调度循环在抢占后继续运行，其他 running sequence 的 `may_append` 调用可能立即消费这些刚释放的 block。

不过，`may_append` 每条 sequence 每次 decode step 最多消费 1 个 block（且只有在跨越 block 边界时），而跨越概率仅为 1/256。所以在单个调度轮次中，实际被消费的 block 数量通常很少（期望值 ≈ `N_running / 256`）。

**结论：prefix cache 在短期恢复（下一个调度轮次 re-prefill）中通常有效，但在持续高压下效果衰减。**

### 3.6 本章小结

> 💡 **小结**
> - `deallocate()` 释放 block 但保留哈希元数据，构成惰性缓存失效策略。
> - 逆序释放 + FIFO free list 确保 prompt 前缀 block 存活时间最长。
> - 只有 `_allocate_block` 才真正清除 block 的哈希指纹，这为 re-prefill 时的 prefix cache 命中创造了窗口。
> - 在显存极度紧张时，freed block 可能被立即消费，prefix cache 效果受限。

---

## 四、重新调度：被抢占的 Sequence 如何回到主流程

### 4.1 设计哲学与核心问题

被抢占的 sequence 面临一个"重生"过程：它从 RUNNING 退回到 WAITING，所有 KV cache 被清除，然后需要重新 prefill。但它不是一个全新的 sequence——它携带了原始 prompt **加上**已经生成的 token。问题是：

1. 它在 waiting 队列中的优先级如何？
2. 它能利用多少 prefix cache？
3. 它的 re-prefill 开销是多少？

### 4.2 源码入口与关键对象

```
nanovllm/engine/scheduler.py
  - Scheduler.preempt()：appendleft 到 waiting 队首（第 79 行）
  - Scheduler.schedule()，prefill 部分：第 29-55 行

nanovllm/engine/block_manager.py
  - BlockManager.can_allocate()：第 58-73 行
  - BlockManager.allocate()：第 75-92 行
```

### 4.3 主流程拆解：从抢占到重生

**Step 1：优先插入 waiting 队首**

```python
# scheduler.py:79
self.waiting.appendleft(seq)    # 被抢占的 sequence 插入队首
```

对比新 sequence 的插入：

```python
# scheduler.py:23
self.waiting.append(seq)         # 新 sequence 插入队尾
```

这构成了一个**两级优先级队列**：

```
waiting 队列：
  [被抢占的 seq_C, 被抢占的 seq_B | 新 seq_X, 新 seq_Y, 新 seq_Z]
   ↑                                ↑
   优先调度（appendleft）          后调度（append）
```

被抢占的 sequence 总是优先于新 sequence 被重新调度。这是合理的——它们已经投入了计算资源，应该尽快恢复。

**当多条 sequence 在同一轮被连续抢占时**，抢占顺序是 LIFO（`running.pop()`），但每条被抢占的 sequence 都 `appendleft` 到 waiting 队首。效果是：后被抢占的 sequence 在 waiting 队列中更靠前。但由于 LIFO 先抢占队尾（最新的），后抢占队首（较老的），最终 waiting 中的顺序恢复为与原始 running 顺序一致：

```
running: [A, B, C, D]  (A 最老，D 最新)
抢占顺序：D → C → B
waiting 结果：[B, C, D, ...]  (B 最先被重新调度)
```

**Step 2：重新进入 prefill 调度**

在下一个 `schedule()` 调用中，被抢占的 sequence 从 prefill 路径重新进入：

```python
# scheduler.py:30-52
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    ...
    if not seq.block_table:                                    # [A] block_table 已被清空
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1:
            break
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size  # [B]
    ...
    self.block_manager.allocate(seq, num_cached_blocks)
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    ...
```

**[A]** 由于 `preempt()` 通过 `deallocate()` 清空了 `block_table`，条件成立，进入 `can_allocate` 路径。

**[B]** `seq.num_tokens` 此时包含原始 prompt + 已生成的 token（比如 512 prompt + 200 generated = 712 token）。`num_cached_blocks` 反映了 prefix cache 的命中数量。实际需要重新计算的 token 数 = `712 - num_cached_blocks * 256`。

**Step 3：Prefix cache 恢复**

`can_allocate()` 遍历 sequence 的所有完整 block（除最后一个可能不完整的 block），通过链式哈希查找 `hash_to_block_id` 映射：

```python
# block_manager.py:58-73
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):           # 遍历所有完整 block
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)        # 链式哈希
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break                                   # 链断裂，停止匹配
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1                     # 已被使用的 block 不需要新分配
    if len(self.free_block_ids) < num_new_blocks:
        return -1
    return num_cached_blocks
```

**关键特性：哈希链的脆弱性**。哈希是链式计算的——每个 block 的哈希依赖前一个 block 的哈希。这意味着**如果中间某个 block 的哈希映射被清除（因为该 block 被重新分配给了其他 sequence），后续所有 block 的缓存链都会断裂**，即使后续 block 本身仍然完好。

```
block 链：[block_0] → [block_1] → [block_2] → [block_3]
                          ↑
                    如果这个 block 的哈希被清除
                          |
           [block_0] ✓  [block_1] ✗  [block_2] ✗  [block_3] ✗
                                      ↑              ↑
                             即使这两个 block 仍然完好，也无法匹配
```

### 4.4 关键细节与误区澄清

**误区六：被抢占的 sequence re-prefill 时，是否只需要重新计算"新增"的 token？**

不是。re-prefill 需要处理**所有未被 prefix cache 覆盖的 token**。如果 prefix cache 完全未命中（比如所有 block 都已被其他 sequence 消费），则需要重新计算完整的 `prompt + generated` token。这包括原始 prompt——即使 prompt 在第一次 prefill 时已经处理过。

最佳情况下（所有 block 都被 prefix cache 命中），只需重新计算最后一个不完整 block 中的 token，开销接近于零。

最差情况下（所有 block 都被消费），需要重新计算全部 `P + G` 个 token，开销与首次 prefill 相当，甚至更大（因为 G > 0）。

### 4.5 本章小结

> 💡 **小结**
> - 被抢占的 sequence 通过 `appendleft` 获得 waiting 队列的最高优先级。
> - Re-prefill 的实际计算量取决于 prefix cache 命中：最佳为接近零，最差为全量重算。
> - 哈希链的链式依赖意味着中间任何一个 block 被消费，都会导致后续所有 cache 失效。
> - 多条 sequence 连续抢占时，waiting 队列的顺序与原始 running 队列一致。

---

## 五、完整主路径串联

### 5.1 完整调用栈

```
User: llm.generate(prompts, sampling_params)
  │
  ├─ Step 1: 添加请求
  │     └─ LLMEngine.add_request() → Scheduler.add(seq)
  │        seq 加入 waiting 队尾
  │
  ├─ Step 2: 主循环 (while not is_finished)
  │     └─ LLMEngine.step()
  │           │
  │           ├─ Scheduler.schedule()
  │           │     │
  │           │     ├─ Prefill 阶段（优先）
  │           │     │     ├─ 从 waiting 队首取 seq
  │           │     │     ├─ can_allocate() → 检查 block 可用性 + prefix cache
  │           │     │     ├─ allocate() → 分配 block（复用 cached block）
  │           │     │     ├─ 设置 num_scheduled_tokens
  │           │     │     ├─ 若 prefill 完成 → status=RUNNING, 移入 running
  │           │     │     └─ 返回 (seqs, is_prefill=True)
  │           │     │
  │           │     └─ Decode 阶段（prefill 无任务时）
  │           │           ├─ 从 running 队首取 seq
  │           │           ├─ can_append() → 是否需要新 block？
  │           │           │     ├─ True: 调度成功
  │           │           │     └─ False: ★ 触发抢占 ★
  │           │           │           ├─ running.pop() → preempt(victim)
  │           │           │           │     ├─ victim.status = WAITING
  │           │           │           │     ├─ victim.is_prefill = True
  │           │           │           │     ├─ deallocate(victim) → block 释放
  │           │           │           │     └─ waiting.appendleft(victim)
  │           │           │           └─ 重试 can_append（或自我抢占）
  │           │           ├─ may_append() → 分配新 block（如需）
  │           │           ├─ scheduled_seqs.append(seq)
  │           │           └─ 返回 (seqs, is_prefill=False)
  │           │
  │           ├─ ModelRunner.call("run", seqs, is_prefill)
  │           │     ├─ prepare_prefill/decode → 构建 input_ids, positions, slot_mapping
  │           │     ├─ run_model → 模型前向（prefill: flash_attn_varlen_func; decode: flash_attn_with_kvcache）
  │           │     └─ sampler → 采样生成 token_id
  │           │
  │           └─ Scheduler.postprocess(seqs, token_ids, is_prefill)
  │                 ├─ hash_blocks() → 为新填满的 block 计算哈希（喂入 prefix cache）
  │                 ├─ num_cached_tokens += num_scheduled_tokens
  │                 ├─ append_token(token_id) → 追加生成的 token
  │                 └─ 检查终止条件 → FINISHED → deallocate
  │
  └─ Step 3: 收集输出
        └─ outputs[seq_id] = completion_token_ids
```

### 5.2 每一层做了什么

| 层级 | 输入 | 输出 | 状态变化 | 是否触发通信 | 是否影响显存 | 频率 |
|---|---|---|---|---|---|---|
| `schedule()` | waiting/running 队列 | scheduled_seqs, is_prefill | seq 状态转换、block 分配/释放 | 否 | 是（block 分配/释放） | 每 step |
| `preempt()` | victim sequence | - | status→WAITING, deallocate block | 否 | 是（释放 block） | 仅显存不足时 |
| `prepare_prefill/decode` | seqs | input_ids, positions, slot_mapping | 设置 Context 全局状态 | 否 | 是（CUDA tensor 创建） | 每 step |
| `run_model` | input_ids, positions | logits | - | 是（TP: all_reduce） | 否 | 每 step |
| `postprocess` | seqs, token_ids | - | hash_blocks, append_token, 可能 FINISHED | 否 | 是（FINISHED 时释放 block） | 每 step |

### 5.3 哪些逻辑不在主路径

| 看似相关的逻辑 | 为什么容易误解 | 实际是否在主流程 |
|---|---|---|
| `capture_cudagraph()` | 涉及模型执行 | 仅在初始化时执行一次，不在每步主路径 |
| `warmup_model()` | 也调用 `run()` | 仅初始化时用于计算显存峰值 |
| `Sequence.__getstate__` | 看似每次都序列化 | 仅在 TP > 1 时通过 shared memory 传输 |
| `allocate_kv_cache()` | 分配 KV cache 大块内存 | 仅初始化时执行一次 |

---

## 六、关键机制深挖

### 6.1 FIFO Free List：一个精巧的缓存保留策略

BlockManager 的 free list 使用 `deque` 实现，遵循 FIFO 策略：

```python
# 释放：追加到尾部
self.free_block_ids.append(block_id)     # block_manager.py:56

# 分配：从头部取出
block_id = self.free_block_ids.popleft() # block_manager.py:44
```

这不是偶然的选择，而是一个**刻意的缓存友好设计**。如果使用 LIFO（栈），刚释放的 block 会立即被下一次分配消费，其哈希元数据瞬间失效。FIFO 确保了最近释放的 block 排在队列最后，需要等到前面所有更老的 free block 被消费后才被回收。

**效果量化**：假设 free list 中有 F 个 block，被抢占的 sequence 释放了 R 个 block。这 R 个 block 现在排在 free list 的第 F+1 到第 F+R 位。在它们被回收之前，需要有 F+1 到 F+R 次新的 block 分配。

但在抢占场景下有一个悖论：**抢占发生时 F ≈ 0**（正是因为 free list 为空才触发了抢占）。此时释放的 R 个 block 就是 free list 中仅有的 block，它们排在第 1 到第 R 位。任何后续的 `_allocate_block()` 调用都会立即消费它们。

然而在实践中，每个 decode step 中每条 running sequence 需要新 block 的概率仅为 1/256（仅在跨越 block 边界时）。对于 100 条 running sequence，期望消费 ≈ 0.39 个 block。所以在抢占发生的当前轮次中，**几乎所有释放的 block 都能存活**。被抢占的 sequence 在下一个调度轮次通过 prefill 路径恢复时，大概率能找到自己的 block 仍然在 free list 中，prefix cache 命中率很高。

### 6.2 自我抢占：一个潜在的崩溃点

当 `running` 队列中只有一条 sequence，且这条 sequence 无法获得新 block 时，代码执行自我抢占：

```python
# scheduler.py:63-65
else:
    self.preempt(seq)    # 自我抢占
    break
```

`preempt(seq)` 释放了 seq 的所有 block（这使得 free list 不再为空），然后将 seq 推入 waiting 队首。但 `break` 导致 seq 不被加入 `scheduled_seqs`。外层循环因 `self.running` 为空而退出。随后：

```python
assert scheduled_seqs    # scheduler.py:71  ← 崩溃！
```

这个 `assert` 在 `scheduled_seqs` 为空时触发 `AssertionError`。

**这是一个真实的 bug 吗？**

从纯逻辑角度看，是的。自我抢占后 free list 中有了空闲 block，但代码没有重试。在下一次 `schedule()` 调用中，seq 会从 prefill 路径成功恢复——但当前调用在到达那一步之前就崩溃了。

**是否在实际工作负载中可触发？**

需要满足的条件极为苛刻：

1. `running` 中只有 1 条 sequence（其他 sequence 已在之前的迭代中被正常调度或抢占）
2. 这条 sequence 的 `len(seq) % block_size == 1`（刚跨越 block 边界）
3. `free_block_ids` 为空（所有 block 被占用）

条件 3 意味着这一条 sequence 的 block 占满了整个 KV cache。对于 Qwen3-0.6B 在 A100 80GB 上（约 17,000 个 block = 435 万 token 容量），一条 sequence 要占满所有 block 需要超过 435 万 token——远超 `max_model_len=4096` 的限制。

**结论：在默认配置下不可触发，但在大模型 + 小显存 + 长上下文配置下是真实风险。** 修复方案很简单——将 `assert` 替换为优雅的空返回。

### 6.3 Recompute vs. Swap：nano-vllm 的取舍

nano-vllm 采用纯 recompute 策略——被抢占的 sequence 丢弃所有 KV cache，在 re-prefill 时重新计算。vLLM 完整版还支持 swap 策略——将 KV cache 复制到 CPU 内存，恢复时再拷回。

**为什么 recompute 是合理的简化？**

对于 Qwen3-0.6B（28 层，GQA 4 KV heads，head_dim=128，bfloat16），每个 token 的 KV cache 大小为：

```
KV_per_token = 2 (K+V) × 28 (layers) × 4 (kv_heads) × 128 (head_dim) × 2 (bytes)
             = 57,344 bytes ≈ 56 KB
```

一条 1000 token 的 sequence 的 KV cache ≈ 56 MB。通过 PCIe 4.0（~25 GB/s）传输需要 ~2.2 ms。

而 re-prefill 1000 个 token 在 A100 上（假设 50% 利用率）需要约 5-10 ms。

表面看 swap 更快，但 swap 策略需要引入：
- CPU 端的 block allocator（镜像 GPU 端）
- 异步 CUDA memcpy 编排
- SWAPPED 状态的管理
- Pinned memory 管理
- 防止正在传输的 block 被意外释放

这些会将 BlockManager 的复杂度翻倍。对于一个 ~1,200 行的精简实现，recompute + prefix cache 的组合提供了足够好的恢复能力，同时保持了代码的简洁性。

此外，prefix cache 提供了 recompute 的"最优情况"——如果 freed block 未被消费，re-prefill 只需计算最后一个不完整 block 中的 token，开销接近于零。这在低压场景下几乎等同于免费的 swap。

---

## 七、显存、性能与通信分析

### 7.1 显存收益范围

| 内容 | 抢占是否释放 | 原因 |
|---|---|---|
| KV cache block | ✅ 释放 | `deallocate()` 将 block 归还 free list |
| block 哈希元数据 | ❌ 保留 | 惰性失效，为 prefix cache 保留恢复窗口 |
| Sequence 的 token_ids | ❌ 保留 | 存储在 CPU 内存（Python 列表），不占 GPU 显存 |
| 模型参数 | 不受影响 | 参数在所有 sequence 间共享 |
| CUDA graph 缓存 | 不受影响 | 静态分配，与 sequence 无关 |
| 激活值 | 不适用 | 推理模式下不保留中间激活值 |

**真正的显存大头是 KV cache block**。每个 block 占用 4 MiB（Qwen3-0.6B，block_size=256，TP=1 配置下为 `2 × 32 × 256 × 2 × 64 × 2 = 4,194,304 bytes`；实际 Qwen3-0.6B 为 28 层 4 KV heads 128 head_dim，每 block 约 ~28 MB。具体值取决于模型配置）。抢占一条占用 5 个 block 的 sequence 能释放 ~20-140 MB 显存。

### 7.2 通信开销

抢占本身**不触发任何 GPU 通信**。所有抢占逻辑在 CPU 上的 Scheduler 中完成（仅修改 Python 对象的属性和 deque 操作）。

通信发生在**后续步骤**：
- 被抢占的 sequence re-prefill 时，模型前向传播会触发 TP all_reduce（在 `RowParallelLinear` 和 `ParallelLMHead` 中）
- 序列化开销：TP > 1 时，被抢占的 sequence 因 `is_prefill=True` 会通过 `__getstate__` 发送完整 `token_ids`（而非仅 `last_token`），数据量约为正常 decode 的 70 倍——但绝对值仍在 KB 级别（~9 KiB / 1000 token），远小于 1 MiB shared memory 缓冲区，延迟可忽略（~20 μs）

### 7.3 性能取舍

| 收益 | 代价 |
|---|---|
| 避免 OOM 崩溃 | 被抢占 sequence 需要重新 prefill |
| 保护已投入大量计算的 sequence（LIFO 策略） | 同一条 sequence 可能反复成为 victim |
| Prefix cache 可能减少 re-prefill 开销 | 高压下 prefix cache 命中率低 |
| 实现极简（~10 行核心代码） | 不支持 swap，无法利用 CPU 内存 |

**Re-prefill 的计算开销**（以 Qwen3-0.6B 在 A100 上为例）：

| 被抢占 sequence 长度 | Prefix cache 命中 | 需重算 token 数 | 预估耗时 |
|---|---|---|---|
| 512 prompt + 200 generated | 全部命中（最佳） | (712) % 256 = 200 | ~1-2 ms |
| 512 prompt + 200 generated | 无命中（最差） | 712 | ~5-10 ms |
| 1024 prompt + 500 generated | 全部命中 | (1524) % 256 = 244 | ~1-3 ms |
| 1024 prompt + 500 generated | 无命中 | 1524 | ~10-20 ms |

作为对比，一个正常的 decode step（256 条 sequence 并行）耗时约 5-10 ms。所以一次 re-prefill 的开销大致等于 1-2 个 decode step。

### 7.4 内部碎片分析

block_size=256 的内部碎片：

| 指标 | 值 |
|---|---|
| 最差碎片（每 sequence） | 255 token 的 block 空间被浪费（99.6%） |
| 平均碎片（均匀分布假设） | 127.5 token（约 50% 的最后一个 block） |
| 对总体利用率的影响 | 平均 sequence 1124 token → 5 blocks → 利用率 87.8% |

碎片对抢占的间接影响：碎片浪费了 block 空间，减少了有效 token 容量，理论上会使抢占更早发生。但在典型工作负载（A100 + Qwen3-0.6B）下，KV cache 容量远超需求，碎片的影响可以忽略。

---

## 八、配置项、边界条件与坑点

### 8.1 影响抢占行为的配置

| 配置项 | 影响 | 源码路径 | 行为变化 | 风险/坑点 |
|---|---|---|---|---|
| `num_kvcache_blocks` | 直接决定总 block 数量 | `model_runner.py:113` 自动计算 | block 越多，抢占越不可能发生 | 由 `gpu_memory_utilization` 间接控制 |
| `gpu_memory_utilization` | 控制 KV cache 可用显存比例 | `config.py:11` | 默认 0.9，降低则减少可用 block | 设置过低可能导致频繁抢占 |
| `max_num_seqs` | 最大并发 sequence 数 | `config.py:10` | 默认 512，减少可降低显存压力 | 减少并发但降低吞吐 |
| `max_model_len` | 单 sequence 最大长度 | `config.py:12` | 默认 4096，增加可能触发自我抢占 | 与 block 数量形成直接约束 |
| `kvcache_block_size` | block 粒度 | `config.py:17` | 默认 256，减小可降低碎片但增加管理开销 | 必须是 256 的倍数 |
| `max_num_batched_tokens` | 每步最大处理 token 数 | `config.py:9` | 默认 16384，影响 re-prefill 时 chunked prefill 的分块大小 | 设置过小会导致 re-prefill 分多步完成 |

### 8.2 抢占触发的精确条件

```
抢占发生 ⟺ 以下条件同时成立：
  1. waiting 队列为空（否则 prefill 阶段会先执行，不进入 decode）
  2. 某条 running sequence 的 len(seq) % block_size == 1
  3. free_block_ids 为空
```

条件 2 中 `len(seq)` 是包含已生成 token 的总长度。每条 sequence 每 256 步才跨越一次 block 边界，所以条件 2 在任一给定步只影响约 1/256 的 running sequence。

### 8.3 不会触发抢占的场景

| 场景 | 原因 |
|---|---|
| Prefill 阶段 block 不足 | `can_allocate` 返回 -1，sequence 留在 waiting，不触发抢占 |
| Decode 但不需要新 block | `can_append` 返回 True，正常调度 |
| Decode 需要新 block 但 free list 有余 | `can_append` 返回 True |

---

## 九、局限性与已知优化点

### 9.1 硬约束

1. **不支持 swap-to-CPU**：被抢占的 sequence 只能通过 recompute 恢复，无法利用 CPU 内存作为溢出空间
2. **全量释放**：抢占释放 sequence 的**所有** block，无法做部分释放（例如只释放最近的几个 block）
3. **单一 victim 策略**：LIFO 是唯一的 victim 选择策略，无法根据 sequence 特征（长度、剩余 token 数、block 占用量）选择最优 victim
4. **哈希链脆弱**：prefix cache 的链式哈希在中间任一 block 被消费后全链断裂

### 9.2 维护成本

1. **`while...else` 语法**：这是 Python 中最不直观的语法之一，新维护者很可能误解控制流
2. **`can_append` 的布尔运算技巧**：`len(free) >= (len(seq) % bs == 1)` 虽然简洁但不易读
3. **`num_scheduled_tokens` 的隐式不变量**：`preempt()` 不重置该字段，依赖调用顺序的隐式保证
4. **`assert scheduled_seqs` 的隐式假设**：假设自我抢占不可能发生，但未通过配置验证保证

### 9.3 性能瓶颈

1. **`free_block_ids.remove(block_id)`**（`block_manager.py:87`）：在 `allocate()` 中，当 prefix cache 命中的 block 仍在 free list 中时，需要通过 `deque.remove()` 将其取出。这是 O(N) 操作，N 为 free list 长度。在大规模场景（数万 block）下可能成为瓶颈
2. **逐条抢占**：内层 `while` 循环每次只抢占一条 sequence，每次抢占都要重新检查 `can_append`。理论上可以计算需要释放多少 block 后一次性抢占足够的 victim
3. **Re-prefill 无法与 decode 重叠**：两阶段调度保证了 prefill 和 decode 互斥，被抢占的 sequence 的 re-prefill 会占用一整个调度轮次

### 9.4 已知优化点

1. **victim 选择优化**：可以按 `min(num_completion_tokens)`（最少已生成 token）选择 victim，最小化浪费的计算量
2. **部分释放**：只释放 victim 最后几个 block，保留 prefix 部分，减少 re-prefill 开销
3. **Free list 查找优化**：用 `set` 辅助 `deque` 实现 O(1) 的 block 存在性检查
4. **Batch preemption**：一次性计算需要释放多少 block，然后按策略选择最少数量的 victim
5. **Swap 支持**：为大模型场景引入 CPU 内存 swap，避免 recompute 开销
6. **自我抢占修复**：将 `assert scheduled_seqs` 替换为条件判断，允许当前轮次返回空调度，在下一轮通过 prefill 恢复

---

## 十、测试覆盖与风险分析

### 10.1 当前状态

nano-vllm **没有测试套件**（CLAUDE.md 明确指出"There is no test suite or linter configured"）。抢占机制的正确性完全依赖于：
- `bench.py`：256 条 sequence，随机长度 100-1024 input + 100-1024 output。但在 A100 + Qwen3-0.6B 配置下，KV cache 容量远超需求，**抢占几乎不会被触发**
- `example.py`：仅 2 条短 sequence，更不可能触发抢占

### 10.2 未覆盖风险

| 风险点 | 是否有测试 | 可能后果 |
|---|---|---|
| 单 sequence 自我抢占 | ❌ | `assert scheduled_seqs` 崩溃 |
| 多条 sequence 连续抢占 | ❌ | 逻辑正确但未验证 |
| 抢占后 prefix cache 命中恢复 | ❌ | 效果未量化 |
| 抢占-恢复-再抢占循环（thrashing） | ❌ | 未验证系统最终收敛 |
| 抢占与 chunked prefill 的交互 | ❌ | re-prefill 可能被分多块，未验证 |
| 抢占与 tensor parallelism 的交互 | ❌ | `is_prefill=True` 导致序列化数据量增大，未验证 SharedMemory 大小限制 |
| 大模型 + 小显存配置 | ❌ | 自我抢占 / 频繁抢占 / 哈希链断裂 |

---

## 小结与展望

nano-vllm 的抢占机制实现可以用几个关键词概括。

**关键词一：Recompute-only**

nano-vllm 选择了最简单的抢占恢复策略——完全丢弃 KV cache，在 re-prefill 时重新计算。这避免了 swap 策略所需的 CPU 内存管理、异步传输、状态机扩展等工程复杂度，代价是在显存紧张时需要付出重复计算的代价。对于一个 ~1,200 行的教学级实现，这是正确的取舍。

**关键词二：惰性缓存失效**

block 释放后保留哈希元数据，直到被重新分配时才真正失效。这使得 prefix cache 能在抢占-恢复周期中发挥作用，将 recompute 的最佳情况开销降低到接近零。FIFO free list 进一步延长了 freed block 的存活窗口。

**关键词三：LIFO victim + 优先恢复**

victim 选择使用 LIFO 策略（保护已投入最多计算的 sequence），恢复使用 `appendleft`（被抢占的 sequence 获得最高调度优先级）。这构成了一个"牺牲新手、优先恢复"的调度哲学——合理但可能导致 victim 锁定。

**关键词四：10 行代码的工程杠杆**

`preempt()` 方法仅 5 行，decode 循环中的抢占逻辑仅 8 行。加上 `deallocate` 和 `can_append`，整个抢占机制不超过 25 行核心代码。但这 25 行代码解决了 LLM 推理引擎中最关键的资源管理问题之一：在有限显存下保证系统不崩溃、不停滞。

---

**适合的场景**：中小模型、显存充裕、抢占罕见发生的推理场景。在这种场景下，抢占机制作为安全网存在，prefix cache 提供了几乎免费的恢复能力。

**不适合的场景**：大模型、长上下文、显存紧张、高并发的生产场景。频繁抢占会导致 recompute 开销显著，LIFO victim 选择可能引发公平性问题，自我抢占的 assert 崩溃是潜在风险。

**与 vLLM 完整版的比较**：vLLM 同时支持 recompute 和 swap 两种策略，有更复杂的 victim 选择逻辑（考虑 sequence 优先级），支持部分 block 保留，且有完整的测试覆盖。nano-vllm 的实现是对核心逻辑的精确提炼——它证明了抢占机制的本质并不复杂，复杂的是工程化的边界处理。

**后续值得继续走读的方向**：
1. vLLM 的 swap-to-CPU 实现——理解完整版的 KV cache 迁移如何在不丢弃数据的前提下释放 GPU 显存
2. vLLM 的 preemption priority 策略——如何基于 sequence 的 arrival time、prompt length、SLO 等因素进行更精细的 victim 选择
3. nano-vllm 的 Chunked Prefill 与抢占的交互——一条被抢占的长 sequence 在 re-prefill 时可能需要多轮 chunked prefill，这对调度延迟和吞吐有何影响
