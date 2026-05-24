# nano-vllm 源码走读：KV Cache 容量计算 实现解析

> 在上一篇 Paged Attention 走读中，我们已经了解了 nano-vllm 如何用"分页"机制管理 KV Cache 的物理块。但有一个根本性的问题一直没有展开：**这些物理块到底有多少个？每个块需要多少显存？系统是如何在启动时计算出"我能分配多少块 KV Cache"的？**
>
> 这不是一个简单的除法问题。推理引擎需要在一块 GPU 上同时容纳模型权重、推理时的激活值、CUDA Graph 的中间态，以及 KV Cache 本身——这些消费者争抢同一块显存，任何一方高估都会导致 OOM，低估则浪费了本可用于缓存更多请求的空间。本文不讲 PagedAttention 的原理，而是聚焦源码，分析 nano-vllm 如何把"GPU 还剩多少显存"这个物理层问题，翻译成 BlockManager 能管理的逻辑块数量，以及这个过程中的设计取舍和隐藏陷阱。

---

# 前言

## 业务与工程背景

KV Cache 容量计算处于 LLM 推理引擎初始化的核心位置。它决定了引擎能同时服务多少个请求、每个请求能支撑多长的上下文。容量算少了，scheduler 频繁触发抢占（preemption），吞吐量下降；算多了，KV Cache 张量分配时直接 OOM，引擎启动失败。

## 核心矛盾

矛盾可以用一句话概括：**KV Cache 必须在所有其他显存消费者"就位"之后才能知道自己能分多少，但分配时又必须提前预判前向推理时的峰值显存占用**。

具体来说，存在三重冲突：

1. **模型权重 vs. KV Cache**：权重是确定的，但加载后的碎片化不确定。
2. **激活值峰值 vs. 实际推理**：prefill 和 decode 的激活值峰值不同，需要用最坏情况来预留，但这个最坏情况只有跑一遍才知道。
3. **CUDA Graph vs. KV Cache**：CUDA Graph 在 KV Cache 分配**之后**才捕获，但它会吃掉一部分显存，这部分在 KV Cache 分配时还不存在。

## 本文主线

本文围绕以下机制逐章展开：

- **第一章**：Warmup 显存探测——用一次"假"前向跑出真实峰值显存
- **第二章**：容量公式拆解——从 GPU 总显存推导出可用 block 数
- **第三章**：KV Cache 张量的物理布局与层分配
- **第四章**：Block 生命周期与调度器的容量交互
- **第五章**：完整主路径串联
- **第六章**：显存、性能与通信分析
- **第七章**：配置项、边界条件与坑点
- **第八章**：局限性与已知优化点

## 不展开的内容

本文不讲 PagedAttention 的学术原理，不讲 Flash Attention 的内部实现，不讲 Qwen3 模型架构——只讲 nano-vllm 框架是如何**计算出 KV Cache 应该分配多少显存、分成多少块**的。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `nanovllm/config.py` | 定义 `gpu_memory_utilization`、`kvcache_block_size` 等用户配置 |
| `nanovllm/engine/model_runner.py` | `warmup_model()` 探测显存峰值、`allocate_kv_cache()` 计算 block 数量并分配张量 |
| `nanovllm/engine/block_manager.py` | 管理 block 池的分配、回收、prefix cache 哈希 |
| `nanovllm/engine/scheduler.py` | 在调度时查询和消费 block 容量 |
| `nanovllm/engine/sequence.py` | 每条序列的 block table、块数计算 |
| `nanovllm/engine/llm_engine.py` | 编排初始化顺序：ModelRunner 先于 Scheduler |
| `nanovllm/layers/attention.py` | KV Cache 的最终消费者：Triton 写入和 Flash Attention 读取 |
| `nanovllm/utils/context.py` | 全局 Context，在 ModelRunner 和 Attention 之间传递 slot_mapping 等元数据 |

---

# 一、Warmup 显存探测：用一次"假"前向跑出真实峰值

## 1.1 设计哲学与核心问题

要计算"GPU 还剩多少显存给 KV Cache"，首先需要知道"模型运行时最多要吃多少显存"。这是一个看似简单、实则棘手的问题。

静态估算的方式——根据模型参数量乘以 dtype size 加上理论激活值——在实际中误差很大。原因包括：PyTorch 内存分配器的碎片化、`torch.compile` 编译产生的缓存、Flash Attention 的 workspace、RoPE 的 cos/sin 缓存等等，这些都无法从模型配置文件精确推算。

nano-vllm 的解决方案是**经验主义的**：跑一次真实的前向推理，让 PyTorch 的内存统计告诉你实际峰值是多少。

## 1.2 源码入口与关键对象

```
nanovllm/engine/model_runner.py
  - warmup_model()（L:91-101）：构造最大规模的 dummy 输入，执行一次前向
  - allocate_kv_cache()（L:103-121）：读取峰值统计，计算 block 数
```

## 1.3 主流程拆解

`warmup_model()` 的完整代码只有 11 行，但每一行都有讲究：

```python
def warmup_model(self):
    torch.cuda.empty_cache()                    # 释放 PyTorch 缓存池，归还给 CUDA 运行时
    torch.cuda.reset_peak_memory_stats()        # 重置峰值水位线
    max_num_batched_tokens = self.config.max_num_batched_tokens  # 默认 16384
    max_model_len = self.config.max_model_len                    # 默认 4096
    seq_len = min(max_num_batched_tokens, max_model_len)         # = 4096
    num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)  # = 4
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)                        # 执行一次完整的 prefill 前向
    torch.cuda.empty_cache()                    # 清掉中间激活，只留权重
```

**为什么用默认参数得到的是 4 条长度 4096 的序列？** `seq_len = min(16384, 4096) = 4096`，`num_seqs = min(16384 // 4096, 512) = 4`。总共 16384 个 token——刚好等于 `max_num_batched_tokens`，这是系统允许的单步最大处理量。这确保 warmup 测量的是**最坏情况**下的激活值峰值。

**前向跑完后调 `empty_cache()` 的用意**：前向推理产生的中间张量（attention scores、MLP 中间层等）在 `self.run()` 结束后已经没有引用，但它们仍然占据 PyTorch 的缓存池。`empty_cache()` 把这些缓存池归还给 CUDA 运行时，使得后续的 `torch.cuda.mem_get_info()` 能看到这些显存已经 free。但关键是：PyTorch 的 `memory_stats()["allocated_bytes.all.peak"]` 记住了这次前向推理的峰值——这个统计值不会因为 `empty_cache()` 而丢失。

## 1.4 关键细节与误区澄清

**误区一：warmup 跑的是"完整"的推理流程。**

不完全正确。warmup 时 KV Cache 尚未分配（`self.k_cache` 和 `self.v_cache` 还是空张量 `torch.tensor([])`），所以 Triton 的 `store_kvcache_kernel` 在 `attention.py:62-63` 处被跳过：

```python
if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

这意味着 warmup **不测量** KV Cache 写入的显存开销。不过这个写入本身不分配新内存（它写入的是已分配的 `kv_cache` 张量），所以不影响峰值统计的正确性。

**误区二：warmup 捕获的峰值涵盖了所有运行时开销。**

不完全涵盖。warmup 只跑 prefill 路径，不跑 decode。但 decode 的激活值远小于 prefill（每条序列只有 1 个 token vs. 上千个），所以 prefill 的峰值确实是上界。真正的遗漏是 **CUDA Graph 的显存**——它在 `allocate_kv_cache()` 之后的 `capture_cudagraph()` 中才分配，warmup 阶段根本不知道 CUDA Graph 会吃多少显存。

## 1.5 本章小结

> 💡 小结
> - Warmup 用经验测量替代了静态估算，通过一次最大规模的 dummy prefill 跑出真实的显存峰值。
> - 峰值信息保存在 PyTorch 的 `memory_stats()` 中，不受后续 `empty_cache()` 影响。
> - Warmup 的盲区是 KV Cache 写入内核和 CUDA Graph——前者无影响，后者是个实际风险。

---

# 二、容量公式拆解：从 GPU 总显存到可用 Block 数

## 2.1 设计哲学与核心问题

Warmup 解决了"运行时峰值是多少"的问题，接下来要回答"KV Cache 能分多少"。这需要一个公式，把 GPU 的物理显存总量、已占用量、峰值激活量折算成 KV Cache 的 block 数。

这个公式看似是一行简单的除法，但它混合了两套不同的显存统计体系（CUDA 运行时 vs. PyTorch 分配器），理解起来需要格外小心。

## 2.2 源码入口与关键对象

```
nanovllm/engine/model_runner.py
  - allocate_kv_cache()（L:103-121）：核心容量计算和张量分配
nanovllm/config.py
  - gpu_memory_utilization（L:12）：用户控制的显存使用比例，默认 0.9
  - kvcache_block_size（L:17）：每个 block 的 token 容量，默认 256
  - num_kvcache_blocks（L:18）：运行时计算得出的 block 总数，初始值 -1
```

## 2.3 主流程拆解

### 第一步：读取显存状态

```python
free, total = torch.cuda.mem_get_info()           # CUDA 运行时级别的统计
used = total - free                                 # 当前 GPU 上所有显存消费
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]     # PyTorch 分配器峰值
current = torch.cuda.memory_stats()["allocated_bytes.all.current"] # PyTorch 当前占用
```

这里涉及两套统计体系：

| 统计来源 | 涵盖范围 | 此刻的含义 |
|---|---|---|
| `torch.cuda.mem_get_info()` | CUDA 运行时级别，包含所有 GPU 消费者 | `free` = GPU 完全空闲的显存；`total` = GPU 总显存 |
| `memory_stats()["peak"]` | 仅 PyTorch 分配器 | warmup 期间的最大 PyTorch 分配（权重 + 激活峰值） |
| `memory_stats()["current"]` | 仅 PyTorch 分配器 | warmup 后的 PyTorch 当前分配（≈ 模型权重） |

`used = total - free` 包含了 CUDA context（约 300-800 MB）、NCCL 静态缓冲、PyTorch 分配器的持有量等所有开销。而 `peak` 和 `current` 只看到 PyTorch 自己分配的部分。

### 第二步：计算每个 block 的字节大小

```python
num_kv_heads = hf_config.num_key_value_heads // self.world_size
head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
```

公式含义：

```
block_bytes = 2（K+V） × num_layers × block_size × num_kv_heads_per_rank × head_dim × dtype_size
```

- `2`：每个 token 需要同时存储 Key 和 Value
- `num_layers`：每一层 Transformer 都有独立的 KV Cache
- `block_size`：每个 block 容纳的 token 数（默认 256）
- `num_kv_heads_per_rank`：Tensor Parallelism 下，KV head 数被 `world_size` 整除后分到每个 rank
- `head_dim`：每个 attention head 的维度
- `dtype_size`：每个元素的字节数（bf16 = 2，fp32 = 4）

### 第三步：核心容量公式

```python
config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
```

这行代码是整个 KV Cache 容量计算的灵魂，值得拆成独立的数学表达式理解：

```
可用显存 = total × utilization − used − (peak − current)
         = 显存预算 − GPU当前总占用 − 推理峰值激活开销
```

其中：

```
显存预算      = total × utilization        # 例如 80GB × 0.9 = 72GB
GPU当前总占用 = used = total − free         # 包括 CUDA 上下文 + 模型权重 + PyTorch 分配器开销
峰值激活开销  = peak − current             # warmup 跑出的最大激活值减去当前持有量（≈权重）
```

进一步化简：

```
可用显存 = total × utilization − (total − free) − (peak − current)
         = free − total × (1 − utilization) − (peak − current)
         = 当前空闲显存 − 安全边际 − 峰值激活预留
```

**直觉理解**：从当前空闲显存出发，扣掉 10% 安全边际（用于容纳 CUDA Graph、Triton JIT 等未知开销），再扣掉 warmup 测出的峰值激活预留，剩下的就是 KV Cache 能用的。

### 第四步：写回 Config

```python
config.num_kvcache_blocks = ... // block_bytes
assert config.num_kvcache_blocks > 0
```

计算结果直接写入 `config` 对象。Scheduler 在之后的初始化中读取这个值来创建 BlockManager。这里有一个**隐式的时序耦合**：`LLMEngine.__init__()` 在 `llm_engine.py:31` 先创建 `ModelRunner`（触发 warmup 和 allocate_kv_cache），然后在 `llm_engine.py:34` 创建 `Scheduler`（读取 `config.num_kvcache_blocks`）。如果构造顺序反过来，Scheduler 会拿到初始值 -1。

## 2.4 关键细节与误区澄清

**误区三：`used` 和 `peak` / `current` 是同一套统计。**

不是。`used` 来自 CUDA 运行时（`torch.cuda.mem_get_info()`），包含所有 GPU 消费者；`peak` 和 `current` 来自 PyTorch 的内部分配器（`torch.cuda.memory_stats()`），只统计 PyTorch 自己分配的显存。两套体系的差值 `used - current` 代表了非 PyTorch 的 GPU 开销（CUDA context、NCCL 缓冲、驱动程序等）。公式通过同时使用两套统计，巧妙地分离出了这部分开销：

```
non_pytorch_overhead = used − current
available = total × utilization − non_pytorch_overhead − current − (peak − current)
          = total × utilization − used − peak + current    ✓
```

**误区四：`gpu_memory_utilization = 0.9` 意味着 10% 的显存被浪费了。**

这 10% 并非浪费，它是 CUDA Graph 和其他不可预见开销的隐性预算。如果把 utilization 设到 0.99，CUDA Graph 捕获（发生在 KV Cache 分配之后）可能直接 OOM。

## 2.5 数值示例

以一个具体模型配置来感受公式的实际效果：

| 参数 | 值 |
|---|---|
| 模型 | Qwen3-0.6B |
| num_layers | 28 |
| num_kv_heads | 4 |
| head_dim | 128 |
| dtype | bf16 (2 bytes) |
| block_size | 256 |
| GPU | A100 80GB |
| gpu_memory_utilization | 0.9 |

**每个 block 的显存：**

```
block_bytes = 2 × 28 × 256 × 4 × 128 × 2
            = 2 × 28 × 256 × 1024
            = 14,680,064 bytes
            ≈ 14 MB
```

**可用 block 数（假设）：**

```
假设 GPU 总显存 = 80 GB
     模型权重 ≈ 1.2 GB（0.6B params × 2 bytes）
     CUDA 上下文 ≈ 0.5 GB
     峰值激活 ≈ 1.5 GB（16384 tokens prefill）

available ≈ 80 × 0.9 − (0.5 + 1.2) − (1.2 + 1.5) + 1.2
          = 72 − 1.7 − 2.7 + 1.2
          ≈ 68.8 GB

num_blocks ≈ 68.8 × 1024³ / 14,680,064
           ≈ 4,907 个 block
```

每个 block 可存 256 个 token 的 KV，总共可缓存约 **125 万个 token** 的 KV 数据。对于 0.6B 的小模型来说，KV Cache 不是瓶颈。

**换成 Qwen3-72B（80 层，8 KV heads，head_dim 128）：**

```
block_bytes = 2 × 80 × 256 × 8 × 128 × 2
            = 83,886,080 bytes
            ≈ 80 MB per block

假设 8 卡 TP，每卡 num_kv_heads = 1：
block_bytes_per_rank = 2 × 80 × 256 × 1 × 128 × 2
                     ≈ 10 MB per block per rank
```

大模型的每 block 显存成本高出数倍，可用 block 数大幅减少，KV Cache 容量成为真正的瓶颈。

## 2.6 本章小结

> 💡 小结
> - 容量公式混合使用 CUDA 运行时和 PyTorch 分配器两套统计，分离出非 PyTorch 开销和峰值激活预留。
> - `gpu_memory_utilization` 不是"显存利用率"的目标值，而是"显存使用上限"的安全阀，它的 10% 余量隐性地覆盖了 CUDA Graph 等后续开销。
> - 计算结果通过 `config` 对象的可变引用传递给 Scheduler，存在隐式的初始化时序耦合。

---

# 三、KV Cache 张量的物理布局与层分配

## 3.1 设计哲学与核心问题

计算出 block 数之后，下一步是把这些 block 实际分配成 GPU 上的张量。这里的核心设计决策是：**用一个整体张量还是每层一个张量？** nano-vllm 选择了前者——一个 6 维的巨型张量，所有层共享。

## 3.2 源码入口与关键对象

```
nanovllm/engine/model_runner.py
  - allocate_kv_cache()（L:115-121）：分配张量并切片分发给各层
nanovllm/layers/attention.py
  - Attention.__init__()（L:57）：初始化时设置空张量占位
  - Attention.forward()（L:59-75）：使用 k_cache/v_cache 进行缓存读写
```

## 3.3 主流程拆解

### 张量分配

```python
self.kv_cache = torch.empty(
    2,                              # 维度 0：K 和 V
    hf_config.num_hidden_layers,    # 维度 1：每层 Transformer
    config.num_kvcache_blocks,      # 维度 2：block 数（分页维度）
    self.block_size,                # 维度 3：每个 block 内的 token 槽位
    num_kv_heads,                   # 维度 4：KV head 数（TP 下按 rank 分片）
    head_dim                        # 维度 5：每个 head 的维度
)
```

Shape 示意（以 Qwen3-0.6B, 500 blocks 为例）：

```
kv_cache: [2, 28, 500, 256, 4, 128]
           │   │   │    │   │   └── head_dim
           │   │   │    │   └── num_kv_heads (per rank)
           │   │   │    └── tokens per block
           │   │   └── total blocks
           │   └── layers
           └── K / V
```

### 层切片分发

```python
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]   # shape: [blocks, 256, heads, dim]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

这段代码遍历模型所有子模块，找到 `Attention` 层（它们在 `__init__` 时设置了 `k_cache` 和 `v_cache` 属性），然后用 `self.kv_cache` 的切片**替换**空占位张量。

关键点：切片是 **view 而非 copy**，不产生额外显存开销。所有层共享同一个物理张量的不同区域，这意味着所有层使用**同一套 block ID 空间**——block #42 在第 0 层和第 27 层对应的物理显存位置不同，但 block_table 中的 ID 是相同的。

### 为什么选择单一整体张量？

**优点**：一次 `torch.empty()` 调用避免了 28 次（或更多次）独立分配带来的显存碎片化问题。PyTorch 的 CUDA 分配器在分配小块时可能产生内部碎片，而一次性分配一个大块可以最大化利用连续显存。

**代价**：全有或全无——如果计算出的 block 数使得张量刚好超出可用显存，整个分配失败，没有"少分几个 block 再试一次"的回退逻辑。

## 3.4 Triton 写入内核如何寻址

`store_kvcache_kernel`（`attention.py:10-30`）负责把当前步的 K/V 写入缓存。它的寻址逻辑依赖 `slot_mapping`：

```
物理地址 = slot × D + head_offset
其中 slot = block_id × block_size + offset_in_block
     D = num_kv_heads × head_dim
```

核心代码：

```python
slot = tl.load(slot_mapping_ptr + idx)
if slot == -1: return                      # CUDA Graph 填充的无效槽位
cache_offsets = slot * D + tl.arange(0, D)
tl.store(k_cache_ptr + cache_offsets, key)
```

`slot == -1` 的判断是为了 CUDA Graph 服务的：Graph 使用固定大小的 `slot_mapping` 张量（大小 = `max_num_seqs`），实际 batch 可能小于这个大小，多余的位置用 -1 填充。

## 3.5 关键细节与误区澄清

**误区五：KV Cache 张量没有指定 dtype，可能默认为 fp32。**

实际上是安全的。`allocate_kv_cache()` 发生在 `model_runner.py:35`，而 `torch.set_default_dtype(hf_config.dtype)` 在 `model_runner.py:29` 已经执行。所以 `torch.empty(...)` 使用的是模型的 dtype（通常是 bf16）。但这是个隐式依赖——如果未来在两行之间插入了改变 default dtype 的代码，张量的 dtype 就会出错。显式传入 `dtype=hf_config.dtype` 会更稳健。

## 3.6 本章小结

> 💡 小结
> - KV Cache 用一个 6 维张量一次性分配，避免碎片化但牺牲了分配失败时的回退能力。
> - 各 Attention 层通过 view 切片引用同一张量的不同区域，block ID 空间全局统一。
> - Triton 写入内核用 `slot_mapping` 定位物理地址，`slot == -1` 作为 CUDA Graph padding 的哨兵值。

---

# 四、Block 生命周期与调度器的容量交互

## 4.1 设计哲学与核心问题

KV Cache 张量分配完之后，数据层面的容量已经确定。但"有多少个 block 可以用"是一个**动态问题**——运行中的序列占用 block，完成的序列释放 block，新来的序列需要分配 block。BlockManager 负责管理这个动态池，Scheduler 在每一步调度时向它查询"还够不够用"。

这层的核心问题是：**如何在保证正确性的前提下，最大化 block 的重用（通过 prefix caching）并优雅地处理容量不足（通过 preemption）？**

## 4.2 源码入口与关键对象

```
nanovllm/engine/block_manager.py
  - BlockManager.__init__()（L:28-33）：初始化 block 池
  - can_allocate()（L:58-73）：检查是否有足够 block 服务一个新序列
  - allocate()（L:75-92）：实际分配 block
  - can_append()（L:103-104）：decode 阶段检查是否需要新 block
  - deallocate()（L:94-101）：释放序列的所有 block

nanovllm/engine/scheduler.py
  - schedule()（L:25-73）：prefill/decode 两阶段调度
  - preempt()（L:75-79）：抢占一个运行中的序列
```

## 4.3 Block 状态机

一个 Block 在其生命周期中经历以下状态转换：

```
                        ┌──────────────────────────────────────┐
                        │                                      │
                        ▼                                      │
  ┌──────────┐  _allocate_block()  ┌──────────┐  hash_blocks() ┌──────────────┐
  │   FREE   │ ──────────────────> │  IN-USE  │ ─────────────> │ IN-USE+HASHED│
  │ ref=0    │    (L:43-51)        │ ref=1    │   (L:110-121)  │ ref≥1        │
  │ hash=-1  │                     │ hash=-1  │                │ hash=H       │
  └──────────┘                     └──────────┘                └──────┬───────┘
       ▲                                                              │
       │                                                    deallocate()
       │                                                    ref_count--
       │                     ┌──────────────┐                         │
       │   _deallocate       │  EVICTABLE   │ <───────────────────────┘
       │   _block()          │  ref=0       │   当 ref_count 降到 0
       │   (当被新分配      │  hash仍保留  │
       │    从队首取出)      │  (在free队列) │
       └─────────────────────┴──────────────┘
                                    │
                              新序列hash匹配
                              allocate()重新激活
                                    │
                                    ▼
                            ┌──────────────┐
                            │ PREFIX CACHE  │
                            │ HIT: ref=0→1 │
                            └──────────────┘
```

关键设计：Block 被释放后**不清除** hash 和 token_ids。它进入 free 队列的**尾部**（`deque.append`），而新分配从**头部**取出（`deque.popleft`）。这构成了一个隐式的 **LRU 淘汰策略**——最近释放的 block 在队列尾部有最长的存活时间，最大化被后续相同前缀的请求复用的概率。

## 4.4 调度器如何消费容量

### Prefill 阶段：`can_allocate` + `allocate`

```python
# scheduler.py:36-45
num_cached_blocks = self.block_manager.can_allocate(seq)
if num_cached_blocks == -1:
    break                                    # 容量不足，停止调度
# ...
self.block_manager.allocate(seq, num_cached_blocks)
```

`can_allocate` 做两件事：（1）通过 xxhash 链式哈希检查前缀缓存命中；（2）判断 free block 是否足够。它的判断是**保守的**：只有已经在 `used_block_ids` 中的 cached block 才能减少 `num_new_blocks` 的计数。在 free 队列中的 cached block（虽然数据还在，且会被 `allocate()` 重新激活）不减少 `num_new_blocks`——这意味着某些本可以调度的请求可能被不必要地拒绝。

### Decode 阶段：`can_append` + preemption

Decode 每步只追加一个 token。只有当新 token 刚好越过 block 边界时才需要分配新 block：

```python
# block_manager.py:103-104
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

这行代码把布尔值 `(len(seq) % self.block_size == 1)` 隐式转换为 0 或 1 参与比较。当 `len(seq)` 是 block_size 的倍数加 1 时（刚越过边界），需要 1 个新 block。

如果没有空闲 block，调度器触发**抢占**：

```python
# scheduler.py:60-65
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())    # 驱逐最后加入的序列（LIFO）
    else:
        self.preempt(seq)                    # 自我驱逐
        break
```

抢占策略是**全量回收 + 重计算**：被抢占的序列释放所有 block，回到 waiting 队列重新 prefill。但由于释放的 block 保留了 hash，后续重新调度时可能通过 prefix cache 跳过大部分重计算。

## 4.5 关键细节与误区澄清

**误区六：抢占后序列可以"断点续传"。**

不完全对。`preempt()` 把序列状态重置为 `WAITING` 和 `is_prefill = True`，block_table 被清空。它必须从头 prefill——但如果之前的 block hash 还在 free 队列中，`can_allocate` 会识别出来，`allocate` 会复用这些 block，跳过已缓存 token 的 KV 计算。所以它不是真正的"从头来"，而是"从上次的 prefix cache 点续传"。

**误区七：单个超长序列不会导致系统崩溃。**

会的。在 `scheduler.py:58-71` 的 decode 阶段，如果一个序列的 block 需求耗尽了所有 block，且它是唯一的运行中序列，代码会执行 `self.preempt(seq)` 后 `break`。此时 `scheduled_seqs` 为空，随后的 `assert scheduled_seqs`（L:71）会触发 `AssertionError`。这是一个真实的边界条件 bug：当序列长度乘以 per-block 开销超过整个 KV Cache 容量时，系统没有优雅的错误处理。

## 4.6 本章小结

> 💡 小结
> - BlockManager 用 deque 的 FIFO 语义实现了隐式 LRU 淘汰，最大化 prefix cache 命中率。
> - `can_allocate` 的保守估计可能导致不必要的调度拒绝，但保证了正确性。
> - 抢占是全量回收策略（recompute 而非 swap），依赖 prefix cache 减轻重计算开销。
> - 单序列耗尽全部 block 时存在 assert 崩溃的边界 bug。

---

# 五、完整主路径串联

## 5.1 完整调用栈

```
User: LLM(model_path, gpu_memory_utilization=0.9, kvcache_block_size=256)
  │
  ├─ Step 1: 配置加载与校验
  │     └─ Config.__post_init__()                    config.py:20-25
  │         ├─ 加载 HuggingFace 配置
  │         ├─ assert kvcache_block_size % 256 == 0
  │         └─ max_model_len = min(user_val, hf_config.max_position_embeddings)
  │
  ├─ Step 2: Sequence 类配置
  │     └─ Sequence.block_size = config.kvcache_block_size    llm_engine.py:21
  │
  ├─ Step 3: Tensor Parallelism Worker 进程启动（如果 TP > 1）
  │     └─ 为每个 rank > 0 创建 mp.Process            llm_engine.py:25-30
  │
  ├─ Step 4: ModelRunner 初始化（rank 0，也是每个 worker 的入口）
  │     ├─ 4a: 加载模型权重到 GPU                      model_runner.py:31-32
  │     ├─ 4b: warmup_model()                           model_runner.py:91-101
  │     │       ├─ empty_cache() + reset_peak_stats()
  │     │       ├─ 构造 max_num_batched_tokens 个 dummy token
  │     │       ├─ 执行一次完整 prefill 前向
  │     │       └─ empty_cache()（释放中间激活，保留峰值统计）
  │     ├─ 4c: allocate_kv_cache()                      model_runner.py:103-121
  │     │       ├─ 读取 free/total（CUDA 运行时）
  │     │       ├─ 读取 peak/current（PyTorch 分配器）
  │     │       ├─ 计算 block_bytes = 2 × L × B × H × D × dtype_size
  │     │       ├─ 计算 num_blocks = (budget - overhead) // block_bytes
  │     │       ├─ 分配 kv_cache 张量 [2, L, blocks, B, H, D]
  │     │       └─ 将切片分配给各 Attention 层
  │     └─ 4d: capture_cudagraph()（如果 enforce_eager=False）
  │             └─ 在 KV Cache 分配之后，吃掉一部分显存     model_runner.py:222-257
  │
  ├─ Step 5: Scheduler 初始化
  │     └─ BlockManager(config.num_kvcache_blocks, ...)   scheduler.py:15
  │         └─ 创建 num_blocks 个 Block 对象，全部进入 free 队列
  │
  └─ Step 6: 运行时推理循环
        └─ 每次 step():
              ├─ scheduler.schedule()
              │   ├─ prefill: can_allocate → allocate → 设置 block_table
              │   └─ decode: can_append → may_append（或 preempt）
              ├─ model_runner.run()
              │   ├─ prepare_prefill/decode → 计算 slot_mapping
              │   ├─ set_context() → 全局 Context 写入
              │   ├─ model.forward() → 每层 Attention 读 Context
              │   │   └─ store_kvcache_kernel → 按 slot_mapping 写入缓存
              │   └─ reset_context()
              └─ scheduler.postprocess()
                  ├─ hash_blocks() → 注册已填满 block 的哈希
                  ├─ 更新 num_cached_tokens
                  └─ 如果完成 → deallocate → 释放 block
```

## 5.2 每一层做了什么

| 阶段 | 输入 | 输出 / 副作用 | 是否涉及显存 | 执行频率 |
|---|---|---|---|---|
| Config 校验 | 用户参数 | 校验后的 config 对象 | 否 | 一次 |
| warmup | dummy 序列 | PyTorch peak 统计写入 | 是（临时） | 一次 |
| allocate_kv_cache | peak/current/free/total | kv_cache 张量，config.num_kvcache_blocks | 是（永久） | 一次 |
| capture_cudagraph | kv_cache 已存在 | Graph 对象，graph_vars 缓冲 | 是（永久） | 一次 |
| BlockManager 创建 | num_kvcache_blocks | Block 对象池，free_block_ids | 否（CPU） | 一次 |
| schedule | waiting/running 队列 | scheduled_seqs, block_table 变化 | 否 | 每步 |
| prepare_prefill/decode | seqs + block_table | slot_mapping, Context | 否 | 每步 |
| store_kvcache | K/V + slot_mapping | 写入 kv_cache 张量 | 否（写已分配的） | 每步每层 |
| hash_blocks | completed blocks | hash_to_block_id 更新 | 否（CPU） | 每步 |

## 5.3 哪些逻辑不在主路径

| 函数 / 概念 | 容易误解为 | 实际角色 |
|---|---|---|
| `Attention.__init__` 中的 `torch.tensor([])` | KV Cache 的初始分配 | 仅为占位符，被 `allocate_kv_cache` 覆盖 |
| `Sequence.__getstate__` | 序列的完整序列化 | 仅用于 TP 跨进程通信，decode 时只传 `last_token` |
| `Config.num_kvcache_blocks = -1` | 需要用户手动设置 | 运行时由 `allocate_kv_cache` 自动填充 |
| `config.py:22` 的 `block_size % 256 == 0` 断言 | 性能优化的 hint | 硬约束，违反直接 crash |

---

# 六、显存、性能与通信分析

## 6.1 显存收益范围

| 内容 | 是否受 KV Cache 容量影响 | 原因 |
|---|---|---|
| 模型参数 | ❌ | 固定大小，与 KV Cache 无关 |
| KV Cache 张量 | ✅ | block 数直接决定张量大小 |
| 激活值（prefill） | ❌ | 由 batch size 和 seq_len 决定，公式中已预留 |
| CUDA Graph 缓冲 | ❌ | 在 KV Cache 之后分配，用 utilization 余量 |
| Block 元数据 | ❌ | CPU 内存，每 block 约 9 KB |
| slot_mapping 等 | ❌ | 每步临时创建，极小 |

**真正的显存大头**是 KV Cache 张量本身。以上面 Qwen3-0.6B 的 4907 个 block 为例：

```
kv_cache 总大小 = 4907 × 14 MB ≈ 67 GB
```

这占了 80 GB GPU 的 84%。对于小模型（权重只有 1.2 GB），KV Cache 是 GPU 显存的绝对主力消费者。

## 6.2 通信开销

KV Cache 容量计算本身不涉及通信。但 KV Cache 的使用受 Tensor Parallelism 影响：

| 阶段 | 通信原语 | 频率 | 与 KV Cache 的关系 |
|---|---|---|---|
| allocate_kv_cache | 无 | 每个 rank 独立执行 | 各 rank 分别分配自己的 KV Cache 分片 |
| schedule | 无通信（rank 0 only） | 每步 | Block table 通过 SharedMemory 发送给 worker |
| store_kvcache | 无（本地写入） | 每步每层 | 各 rank 写自己的 KV head 分片 |
| attention forward | NCCL all_reduce（在 o_proj 中） | 每步每层 | 与 KV Cache 无直接关系，是 TP 的固有通信 |

**关键点**：KV Cache 的 block table 是全局统一的（由 rank 0 的 Scheduler 计算），但物理 KV 数据是按 KV head 分片到各 rank 的。所有 rank 使用相同的 slot_mapping 定位，但写入不同的物理缓存。

## 6.3 性能取舍

| 取舍 | 收益 | 代价 |
|---|---|---|
| 经验探测 vs. 静态估算 | 更准确的可用显存估计 | 一次额外的 prefill 前向（约几秒） |
| 整体张量 vs. 逐层分配 | 零碎片化 | 无回退机制，全有或全无 |
| block_size=256 vs. 小 block | 更少的 block 管理开销和哈希表大小 | 平均每序列浪费 127.5 个 token 的显存 |
| CUDA Graph 后置分配 | 简化初始化流程 | KV Cache 可能过度分配，CUDA Graph 时 OOM |
| 保守的 can_allocate | 避免分配后无法 append 的死锁 | 可能不必要地拒绝或延迟可调度的请求 |

---

# 七、配置项、边界条件与坑点

## 7.1 配置如何改变源码路径

| 配置项 | 默认值 | 影响的源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|---|
| `gpu_memory_utilization` | 0.9 | `model_runner.py:113` | 直接乘以 `total` 作为预算上限 | 设到 0.95 以上可能导致 CUDA Graph OOM |
| `kvcache_block_size` | 256 | `model_runner.py:112`, `block_manager.py:28` | 影响每 block 字节数和 block 粒度 | 必须是 256 的倍数；越大碎片越大 |
| `max_num_batched_tokens` | 16384 | `model_runner.py:94-96` | 决定 warmup 的输入规模 → 影响 peak 统计 | 值越大，warmup 的峰值越高，留给 KV Cache 的越少 |
| `max_model_len` | 4096 | `model_runner.py:95`, `capture_cudagraph:227` | 影响 warmup seq_len 和 CUDA Graph block_table 大小 | 被 `hf_config.max_position_embeddings` 截断 |
| `max_num_seqs` | 512 | `capture_cudagraph:226`, `scheduler:58` | CUDA Graph 捕获的最大 batch size | 值越大，CUDA Graph 吃的显存越多 |
| `enforce_eager` | False | `model_runner.py:36-37` | 跳过 CUDA Graph 捕获 | 开启后无 CUDA Graph 开销，KV Cache 可用显存更多 |
| `tensor_parallel_size` | 1 | `model_runner.py:110` | KV head 数除以 TP size | 必须整除 `num_key_value_heads` |

## 7.2 静默失效条件

1. **`num_kvcache_blocks` 用户手动设置无效**：`config.py:18` 的默认值 -1 总是被 `allocate_kv_cache()` 覆盖。即使用户传入 `num_kvcache_blocks=100`，也会被运行时计算结果替换。
2. **block_size=256 时，短序列的 prefix cache 几乎无效**：`can_allocate` 只匹配**完整 block** 的前缀（`range(seq.num_blocks - 1)`，最后一个 block 被跳过）。长度不足 512 token 的序列最多只能匹配 1 个 cached block。
3. **`hash_blocks` 只哈希满块**：`end = (num_cached_tokens + num_scheduled_tokens) // block_size` 使用地板除，未填满的最后一个 block 不会被注册到哈希表。

## 7.3 不兼容组合

- `gpu_memory_utilization = 1.0` + `enforce_eager = False`：CUDA Graph 几乎一定 OOM。
- 超大 `max_model_len` + 小 GPU：warmup 的峰值激活可能吃掉大部分显存，留给 KV Cache 的所剩无几。
- `tensor_parallel_size > 1` + `num_key_value_heads` 不能整除：`qwen3.py:34` 的断言会在模型构造时直接报错，但 `model_runner.py:110` 的整除计算在此之前不会报错。

---

# 八、测试、示例与覆盖缺口

## 8.1 已有示例

| 文件 | 覆盖的行为 |
|---|---|
| `example.py` | 基本推理：2 条 prompt，`enforce_eager=True` |
| `bench.py` | 批量性能测试：256 条随机序列，默认启用 CUDA Graph |

两个示例都使用 `Qwen3-0.6B`，均未显式设置 `gpu_memory_utilization` 或 `kvcache_block_size`。

## 8.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| `gpu_memory_utilization` 过高导致 CUDA Graph OOM | ❌ | 引擎启动时崩溃 |
| 单序列耗尽所有 block 的 assert 崩溃 | ❌ | 运行时 AssertionError |
| `num_kvcache_blocks = 0` 的错误提示 | ❌ | 只有裸 `assert`，无诊断信息 |
| Tensor Parallelism 下的 KV Cache 一致性 | ❌ | 可能出现 rank 间 block 数不一致 |
| block_size 非 256（如 512）的正确性 | ❌ | 理论上支持，但无示例验证 |
| prefix cache 命中后的 KV 正确性 | ❌ | 重用 block 的数据可能与预期不一致 |
| 多轮对话中 preemption → prefix cache 恢复的正确性 | ❌ | 抢占后重新调度的流程未被端到端验证 |
| 显存使用量是否与公式预期一致 | ❌ | 无显存使用的回归测试 |

项目明确声明"**There is no test suite or linter configured**"（`CLAUDE.md`），所以以上所有风险点均无自动化保护。

---

# 九、局限性与已知优化点

## 9.1 硬约束

- **block_size 必须是 256 的倍数**（`config.py:22`）。这极大限制了灵活性——vLLM 的默认 block_size 是 16，而 nano-vllm 的最小值是 256。对于短序列（如 100 token），最后一个 block 浪费了 60% 的空间。
- **Tensor Parallel size 必须 1-8，且整除 `num_key_value_heads`**（`config.py:23`, `qwen3.py:34`）。
- **只支持 Qwen3 架构**。其他模型需要新增 `packed_modules_mapping`。
- **greedy sampling 被禁止**（`SamplingParams` 中 `temperature=0` 会触发断言）。

## 9.2 维护成本

- **CUDA Graph 显存未纳入容量公式**：这是当前最大的维护风险。`gpu_memory_utilization = 0.9` 的 10% 余量能否覆盖 CUDA Graph 取决于模型大小和 `max_num_seqs`——没有代码保证这一点。一个稳健的方案是先捕获 CUDA Graph，再用剩余显存分配 KV Cache。
- **config 可变引用的时序耦合**：`ModelRunner` 写 `config.num_kvcache_blocks`，`Scheduler` 读——如果初始化顺序改变，默默拿到 -1。
- **`free_block_ids.remove(block_id)` 是 O(n)**（`block_manager.py:87`）：当 free block 数达到数千时，前缀缓存命中的 block 回收可能成为调度延迟的来源。
- **SharedMemory 大小硬编码为 1 MB**（`model_runner.py:43`）：大量长 prefill 序列的 pickle 数据可能超限。

## 9.3 性能瓶颈

- **block_size=256 的碎片化**：平均每序列浪费 127.5 个 token 的空间。512 个并发序列的总碎片化浪费可达数 GB。
- **Warmup 的 prefill 成本**：对于大模型（如 72B），warmup 的 16384 token prefill 可能需要数十秒。
- **保守的 `can_allocate`**：在 free 队列中的 cached block 不计入可用量，可能导致不必要的调度延迟或抢占。

## 9.4 已知优化方向

1. **将 CUDA Graph 捕获移到 KV Cache 分配之前**（或测量 CUDA Graph 占用后再分配 KV Cache），消除 OOM 风险。
2. **支持更小的 block_size**（如 16 或 64），减少碎片化，提升短序列场景的显存利用率。
3. **在 `can_allocate` 中计入 free 队列中的 cached block**，减少不必要的调度拒绝。
4. **添加 block 数量的 fallback 逻辑**：如果 `torch.empty` 失败，尝试减少 10% 的 block 数重试。
5. **将 `num_kvcache_blocks` 作为一个显式的可选输入**：允许用户绕过自动计算，手动指定 block 数（当前虽然有该字段但会被覆盖）。
6. **Chunked prefill 的分块 block 分配**：当前 chunked prefill 一次性分配序列的所有 block，即使只处理一小块 token。可以改为按 chunk 增量分配。

---

# 十、小结与展望

nano-vllm 的 KV Cache 容量计算实现可以用几个关键词概括。

**关键词一：经验探测**

不依赖静态公式估算显存，而是通过一次真实的 warmup prefill 前向推理，让 PyTorch 的 `memory_stats` 告诉你实际的峰值激活开销。这种"先跑一遍再看"的策略简单粗暴但有效，避免了因低估 `torch.compile` 缓存、Flash Attention workspace、RoPE 预计算等隐性开销而导致的 OOM。代价是启动时多花几秒。

**关键词二：双统计体系融合**

容量公式同时使用了 CUDA 运行时级别的 `mem_get_info()`（看到 GPU 上所有消费者）和 PyTorch 分配器级别的 `memory_stats()`（只看 PyTorch 自己的分配）。通过 `used - current` 分离出非 PyTorch 开销，通过 `peak - current` 分离出激活值峰值，最终得到一个准确（但脆弱）的可用显存估计。

**关键词三：隐式预算余量**

CUDA Graph 的显存开销没有被公式显式计算——它隐藏在 `gpu_memory_utilization < 1.0` 的余量中。这是一个 pragmatic 的设计：计算 CUDA Graph 的精确开销需要先捕获图（但图需要 KV Cache），形成鸡蛋问题。当前的解决方案是留 10% 余量"祈祷够用"。

**关键词四：一次性整体分配**

KV Cache 是一个 6 维张量，一次 `torch.empty` 分配完毕，然后通过 view 切片分发给各层。零碎片化、零重复分配，但也意味着分配失败时没有 graceful degradation。

**适用场景**：这套实现适合中等规模模型（0.5B-7B）在单卡或少量卡上的在线推理，此时 KV Cache 不是瓶颈，10% 余量足以覆盖 CUDA Graph。

**不适用场景**：大模型（70B+）在显存紧张的情况下，CUDA Graph 的隐性消耗可能打破 10% 余量的假设；超短序列场景下 block_size=256 的碎片化浪费严重。

**与 vLLM 的取舍对比**：vLLM 在容量计算上更精细——它先用一个 dummy 模型跑 profile，再显式预留 CUDA Graph 的空间，且支持 block_size=16。nano-vllm 的简化牺牲了这些精度，换来了不到 1200 行的极简实现。

**后续值得走读的方向**：
- Scheduler 的 chunked prefill 机制如何与 block 分配协调
- prefix caching 的 xxhash 链式哈希在实际工作负载中的命中率
- CUDA Graph 捕获与 KV Cache 交互的显存生命周期
