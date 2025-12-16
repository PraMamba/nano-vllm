# nano-vllm 架构概览

## 1. 项目结构（基于源码）

```
nanovllm/
├── __init__.py              # 导出 LLM, SamplingParams
├── llm.py                   # LLM 类（LLMEngine 的别名）
├── config.py                # 配置数据类
├── sampling_params.py       # 采样参数
├── engine/
│   ├── llm_engine.py        # 核心引擎，协调所有组件
│   ├── scheduler.py         # 调度器，管理 waiting/running 队列
│   ├── block_manager.py     # KV Cache 块管理（PagedAttention 核心）
│   ├── sequence.py          # 序列状态管理
│   └── model_runner.py      # 模型执行器（CUDA Graph、张量并行）
├── models/
│   └── qwen3.py             # Qwen2/Qwen3 模型实现
├── layers/
│   ├── attention.py         # FlashAttention + Triton KV 存储
│   ├── linear.py            # 张量并行线性层
│   ├── embed_head.py        # 词嵌入和 LM Head
│   ├── layernorm.py         # RMSNorm
│   ├── rotary_embedding.py  # RoPE 位置编码
│   ├── activation.py        # SiLU 激活
│   └── sampler.py           # 采样器（温度采样）
└── utils/
    ├── context.py           # 全局上下文（传递注意力元数据）
    └── loader.py            # Safetensors 权重加载
```

---

## 2. 核心架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户接口层                                  │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  LLM.generate(prompts, sampling_params)                         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          LLMEngine                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │  Tokenizer   │  │  Scheduler   │  │     ModelRunner(s)       │   │
│  │ (HuggingFace)│  │              │  │                          │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘   │
│         │                 │                      │                   │
│         │                 │  ┌───────────────────┘                   │
│         │                 │  │                                       │
│         │                 ▼  ▼                                       │
│         │          ┌──────────────┐                                  │
│         │          │ BlockManager │  ← PagedAttention 内存管理        │
│         │          └──────────────┘                                  │
└─────────┼───────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Model Layer                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Qwen3ForCausalLM                                               │ │
│  │  ├── embed_tokens (VocabParallelEmbedding)                      │ │
│  │  ├── layers[] (Qwen3DecoderLayer)                               │ │
│  │  │   ├── self_attn (Qwen3Attention → Attention)                 │ │
│  │  │   │   └── FlashAttention + KV Cache (Triton kernel)          │ │
│  │  │   └── mlp (Qwen3MLP)                                         │ │
│  │  ├── norm (RMSNorm)                                             │ │
│  │  └── lm_head (ParallelLMHead)                                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────┐                                                  │
│  │    Sampler     │  ← torch.compile 优化的采样                       │
│  └────────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 数据流详解

### 3.1 初始化流程

```python
# 用户调用
llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=2)

# 内部流程:
# 1. Config 初始化
config = Config(model_path, ...)
config.hf_config = AutoConfig.from_pretrained(model_path)

# 2. 多进程创建 (张量并行)
for i in range(1, tensor_parallel_size):
    process = Process(target=ModelRunner, args=(config, i, event))
    process.start()

# 3. 主进程 ModelRunner (rank=0)
model_runner = ModelRunner(config, rank=0, events)
# 内部:
#   - dist.init_process_group("nccl", ...)
#   - 加载模型权重
#   - 预热模型 (warmup)
#   - 分配 KV Cache
#   - 捕获 CUDA Graph (如果 enforce_eager=False)

# 4. Scheduler 初始化
scheduler = Scheduler(config)
# 内部:
#   - 创建 BlockManager
#   - 初始化 waiting/running 队列
```

### 3.2 推理流程

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 请求入队                                                  │
├─────────────────────────────────────────────────────────────────┤
│ prompts = ["Hello", "Hi there"]                                  │
│     ↓                                                            │
│ tokenizer.encode() → [[123, 45], [67, 89, 12]]                  │
│     ↓                                                            │
│ Sequence 对象创建，加入 scheduler.waiting 队列                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 调度 (scheduler.schedule)                                │
├─────────────────────────────────────────────────────────────────┤
│ 优先级: Prefill > Decode                                         │
│                                                                  │
│ Prefill 调度:                                                    │
│   - 从 waiting 队列取序列                                         │
│   - 检查 BlockManager 是否有足够空间                               │
│   - 分配 KV Cache 块                                             │
│   - 移动到 running 队列                                          │
│   - 返回 (seqs, is_prefill=True)                                 │
│                                                                  │
│ Decode 调度:                                                     │
│   - 从 running 队列取序列                                         │
│   - 检查是否需要新块（每 256 token 一个块）                          │
│   - 如果空间不足，执行 Preemption                                  │
│   - 返回 (seqs, is_prefill=False)                                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: 模型执行 (model_runner.run)                              │
├─────────────────────────────────────────────────────────────────┤
│ Prefill 准备:                                                    │
│   input_ids = [123, 45, 67, 89, 12]  # 拼接所有序列               │
│   positions = [0, 1, 0, 1, 2]        # 每个 token 的位置          │
│   cu_seqlens_q/k = [0, 2, 5]         # 序列边界                   │
│   slot_mapping = [0, 1, 256, 257, 258]  # KV 存储位置             │
│                                                                  │
│ Decode 准备:                                                     │
│   input_ids = [last_token_1, last_token_2]  # 每个序列最后一个     │
│   positions = [seq1_len, seq2_len]                               │
│   context_lens = [seq1_len+1, seq2_len+1]                        │
│   block_tables = [[0, 1], [2, 3, ...]]     # 每个序列的块表        │
│                                                                  │
│ 模型前向传播:                                                     │
│   hidden = model.embed_tokens(input_ids)                         │
│   for layer in layers:                                           │
│       hidden = layer(positions, hidden, residual)                │
│       # Attention 内部会读写 KV Cache                             │
│   logits = model.lm_head(hidden)                                 │
│                                                                  │
│ 采样:                                                            │
│   next_tokens = sampler(logits, temperatures)                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: 后处理 (scheduler.postprocess)                           │
├─────────────────────────────────────────────────────────────────┤
│ for seq, token_id in zip(seqs, token_ids):                       │
│     seq.append_token(token_id)                                   │
│     if token_id == EOS or len(seq) == max_tokens:                │
│         seq.status = FINISHED                                    │
│         block_manager.deallocate(seq)  # 释放 KV Cache            │
│         running.remove(seq)                                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: 循环直到完成                                              │
├─────────────────────────────────────────────────────────────────┤
│ while not scheduler.is_finished():                               │
│     outputs, num_tokens = step()                                 │
│                                                                  │
│ 返回结果:                                                         │
│ [{"text": "...", "token_ids": [...]}]                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 关键概念解释

### 4.1 Prefill vs Decode

| 特性 | Prefill | Decode |
|------|---------|--------|
| 输入 | 完整 prompt | 上一步生成的 token |
| 计算模式 | Compute-bound | Memory-bound |
| 并行性 | 高（多 token 并行计算） | 低（每个序列一个 token） |
| KV Cache | 写入整个 prompt 的 KV | 追加一个 token 的 KV |
| CUDA Graph | 不适用（序列长度变化） | 适用（固定计算模式） |

### 4.2 Continuous Batching

传统 Batching:
```
Batch 1: [seq1, seq2, seq3] → 等待所有完成 → Batch 2: [seq4, seq5, seq6]
```

Continuous Batching:
```
Step 1: [seq1, seq2, seq3]
Step 2: [seq1, seq2, seq3]  # seq1 完成
Step 3: [seq2, seq3, seq4]  # seq4 立即加入
Step 4: [seq2, seq3, seq4]  # seq3 完成
Step 5: [seq2, seq4, seq5]  # seq5 立即加入
...
```

优势：GPU 利用率更高，吞吐量更大

### 4.3 KV Cache 分页

```
传统方式:
┌──────────────────────────────────────┐
│ seq1: [KV_0, KV_1, KV_2, ________]   │  ← 预分配 max_len，浪费空间
│ seq2: [KV_0, KV_1, ________________]  │
│ seq3: [KV_0, KV_1, KV_2, KV_3, ____]  │
└──────────────────────────────────────┘

PagedAttention:
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Block 0 │ │ Block 1 │ │ Block 2 │ │ Block 3 │  ← 固定大小块池
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │           │           │
     ▼           ▼           ▼           ▼
seq1.block_table = [0, 2]     # 使用 Block 0 和 2
seq2.block_table = [1]        # 使用 Block 1
seq3.block_table = [3, ...]   # 使用 Block 3 和后续分配的块
```

---

## 5. 配置参数详解

基于 `nanovllm/config.py`:

```python
@dataclass
class Config:
    model: str                          # 模型路径（必须是本地目录）
    max_num_batched_tokens: int = 16384 # 单批次最大 token 数
    max_num_seqs: int = 512             # 最大并发序列数
    max_model_len: int = 4096           # 最大序列长度
    gpu_memory_utilization: float = 0.9 # KV Cache 使用的 GPU 显存比例
    tensor_parallel_size: int = 1       # 张量并行 GPU 数量
    enforce_eager: bool = False         # 禁用 CUDA Graph
    kvcache_block_size: int = 256       # 每个块的 token 数（必须是 256 的倍数）
```

### 参数影响分析

| 参数 | 增大影响 | 减小影响 |
|------|---------|---------|
| `max_num_batched_tokens` | 更高吞吐，更大显存占用 | 更低延迟，更小吞吐 |
| `max_num_seqs` | 更高并发，更大调度开销 | 更低并发 |
| `gpu_memory_utilization` | 更多 KV Cache 空间 | 更少 OOM 风险 |
| `kvcache_block_size` | 更少块管理开销，更大浪费 | 更细粒度管理，更多开销 |

---

## 6. 疑惑点与核心知识点对照

| 疑惑问题 | 核心知识点 | 对应代码位置 |
|---------|-----------|-------------|
| 为什么 Prefill 不能用 CUDA Graph？ | CUDA Graph 需要固定形状 | `model_runner.py:191-206` |
| Scheduler 如何判断空间是否足够？ | BlockManager 的块分配算法 | `scheduler.py:31`, `block_manager.py:56-57` |
| 多 GPU 间如何同步？ | NCCL All-Reduce/All-Gather | `linear.py`, `embed_head.py` |
| Prefix Caching 如何工作？ | xxhash 哈希 + 块复用 | `block_manager.py:36-41, 59-82` |
| 为什么温度不能为 0？ | Gumbel-max 采样技巧 | `sampler.py:10-15` |
