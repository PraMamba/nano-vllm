# nano-vllm 学习路线图

> 面向 LLM 推理优化方向的博士研究生

## 前置知识要求

在深入学习 nano-vllm 之前，建议先掌握以下基础知识：

### 必备知识
- **PyTorch 基础**: `nn.Module`、`torch.Tensor`、`autograd` 机制
- **Transformer 架构**: Self-Attention、Multi-Head Attention、LayerNorm、FFN
- **GPU 编程概念**: CUDA kernel、显存管理、计算与访存的关系
- **Python 多进程**: `multiprocessing`、进程间通信

### 推荐补充
- **分布式训练基础**: `torch.distributed`、NCCL
- **内存管理**: 虚拟内存、分页机制（OS 层面）

---

## 学习阶段

### 第一阶段：宏观理解（1-2天）

**目标**: 理解整体架构和数据流

**学习顺序**:
1. 阅读 `README.md` 了解项目定位
2. 运行 `example.py` 观察输出
3. 阅读 `nanovllm/__init__.py` → `llm.py` → `engine/llm_engine.py`
4. 绘制自己的架构图

**核心问题**:
- [ ] LLM 推理为什么需要分 Prefill 和 Decode 两个阶段？
- [ ] 什么是 Continuous Batching？它解决了什么问题？

**参考文档**: `01-architecture-overview.md`

---

### 第二阶段：调度与序列管理（2-3天）

**目标**: 理解请求如何被调度和管理

**学习顺序**:
1. `engine/sequence.py` - 序列数据结构
2. `engine/scheduler.py` - 调度逻辑
3. `sampling_params.py` - 采样参数

**核心问题**:
- [ ] Sequence 对象包含哪些关键状态？为什么需要这些状态？
- [ ] Scheduler 如何决定 Prefill 优先还是 Decode 优先？
- [ ] 什么情况下会发生 Preemption（抢占）？如何处理？

**参考文档**: `02-engine-analysis.md`

---

### 第三阶段：内存管理 - PagedAttention（3-4天）

**目标**: 深入理解 KV Cache 的分页管理

**学习顺序**:
1. `engine/block_manager.py` - 核心实现
2. 理解 Block、BlockTable、SlotMapping 的关系
3. 理解 Prefix Caching 的哈希机制

**核心问题**:
- [ ] 为什么传统方式会产生内存碎片？PagedAttention 如何解决？
- [ ] Block Size 的选择对性能有什么影响？
- [ ] Prefix Caching 的哈希是如何计算的？什么情况下会命中缓存？
- [ ] `slot_mapping` 是什么？它如何告诉 Attention 层存储 KV 的位置？

**参考文档**: `03-memory-management.md`

---

### 第四阶段：模型执行与优化（3-4天）

**目标**: 理解模型如何高效执行

**学习顺序**:
1. `engine/model_runner.py` - 执行引擎
2. `utils/context.py` - 全局上下文
3. 理解 CUDA Graph 的捕获和重放
4. 理解 Prefill vs Decode 的不同输入准备

**核心问题**:
- [ ] `prepare_prefill` 和 `prepare_decode` 的输出有什么不同？
- [ ] CUDA Graph 为什么能加速？它有什么限制？
- [ ] 为什么 Decode 阶段使用 CUDA Graph，而 Prefill 不用？
- [ ] `enforce_eager=True` 什么时候需要使用？

**参考文档**: `04-model-execution.md`

---

### 第五阶段：注意力机制与 FlashAttention（2-3天）

**目标**: 理解高效注意力计算

**学习顺序**:
1. `layers/attention.py` - Attention 实现
2. Triton kernel `store_kvcache_kernel`
3. FlashAttention 的调用方式

**核心问题**:
- [ ] FlashAttention 相比标准 Attention 优化了什么？
- [ ] `flash_attn_varlen_func` vs `flash_attn_with_kvcache` 的区别是什么？
- [ ] Triton kernel 如何将 KV 写入分页缓存？

**参考文档**: `05-attention-mechanism.md`

---

### 第六阶段：张量并行（2-3天）

**目标**: 理解多 GPU 推理

**学习顺序**:
1. `layers/linear.py` - 并行线性层
2. `layers/embed_head.py` - 词嵌入并行
3. `engine/model_runner.py` 中的多进程协调

**核心问题**:
- [ ] ColumnParallelLinear vs RowParallelLinear 的切分方式有什么不同？
- [ ] All-Reduce 和 All-Gather 分别在什么时候使用？
- [ ] SharedMemory 是如何实现进程间通信的？

**参考文档**: `06-tensor-parallelism.md`

---

### 第七阶段：模型结构（1-2天）

**目标**: 理解 Qwen3 模型实现

**学习顺序**:
1. `models/qwen3.py` - 模型结构
2. `layers/rotary_embedding.py` - RoPE
3. `layers/layernorm.py` - RMSNorm
4. `utils/loader.py` - 权重加载

**核心问题**:
- [ ] `packed_modules_mapping` 是什么？为什么需要打包 QKV？
- [ ] RoPE 的位置编码是如何应用的？
- [ ] 模型权重是如何从 Hugging Face 格式加载并分片的？

---

## 进阶探索方向

完成基础学习后，可以探索以下方向：

### 性能优化
- 实现 Speculative Decoding
- 添加 Top-K / Top-P 采样
- 优化 Block Size 选择策略

### 功能扩展
- 支持更多模型架构（LLaMA、Mistral）
- 实现 Online Serving（HTTP API）
- 添加 Beam Search

### 研究方向
- KV Cache 压缩（Quantization、Eviction）
- 更高效的调度算法
- 长上下文优化

---

## 学习检验清单

完成学习后，尝试回答以下问题：

### 基础理解
- [ ] 画出 nano-vllm 的整体架构图
- [ ] 描述一个 prompt 从输入到输出的完整生命周期
- [ ] 解释 Prefill 和 Decode 阶段的计算特点

### 核心机制
- [ ] 解释 PagedAttention 的工作原理
- [ ] 说明 Prefix Caching 如何识别和复用缓存
- [ ] 描述 CUDA Graph 的优化原理和限制

### 代码层面
- [ ] 能够修改 Scheduler 实现不同的调度策略
- [ ] 能够添加新的采样方法（如 Top-K）
- [ ] 能够支持新的模型架构

---

## 推荐阅读资源

### 论文
1. **vLLM**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
2. **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
3. **Orca**: "ORCA: A Distributed Serving System for Transformer-Based Generative Models"

### 博客
1. vLLM 官方博客
2. FlashAttention 作者 Tri Dao 的博客
3. Hugging Face 的 LLM 推理优化系列

### 代码
1. vLLM 官方仓库（对比学习）
2. llama.cpp（CPU 推理优化）
3. TensorRT-LLM（NVIDIA 官方优化）
