# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight (~1,200 lines) reimplementation of vLLM's core inference engine. It supports Qwen3 models with PagedAttention, prefix caching, tensor parallelism, CUDA graphs, and chunked prefill. The API mirrors vLLM (`LLM`, `SamplingParams`).

## Build & Run

```bash
# Install (requires Python 3.10-3.12, CUDA GPU, flash-attn)
pip install -e .

# Run inference
python example.py

# Benchmark (256 sequences, random lengths)
python bench.py
```

There is no test suite or linter configured.

## Architecture

### Entry Point

`LLM` (in `llm.py`) is just an alias for `LLMEngine`. The engine owns a `Scheduler`, `ModelRunner`, and tokenizer. The main loop calls `scheduler.schedule()` -> `model_runner.call("run", ...)` -> `scheduler.postprocess()` until all sequences finish.

### Engine Layer (`nanovllm/engine/`)

- **LLMEngine** — orchestrates the generate loop: adds requests, steps through prefill/decode phases, collects outputs. Spawns worker processes for tensor parallelism via `torch.multiprocessing`.
- **Scheduler** — two-phase scheduling: prefill (with chunked prefill support, but only for the first sequence in a batch) then decode. Handles preemption by moving sequences back to the waiting queue when KV cache blocks run out.
- **BlockManager** — paged KV cache allocation with prefix caching via xxhash content-based block hashing. Blocks are ref-counted; evicted blocks remain in free list with their hash for potential reuse.
- **Sequence** — tracks token IDs, block table, scheduling state, and status (WAITING/RUNNING/FINISHED). Custom `__getstate__`/`__setstate__` minimize data sent to worker processes during tensor parallelism.

### Model Layer (`nanovllm/models/`)

Only Qwen3 is implemented. `Qwen3ForCausalLM` defines `packed_modules_mapping` which maps HuggingFace checkpoint weight names (q_proj, k_proj, v_proj, gate_proj, up_proj) to fused parameter names (qkv_proj, gate_up_proj) used by the parallel linear layers.

### Custom Layers (`nanovllm/layers/`)

- **Linear classes** — `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear` handle tensor-parallel weight sharding. Each parameter has a `weight_loader` callback used during checkpoint loading.
- **Attention** — uses `flash_attn_varlen_func` for prefill (variable-length sequences) and `flash_attn_with_kvcache` for decode (paged KV cache). A Triton kernel (`store_kvcache_kernel`) writes K/V into the paged cache via slot mapping.
- **RMSNorm** — fused add-residual variant (`add_rms_forward`) avoids an extra kernel launch per layer.
- **Sampler** — Gumbel-max trick via exponential distribution for sampling; `@torch.compile`d.
- **RotaryEmbedding** — precomputed cos/sin cache; `@torch.compile`d.

### Utilities (`nanovllm/utils/`)

- **Context** — global `_CONTEXT` dataclass passed implicitly to attention layers (prefill vs decode metadata, slot mapping, block tables). Set before each model forward, reset after.
- **Loader** — reads safetensors files and routes weights through `packed_modules_mapping` and per-parameter `weight_loader` callbacks.

### ModelRunner Execution Modes

- **Eager mode** (`enforce_eager=True`) — direct model forward for all batch sizes.
- **CUDA graph mode** (default) — captures graphs at batch sizes [1, 2, 4, 8, 16, 32, ...] up to `max_num_seqs`. Prefill always runs eagerly; decode uses captured graphs for batches <= 512.

### Tensor Parallelism

Rank 0 runs in the main process. Ranks 1+ are spawned via `mp.Process` and communicate through shared memory (`SharedMemory` named "nanovllm") with pickle-serialized method calls. NCCL is used for collective ops (`all_reduce` in RowParallelLinear, `gather` in ParallelLMHead).

## Key Constraints

- Only Qwen3 architecture is supported; adding a new model means creating a new file in `models/` with the appropriate `packed_modules_mapping`.
- Greedy sampling (`temperature=0`) is explicitly disallowed by assertion.
- The KV cache block size must be a multiple of 256.
- Tensor parallel size must be 1-8 and must evenly divide both `num_attention_heads` and `num_key_value_heads`.
- The NCCL rendezvous uses hardcoded `tcp://localhost:2333`.
