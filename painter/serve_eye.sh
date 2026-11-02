#!/usr/bin/env bash
# 目と手のモデル Qwen3-VL-32B を vLLM（別 venv /opt/vllm-venv）で起動する。
# - FP8 重み量子化（A6000 は Marlin の weight-only FP8）で 1 枚約 18GB、TP=2
# - gpu-memory-utilization 0.5: 残りの半分を Genesis（GPU0）と Dobby（GPU1）に残す
# - このコンテナでは GPU 間 P2P で NCCL が固まるので無効化。FlashInfer サンプラーは JIT が要るので切る
export VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_P2P_DISABLE=1 PATH=/opt/vllm-venv/bin:$PATH
CUDA_VISIBLE_DEVICES=0,1 exec vllm serve Qwen/Qwen3-VL-32B-Instruct --served-model-name eye \
  --tensor-parallel-size 2 --disable-custom-all-reduce --quantization fp8 --gpu-memory-utilization 0.5 \
  --max-model-len 16384 --limit-mm-per-prompt '{"image":1,"video":0}' --port 8000
