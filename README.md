# ペンで絵を描くロボット画家
## セットアップ（初回）

```bash
uv sync                                   # Genesis など
apt-get install -y gcc fonts-noto-cjk     # vLLM の JIT 用 / 字幕フォント

# vLLM は torch の版が衝突しないよう別 venv に入れる（CUDA 12.9 ビルド）
uv venv /opt/vllm-venv --python 3.12
uv pip install --python /opt/vllm-venv/bin/python "vllm==0.32.0+cu129" ninja \
  --extra-index-url https://wheels.vllm.ai/0.32.0/cu129 --index-strategy unsafe-best-match --torch-backend=cu129
```

## 実行

```bash
# 1. VLM サーバーを起動
bash painter/serve_eye.sh

# 2. お題を与えて描かせる → outputs/snowman/final.mp4
bash painter/run_all.sh "雪だるま" outputs/snowman
```