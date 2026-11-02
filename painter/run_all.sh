#!/usr/bin/env bash
# お題 1 つで「描く → 自己批評 → 加筆」を繰り返し、動画まで作る（ペン版）。
#   bash painter/run_all.sh "失恋した夜の海" outputs/pen_sea [最大セッション数]
# 前提: Qwen3-VL-32B を vLLM で起動しておく（GPU0,1 を半分ずつ使う）: bash painter/serve_eye.sh
# 画家・審査員・製図係はすべて vLLM 上の同じモデル。描画（Genesis）は SIM_GPU（既定 0）
# 審査員の点数が TARGET（既定 8）以上になるか、最大セッション数に達したら終了
set -euo pipefail
cd "$(dirname "$0")/.."
THEME="$1"
OUT="${2:-outputs/pen_$(date +%m%d_%H%M%S)}"
ROUNDS="${3:-5}"
mkdir -p "$OUT"

uv run python painter/painter_llm.py "$THEME" --out "$OUT" --rounds "$ROUNDS" --target "${TARGET:-8}" \
  --sim-gpu "${SIM_GPU:-0}" 2>&1 | tee "$OUT/session.log"
uv run python painter/make_video.py "$OUT"
