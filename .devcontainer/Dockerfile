ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
FROM --platform=linux/amd64 ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update \
 && apt-get install -y --no-install-recommends locales \
 && rm -rf /var/lib/apt/lists/* \
 && sed -i '/en_US.UTF-8/s/^# //' /etc/locale.gen \
 && locale-gen
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8


# システムライブラリ
#   git/curl/ca-certificates : 基本ツール（CUDAイメージには curl 等が含まれないため明示導入）
#   libgl1/libgomp1/libx11-6 : open3d の実行に必要
#   libglib2.0-0/libsm6/libxext6/libxrender1 : opencv(cv2) の実行時依存
#   ffmpeg                   : cv2 の動画I/O（VideoCapture 等）
#   libegl1/libglvnd0        : Genesis のヘッドレス描画（EGL）。NVIDIA 側の実装は toolkit が注入
#   fonts-noto-cjk           : 動画の日本語字幕（ffmpeg ASS 焼き込み）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tmux \
    curl \
    ca-certificates \
    libgl1 \
    libgomp1 \
    libx11-6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    libegl1 \
    libglvnd0 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# graphics 機能を有効にし NVIDIA EGL を注入させる / pyrender を EGL で動かす
ENV NVIDIA_DRIVER_CAPABILITIES=all \
    PYOPENGL_PLATFORM=egl

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_PYTHON=3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /workspace

# 依存だけ先にインストールしレイヤをキャッシュ（uv.lock が無い初回はロック生成付きでフォールバック）
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --no-install-project \
    || uv sync --all-extras --no-install-project

# /root/.local/bin: postcreate.sh が導入する Claude Code CLI の配置先
ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}"
