"""キャンバスの定義と Genesis の初期化。

座標系:
- Franka は原点に置く。キャンバスはロボットの正面（+x 方向）の床の上にある。
- キャンバス座標 (u, v) は 0〜1。真上画像で u は左→右、v は下→上に対応する。
"""

import os

# pyrender に EGL（GPU ヘッドレス描画）を使わせる。genesis の import 前に設定する必要がある
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402

# キャンバス（床の上の白い板）。ワールド座標 [m]
CANVAS_X = (0.35, 0.75)  # ロボットから見て手前→奥
CANVAS_Y = (-0.2, 0.2)
CANVAS_CENTER = (sum(CANVAS_X) / 2, sum(CANVAS_Y) / 2)


def canvas_to_world(u, v, z):
    """キャンバス座標 (u, v)∈[0,1]² をワールド座標に変換する。

    真上カメラは up=+x で見下ろすため、画像の右方向はワールドの -y、上方向は +x になる。
    """
    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    x = CANVAS_X[0] + v * (CANVAS_X[1] - CANVAS_X[0])
    y = CANVAS_Y[1] - u * (CANVAS_Y[1] - CANVAS_Y[0])
    return np.array([x, y, z])


def init_genesis():
    gs.init(backend=gs.gpu, logging_level="warning")
