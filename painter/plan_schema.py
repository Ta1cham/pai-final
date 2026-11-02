"""plan.json の検証（genesis に依存しないので LLM 側からも import できる）。"""

import numpy as np

# 使えるペンの色（LLM が出す色名はこの中に限定する）
PALETTE = {
    "black": (0.08, 0.08, 0.10),
    "red": (0.85, 0.12, 0.12),
    "blue": (0.12, 0.30, 0.85),
    "yellow": (0.95, 0.80, 0.10),
    "green": (0.10, 0.60, 0.25),
    "orange": (0.98, 0.50, 0.08),
    "purple": (0.50, 0.20, 0.70),
    "brown": (0.45, 0.28, 0.12),
}

PEN_LIMITS = dict(max_strokes=40, max_points=400, max_total_length=12.0)  # 1 回あたり。長さはキャンバス座標単位


def _num(x, default):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _pt(p, default=(0.5, 0.5)):
    try:
        return np.array([float(p[0]), float(p[1])])
    except (TypeError, ValueError, IndexError, KeyError):
        return np.array(default, dtype=float)


def _ellipse(c, rx, ry, a0=0.0, a1=360.0, rot=0.0):
    n = max(8, int(abs(a1 - a0) / 12))
    t = np.radians(np.linspace(a0, a1, n + 1))
    pts = np.stack([rx * np.cos(t), ry * np.sin(t)], axis=1)
    r = np.radians(rot)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return pts @ R.T + c


def _hatch(c, w, h, spacing, angle):
    """軸に沿った矩形 (中心 c, 幅 w, 高さ h) の中を、角度 angle の平行線で塗る（1 本のジグザグにつなぐ）。"""
    spacing = max(spacing, 0.015)
    r = np.radians(angle)
    d = np.array([np.cos(r), np.sin(r)])  # 線の向き
    nrm = np.array([-d[1], d[0]])  # 線を並べる向き
    half = np.array([w / 2, h / 2])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    offs = corners @ nrm
    pts, flip = [], False
    for o in np.arange(offs.min() + spacing / 2, offs.max(), spacing):
        # 直線 {o*nrm + t*d} と矩形の交わる区間 [t0, t1] を求める
        t0, t1 = -np.inf, np.inf
        for k in range(2):
            p0 = o * nrm[k]
            if abs(d[k]) < 1e-9:
                if abs(p0) > half[k]:
                    t0, t1 = 1, 0
                continue
            ta, tb = sorted(((-half[k] - p0) / d[k], (half[k] - p0) / d[k]))
            t0, t1 = max(t0, ta), min(t1, tb)
        if t1 <= t0:
            continue
        seg = [o * nrm + t0 * d, o * nrm + t1 * d]
        pts += seg[::-1] if flip else seg
        flip = not flip
    return np.asarray(pts) + c if pts else c[None]


def shape_to_points(s):
    """図形の指定 → 折れ線の点列。LLM は座標列より「円・弧・波線」の方がずっと正確に指定できる。"""
    kind = str(s.get("shape", "polyline")).lower()
    c = _pt(s.get("center"))
    if kind in ("circle", "ellipse", "arc"):
        r = _num(s.get("radius"), 0.08)
        rx, ry = _num(s.get("rx"), r), _num(s.get("ry"), r)
        a0, a1 = (_num(s.get("start_deg"), 0), _num(s.get("end_deg"), 180)) if kind == "arc" else (0, 360)
        return _ellipse(c, rx, ry, a0, a1, _num(s.get("rotation_deg"), 0))
    if kind == "rect":
        w, h = _num(s.get("w"), 0.2), _num(s.get("h"), 0.1)
        return np.array([[-w, -h], [w, -h], [w, h], [-w, h], [-w, -h]]) / 2 + c
    if kind == "hatch":
        return _hatch(c, _num(s.get("w"), 0.2), _num(s.get("h"), 0.1), _num(s.get("spacing"), 0.025),
                      _num(s.get("angle_deg"), 0))
    if kind == "wave":
        a, b = _pt(s.get("start"), (0.1, 0.5)), _pt(s.get("end"), (0.9, 0.5))
        amp, waves = _num(s.get("amplitude"), 0.03), max(0.5, _num(s.get("waves"), 3))
        t = np.linspace(0, 1, int(16 * waves) + 1)[:, None]
        d = b - a
        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        return a + t * d + amp * np.sin(2 * np.pi * waves * t) * nrm
    if kind == "spiral":
        r, turns = _num(s.get("radius"), 0.08), max(0.5, _num(s.get("turns"), 2.5))
        t = np.linspace(0, 1, int(24 * turns) + 1)
        ang = 2 * np.pi * turns * t
        return c + np.stack([r * t * np.cos(ang), r * t * np.sin(ang)], axis=1)
    if kind == "dot":
        return c[None]
    return [_pt(p) for p in s.get("points", [])]  # line / polyline


def validate_plan(plan, colors=tuple(PALETTE), limits=None):
    """LLM が変な値を返してもシミュレーションが壊れないよう、計画を切り詰めて正規化する。

    各ストロークは points（折れ線）か shape（図形）で指定でき、ここで points に展開する。
    """
    limits = limits or PEN_LIMITS
    strokes, total = [], 0.0
    for s in plan.get("strokes", [])[: limits["max_strokes"]]:
        if not isinstance(s, dict):
            continue
        if "shape" in s:
            s = {**s, "points": [list(p) for p in shape_to_points(s)]}
        color = str(s.get("color", "black")).lower()
        if color not in colors:
            color = "black"
        pts = []
        for p in s.get("points", [])[: limits["max_points"]]:
            try:
                pts.append([float(np.clip(p[0], 0, 1)), float(np.clip(p[1], 0, 1))])
            except (TypeError, ValueError, IndexError):
                continue
        if not pts:
            continue
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0
        if total + length > limits["max_total_length"]:
            continue  # 長すぎるストロークは捨てるが、後ろの（別の色の）ストロークは残す
        total += length
        strokes.append(dict(
            label=str(s.get("label", ""))[:40],
            color=color,
            points=pts,
            thickness=float(np.clip(s.get("thickness", 0.5), 0, 1)),
            speed=float(np.clip(s.get("speed", 0.5), 0, 1)),
        ))
    return {**plan, "strokes": strokes}


def raw_length(plan):
    """切り詰め前の筆跡の総延長（LLM へのフィードバック用）。"""
    total = 0.0
    for s in plan.get("strokes", []):
        try:
            pts = shape_to_points(s) if "shape" in s else s["points"]
            pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        except (KeyError, TypeError, ValueError):
            continue
        if len(pts) > 1:
            total += float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return total
