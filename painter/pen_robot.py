"""Franka がペンで絵を描く（1 回分）。前の回までのインクを再現してから、この回のストロークを足す。

    uv run python painter/pen_robot.py outputs/run1 --round 1

入力:  <run>/plan_r<k>.json（この回に描くストローク）, <run>/traces.json（前の回までの実際の線。無ければ白紙）
出力:  <run>/painting_r<k>.mp4, <run>/artwork_r<k>.png（キャンバスだけを切り出した真上画像）, traces.json を更新

インクは物理計算せず、ペン先がキャンバスに触れている間の「実際の」ペン先軌跡を
デバッグ描画（カメラ debug=True のときだけ映る円柱メッシュ）で残す。
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from plan_schema import PALETTE, PEN_LIMITS, validate_plan  # noqa: E402

import genesis as gs  # noqa: E402
import trimesh  # noqa: E402

DT = 0.01  # 液体が無いので粗くてよい
PEN_LEN = 0.13  # hand リンク → ペン先 [m]（指先 ≈ 0.10 から 3cm 突き出す）
DOWN_Z = 0.0  # ペンを下ろしたときのペン先の目標高さ（キャンバス上面 z=0.001 をわずかに押す）
UP_Z = 0.05
INK_CONTACT = 0.006  # ペン先がこの高さ未満ならインクが出る
TOOL_SPEED = (0.08, 0.30)  # speed 0〜1 → 描画中の移動速度 [m/s]
TRAVEL_SPEED = 0.45
RADIUS = (0.0015, 0.0055)  # thickness 0〜1 → 線の半径 [m]
HAND_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
DWELL_STEPS = 25  # ストローク終点で待つステップ数（0.25 秒）
CHUNK_STEPS = 6  # 描画中の線はこのステップ数ごとに小分けで描き足す
TOP_FOV = 20.0
TOP_HEIGHT = 1.3


def lerp(a, b, t):
    return a + (b - a) * t


def rgba255(color):
    return [int(255 * c) for c in PALETTE[color]] + [255]


def stroke_mesh(points, radius, color, z):
    """折れ線 → 円柱と継ぎ目の球をつないだ 1 つのメッシュ。"""
    pts = [np.array([p[0], p[1], z]) for p in points]
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) > 0.0015:  # 細かすぎる点は間引く（頂点数を抑える）
            keep.append(p)
    parts = []
    for a, b in zip(keep[:-1], keep[1:]):
        parts.append(trimesh.creation.cylinder(radius=radius, segment=np.stack([a, b]), sections=10))
    for p in keep[:: max(1, len(keep) // 40)] + [keep[-1]]:
        s = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        s.apply_translation(p)
        parts.append(s)
    m = trimesh.util.concatenate(parts)
    m.visual.vertex_colors = np.tile(rgba255(color), (len(m.vertices), 1))
    return m


def pen_mesh(color):
    """手先座標系のペン（軸は白、先端だけペンの色）。原点 = hand リンク。"""
    # hand リンクの座標系では指の伸びる向きが +z
    body = trimesh.creation.cylinder(radius=0.007, segment=[[0, 0, 0.03], [0, 0, PEN_LEN - 0.015]], sections=16)
    body.visual.vertex_colors = np.tile([235, 235, 235, 255], (len(body.vertices), 1))
    tip = trimesh.creation.cone(radius=0.007, height=0.015, sections=16)  # 底面 z=0、頂点 z=+0.015
    tip.apply_translation([0, 0, PEN_LEN - 0.015])
    tip.visual.vertex_colors = np.tile(rgba255(color), (len(tip.vertices), 1))
    return trimesh.util.concatenate([body, tip])


def build_scene(res=(1280, 720)):
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    # キャンバスの上面は z=+0.001（床と同じ高さだと z-fighting でちらつく）
    scene.add_entity(
        gs.morphs.Box(pos=(*C.CANVAS_CENTER, -0.004), size=(C.CANVAS_X[1] - C.CANVAS_X[0] + 0.03,
                                                             C.CANVAS_Y[1] - C.CANVAS_Y[0] + 0.03, 0.01),
                      fixed=True, collision=False),
        surface=gs.surfaces.Default(color=(0.98, 0.97, 0.94, 1.0)),
    )
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    cam = scene.add_camera(res=tuple(res), pos=(1.15, -0.85, 0.95), lookat=(0.5, 0.0, 0.05), fov=40, debug=True)
    top = scene.add_camera(res=(900, 900), pos=(*C.CANVAS_CENTER, TOP_HEIGHT), lookat=(*C.CANVAS_CENTER, 0.0),
                           up=(1.0, 0.0, 0.0), fov=TOP_FOV, debug=True)
    scene.build()
    return scene, franka, cam, top


def save_canvas_image(top, path, size=768):
    """真上画像からキャンバス部分だけを切り出す → 画像座標がそのままキャンバス座標 (u, v) になる。"""
    from PIL import Image

    rgb = np.asarray(top.render(rgb=True)[0])
    n = rgb.shape[0]
    half_view = TOP_HEIGHT * math.tan(math.radians(TOP_FOV / 2))
    half_canvas = (C.CANVAS_X[1] - C.CANVAS_X[0]) / 2
    m = int(round(n * (half_view - half_canvas) / (2 * half_view)))
    Image.fromarray(rgb[m: n - m, m: n - m]).resize((size, size), Image.LANCZOS).save(path)


def build_schedule(plan):
    """毎ステップの (ペン先の目標位置, ストローク番号 or None)。"""
    sched, cur = [], None

    def move(target, speed, stroke=None):
        nonlocal cur
        if cur is None:
            cur = target
            return
        n = max(1, int(math.ceil(np.linalg.norm(target - cur) / (speed * DT))))
        for k in range(1, n + 1):
            sched.append((lerp(cur, target, k / n), stroke))
        cur = target

    for i, s in enumerate(plan["strokes"]):
        pts = [C.canvas_to_world(u, v, DOWN_Z) for u, v in s["points"]]
        up = np.array([0, 0, UP_Z - DOWN_Z])
        move(pts[0] + up, TRAVEL_SPEED)
        move(pts[0], 0.15, i)
        if len(pts) == 1:  # 点：少し押し付けて小さな円を描く
            for a in np.linspace(0, 2 * math.pi, 12):
                move(pts[0] + 0.004 * np.array([math.cos(a), math.sin(a), 0]), 0.05, i)
        for p in pts[1:]:
            move(p, lerp(*TOOL_SPEED, s["speed"]), i)
        # 腕は目標から遅れて追従するので、終点で少し待ってから持ち上げる（円の閉じ際に隙間が空くのを防ぐ）
        sched += [(cur, i)] * DWELL_STEPS
        move(cur + up, 0.15)
    return sched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--round", type=int, default=1)
    args = p.parse_args()
    d, k = args.run_dir, args.round

    plan = validate_plan(json.loads((d / f"plan_r{k}.json").read_text()), limits=PEN_LIMITS)
    (d / f"plan_r{k}_validated.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    traces_path = d / "traces.json"
    traces = json.loads(traces_path.read_text()) if traces_path.exists() and k > 1 else []
    sched = build_schedule(plan)
    print(f"round {k}: strokes={len(plan['strokes'])} steps={len(sched)} (~{len(sched) * DT:.1f}s), "
          f"history={len(traces)} strokes", flush=True)

    C.init_genesis()
    scene, franka, cam, top = build_scene()
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    hand = franka.get_link("hand")

    def z_of(n):  # 後から描いた線ほどわずかに上に置き、重なりを正しく見せる
        return 0.0016 + 0.00002 * n

    for n, t in enumerate(traces):  # 前の回までの線を再現
        scene.draw_debug_mesh(stroke_mesh(t["points"], t["radius"], t["color"], z_of(n)))

    def ik(tip):
        q = franka.inverse_kinematics(link=hand, pos=tip + np.array([0, 0, PEN_LEN]), quat=HAND_QUAT)
        q[-2:] = 0.01  # ペンを握っている体
        return q

    pen_meshes = {c: pen_mesh(c) for c in PALETTE}
    pen_obj = None

    def draw_pen(color):
        nonlocal pen_obj
        if pen_obj is not None:
            scene.clear_debug_object(pen_obj)
        T = np.eye(4)
        T[:3, 3] = hand.get_pos().cpu().numpy()
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # quat (0,1,0,0) = x 軸まわり 180°（真下向き）
        pen_obj = scene.draw_debug_mesh(pen_meshes[color], T=T)

    franka.set_dofs_position(ik(sched[0][0] if sched else C.canvas_to_world(0.5, 0.5, UP_Z)))
    cam.start_recording(save_to_filename=str(d / f"painting_r{k}.mp4"), fps=50)

    live, chunk_objs, cur_stroke, max_err = [], [], None, 0.0
    color = plan["strokes"][0]["color"] if plan["strokes"] else "black"

    def finish_stroke():
        nonlocal live, chunk_objs, cur_stroke
        for o in chunk_objs:
            scene.clear_debug_object(o)
        if len(live) >= 2:
            s = plan["strokes"][cur_stroke]
            t = dict(color=s["color"], radius=lerp(*RADIUS, s["thickness"]), points=live, round=k, label=s.get("label", ""))
            scene.draw_debug_mesh(stroke_mesh(t["points"], t["radius"], t["color"], z_of(len(traces))))
            traces.append(t)
        live, chunk_objs, cur_stroke = [], [], None

    for i, (target, si) in enumerate(sched):
        q = ik(target)
        franka.control_dofs_position(q)
        if si is not None:
            color = plan["strokes"][si]["color"]
        if i % 2 == 0:
            draw_pen(color)
        scene.step()
        tip = hand.get_pos().cpu().numpy() - np.array([0, 0, PEN_LEN])
        touching = si is not None and tip[2] < INK_CONTACT
        if touching:
            if cur_stroke is not None and cur_stroke != si:
                finish_stroke()
            cur_stroke = si
            live.append([float(tip[0]), float(tip[1])])
            max_err = max(max_err, float(np.linalg.norm(tip[:2] - target[:2])))
            if len(live) % CHUNK_STEPS == 0:  # 描いている最中の線を小分けで見せる
                s = plan["strokes"][si]
                chunk = live[-CHUNK_STEPS - 1:]
                chunk_objs.append(scene.draw_debug_mesh(
                    stroke_mesh(chunk, lerp(*RADIUS, s["thickness"]), s["color"], z_of(len(traces)))))
        elif cur_stroke is not None:
            finish_stroke()
        if (i + 1) % 1000 == 0:
            print(f"step {i + 1}/{len(sched)}", flush=True)
    if cur_stroke is not None:
        finish_stroke()

    # 腕をキャンバスの外へ退避
    park_q = franka.inverse_kinematics(link=hand, pos=np.array([0.12, 0.0, 0.55]), quat=HAND_QUAT)
    for i in range(80):
        franka.control_dofs_position(park_q)
        if i % 2 == 0:
            draw_pen(color)
        scene.step()
    cam.stop_recording()
    save_canvas_image(top, d / f"artwork_r{k}.png")
    traces_path.write_text(json.dumps(traces))
    print(f"max tracking error: {max_err * 1000:.1f} mm, total strokes on canvas: {len(traces)}", flush=True)
    print(f"saved: {d / f'artwork_r{k}.png'}")


if __name__ == "__main__":
    main()
