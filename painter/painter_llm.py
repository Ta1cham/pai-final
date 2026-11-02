"""性格付きの画家 AI が、ペンで「描く → 審査 → 自己批評 → 加筆」を繰り返す。

全部 vLLM 上の Qwen3-VL-32B 1 つで、役割（システムプロンプト）を切り替えて使う:
- 画家（ペルソナ, 日本語）: 第一声・タイトル・構想・自分の絵を見ての自己批評・次に足すもの
- 審査員（中立）: 加筆前後の画像を並べて採点。悪化した回は取り消し、目標点に届いたら終了
- 製図係: 構想 → 図形プリミティブの JSON（今の絵と、描いた要素の一覧を見て配置）
描画は回ごとに pen_robot.py を別プロセス（--sim-gpu）で実行する。

前提: bash painter/serve_eye.sh で vLLM サーバーを起動しておく
    uv run python painter/painter_llm.py "雪だるま" --out outputs/pen_snowman2
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plan_schema import PALETTE, PEN_LIMITS, raw_length, validate_plan  # noqa: E402

API_URL = os.environ.get("EYE_URL", "http://localhost:8000/v1")
COLORS = list(PALETTE)

# 画家の人柄。演技はさせず、普通の人として話させる
PERSONA = f"""あなたはロボットアームを操作して絵を描く画家です。
フェルトペンだけで描きます。ペンの色は {", ".join(COLORS)} のみ。白や消しゴムはありません。
何回かに分けて描き、毎回描き終えたら自分の絵を見直して、足りないものを描き足します。
ふつうの丁寧な口調で、簡潔に、日本語で答えてください。"""

JUDGE = """You are a strict, neutral art judge. You score simple pen drawings made by a robot on a white canvas.
Score 1-10 for how clearly and cleanly the drawing depicts the given theme:
- 8+ = a viewer immediately recognizes the theme; clean, uncluttered, nothing important missing.
- 5-7 = recognizable but missing parts, messy, or with confusing extra marks.
- 1-4 = hard to recognize.
Messy overdrawing, duplicated outlines and random marks must LOWER the score."""

STROKE_RULES = f"""Canvas coordinates (math convention, NOT image convention): u = left->right, v = bottom->top, both in [0, 1].
(0, 0) is the BOTTOM-LEFT corner and (1, 1) is the TOP-RIGHT corner. "Top" means v near 0.9, "bottom" v near 0.1.
Available pen colors: {COLORS}. Nothing else exists (no white).
Output ONLY a JSON object {{"strokes": [...]}}. Every stroke has "label" (what it depicts), "color",
"thickness" (0 = fine liner .. 1 = fat marker), "speed" (0 = careful .. 1 = sketchy) and ONE shape:
  {{"shape": "line", "points": [[u, v], [u, v]]}}
  {{"shape": "polyline", "points": [[u, v], ...]}}            (2-25 points, for outlines of objects)
  {{"shape": "circle", "center": [u, v], "radius": r}}
  {{"shape": "ellipse", "center": [u, v], "rx": a, "ry": b, "rotation_deg": d}}
  {{"shape": "arc", "center": [u, v], "radius": r, "start_deg": a, "end_deg": b}}   (0 deg = right, 90 = up)
  {{"shape": "rect", "center": [u, v], "w": w, "h": h}}
  {{"shape": "wave", "start": [u, v], "end": [u, v], "amplitude": a, "waves": n}}
  {{"shape": "spiral", "center": [u, v], "radius": r, "turns": n}}
  {{"shape": "hatch", "center": [u, v], "w": w, "h": h, "spacing": s, "angle_deg": d}}  (fills an area with lines)
  {{"shape": "dot", "center": [u, v]}}
Rules:
- Build recognizable objects out of several shapes (e.g. a house = rect + polyline roof + small rect door).
- The SUM of all stroke lengths must be at most {PEN_LIMITS["max_total_length"] - 2:.0f} canvas units
  (a circle of radius r costs 6.3*r; a hatch costs about w*h/spacing - keep hatches small or sparse).
- Strokes are drawn in order; later strokes are drawn over earlier ones.
- Keep everything within [0.04, 0.96]."""

IMAGE_NOTE = ("The image shows the WHOLE canvas: left edge u=0, right edge u=1, bottom edge v=0, top edge v=1. "
              "Pen lines are thin colored tubes on a white canvas.")


# ---------------------------------------------------------------- LLM 呼び出し

def _image_url(image):
    if isinstance(image, (str, Path)):
        data = Path(image).read_bytes()
    else:  # PIL.Image
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def chat(user, system=None, image=None, temperature=0.0, max_tokens=1500):
    """vLLM（OpenAI 互換）の Qwen3-VL に 1 回問い合わせる。"""
    content = ([_image_url(image)] if image is not None else []) + [{"type": "text", "text": user}]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body = {"model": "eye", "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(f"{API_URL}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def say(user, image=None, max_tokens=400):
    """画家（ペルソナ）としてしゃべらせる。"""
    return chat(user, system=PERSONA, image=image, temperature=0.8, max_tokens=max_tokens)


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    i, j = text.find("{"), text.rfind("}")
    return json.loads(text[i: j + 1])


def sections(text, keys):
    """'KEY: ...' 形式の応答を辞書にする。見つからなければ全文を最初のキーに入れる。"""
    pat = r"\**\b(" + "|".join(keys) + r")\b\**\s*[:：]\s*"
    parts = re.split(pat, text)
    d = {k: v.strip().strip("*").strip() for k, v in zip(parts[1::2], parts[2::2])}
    for k in keys:
        d.setdefault(k, "")
    if not any(d.values()):
        d[keys[0]] = text.strip()
    return d


# ---------------------------------------------------------------- 計画

def make_strokes(prompt, image=None, temperature=0.0):
    """筆跡 JSON を作らせる。長すぎ・不正なら理由を伝えて作り直させる。"""
    budget = PEN_LIMITS["max_total_length"] - 2
    last_err = None
    for attempt in range(3):
        text = chat(prompt + (f"\n\nYour previous answer was invalid ({last_err}). Try again." if last_err else ""),
                    image=image, max_tokens=3000, temperature=temperature)
        try:
            raw = extract_json(text)
            plan = validate_plan(raw, limits=PEN_LIMITS)
            length = raw_length(raw)
            if length > budget * 1.1 and attempt < 2:
                last_err = f"total stroke length is {length:.1f}, must be <= {budget:.1f}; use fewer or shorter strokes"
            elif plan["strokes"]:
                return plan
            else:
                last_err = "no valid strokes"
        except (json.JSONDecodeError, ValueError) as e:
            last_err = str(e)[:100]
        print(f"[strokes retry {attempt + 1}] {last_err}", flush=True)
    raise RuntimeError("failed to produce a valid plan")


CANVAS_RGB = (250, 247, 240)
N_CANDIDATES = int(os.environ.get("N_CANDIDATES", "3"))


def render_preview(plan, base=None, size=768):
    """計画をシミュレーションせずに 2D で描いた簡易プレビュー（base = 今のキャンバス画像）。"""
    from PIL import Image, ImageDraw

    img = Image.open(base).convert("RGB").resize((size, size)) if base else Image.new("RGB", (size, size), CANVAS_RGB)
    dr = ImageDraw.Draw(img)
    for s in plan["strokes"]:
        col = tuple(int(255 * c) for c in PALETTE[s["color"]])
        w = int(round(size * 2 * (0.0015 + 0.004 * s["thickness"]) / 0.4))  # pen_robot の線の太さに合わせる
        pts = [(u * size, (1 - v) * size) for u, v in s["points"]]
        if len(pts) == 1:
            x, y = pts[0]
            dr.ellipse([x - w, y - w, x + w, y + w], fill=col)
        else:
            dr.line(pts, fill=col, width=w, joint="curve")
    return img


def best_plan(prompt, theme, base, out, tag):
    """計画を N 案並行で作り、プレビューを審査員に採点させて最良の案を返す。"""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image, ImageDraw

    def one(i):
        try:
            plan = make_strokes(prompt, image=base, temperature=0.0 if i == 0 else 0.8)
        except RuntimeError:
            return None
        prev = render_preview(plan, base)
        prev_path = out / f"{tag}_cand{i}.png"
        prev.save(prev_path)
        return plan, prev_path, judge(theme, prev_path)["after"]

    with ThreadPoolExecutor(N_CANDIDATES) as ex:
        cands = [c for c in ex.map(one, range(N_CANDIDATES)) if c]
    if not cands:
        raise RuntimeError("no candidate plan")
    best = max(range(len(cands)), key=lambda i: cands[i][2])
    print(f"[candidates {tag}] scores={[c[2] for c in cands]} -> pick {best}", flush=True)
    # 動画用に候補を並べた画像を作る
    s = 360
    sheet = Image.new("RGB", (len(cands) * (s + 20) - 20, s + 50), (26, 31, 43))
    dr = ImageDraw.Draw(sheet)
    for i, (_, p, sc) in enumerate(cands):
        sheet.paste(Image.open(p).resize((s, s)), (i * (s + 20), 50))
        if i == best:
            dr.rectangle([i * (s + 20) - 3, 47, i * (s + 20) + s + 2, s + 52], outline=(255, 210, 60), width=5)
        dr.text((i * (s + 20) + 10, 15), f"{sc:g}", fill=(255, 255, 255))
    sheet.save(out / f"{tag}_cands.png")
    (out / f"{tag}_cands.json").write_text(json.dumps({"scores": [c[2] for c in cands], "best": best}))
    return cands[best][0], cands[best][2]


def first_plan(theme, out):
    resp = say(
        f"人間がお題「{theme}」を描けと言ってきました。何回かに分けて描きます。次の形式で答えてください:\n"
        "REACTION: <お題を聞いた最初のひとこと。1〜2 文>\n"
        "TITLE: <作品のタイトル>\n"
        f"IDEA: <完成形の構図。何を・どこに・どの色のペン（{', '.join(COLORS)}）で描くか、具体的な要素を数個>\n"
        "FIRST: <1 回目に描くもの。主要な形・輪郭だけ>"
    )
    print(f"--- 画家 ---\n{resp}\n", flush=True)
    d = sections(resp, ["REACTION", "TITLE", "IDEA", "FIRST"])
    title = d["TITLE"].strip("「」『』\"") or theme
    plan, _ = best_plan(
        f'A robot artist will draw "{theme}" with pens.\nFull concept (Japanese): {d["IDEA"] or resp}\n'
        f"In THIS FIRST SESSION it draws only: {d['FIRST'] or 'the main shapes'}\n"
        f"Turn the first session into concrete pen strokes. Make the main object large and clearly recognizable.\n"
        f"{STROKE_RULES}",
        theme, None, out, "r1",
    )
    plan.update(title=title, theme=theme, round=1)
    dialogue = {"theme": theme, "raw": resp, "reaction": d["REACTION"], "title": title, "idea": d["IDEA"],
                "first": d["FIRST"]}
    (out / "plan_r1.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    (out / "dialogue.json").write_text(json.dumps(dialogue, ensure_ascii=False, indent=1))
    print(json.dumps({"title": title, "n_strokes": len(plan["strokes"])}, ensure_ascii=False), flush=True)
    return dialogue


def drawn_labels(traces):
    """キャンバスに既にある要素の一覧（重複を除く）。描き直し防止のためにプロンプトへ入れる。"""
    seen = []
    for t in traces:
        lab = t.get("label") or ""
        if lab and lab not in seen:
            seen.append(lab)
    return seen


# ---------------------------------------------------------------- 審査

def side_by_side(before, after):
    """加筆前（左）と加筆後（右）を 1 枚に並べる（vLLM は 1 リクエスト 1 画像の設定なので）。"""
    from PIL import Image, ImageDraw

    a, b = Image.open(before).convert("RGB"), Image.open(after).convert("RGB")
    s = 512
    img = Image.new("RGB", (2 * s + 30, s + 50), "white")
    img.paste(a.resize((s, s)), (0, 50))
    img.paste(b.resize((s, s)), (s + 30, 50))
    dr = ImageDraw.Draw(img)
    dr.rectangle([s, 0, s + 30, s + 50], fill="black")
    dr.text((10, 15), "BEFORE (left)", fill="black")
    dr.text((s + 40, 15), "AFTER (right)", fill="black")
    return img


def judge(theme, image, before=None):
    """審査員の採点。before があれば前後比較して両方採点する。戻り値: dict(after, before, comment)。"""
    if before is None:
        q = (f'Theme: "{theme}".\n{IMAGE_NOTE}\nScore this drawing. '
             'Answer ONLY JSON: {"score": <1-10>, "comment": "<one short sentence in Japanese>"}')
        img = image
    else:
        q = (f'Theme: "{theme}".\nThe image shows the same canvas BEFORE (left) and AFTER (right) one more drawing '
             "session. Score each side independently. "
             'Answer ONLY JSON: {"before": <1-10>, "after": <1-10>, '
             '"comment": "<one short sentence in Japanese about what the new session improved or ruined>"}')
        img = side_by_side(before, image)
    for _ in range(3):
        try:
            d = extract_json(chat(q, system=JUDGE, image=img, max_tokens=300))
            after = float(d.get("after", d.get("score")))
            prev = float(d["before"]) if before is not None else None
            return {"after": after, "before": prev, "comment": str(d.get("comment", ""))}
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            continue
    return {"after": 0.0, "before": None, "comment": "（採点に失敗）"}


def critique_and_next(dialogue, image, k, labels, final, failed=""):
    """画家本人が今の絵を見て自己批評し、次に足すものを決める。failed: 直前に取り消された加筆の内容と理由。"""
    ask_next = "" if final else (
        "NEXT: <次の回に描き足すもの。お題が一目で分かるのに足りない要素を 2〜4 個、色と位置を具体的に。"
        "既にある要素は描き直さないこと>\n")
    if failed:
        ask_next += f"（注意: 直前の加筆「{failed}」は審査員に悪化と判定され取り消された。同じことは繰り返さず、別の方法を選ぶこと）\n"
    resp = say(
        f"これはお題「{dialogue['theme']}」の作品『{dialogue['title']}』の、{k} 回目を描き終えた状態です"
        f"{'（これで完成）' if final else ''}。\n"
        f"あなたの構想: {dialogue['idea']}\n既に描いた要素: {', '.join(labels) or 'なし'}\n"
        "画像をよく見て、次の形式で答えてください:\n"
        "CRITIQUE: <今の絵の良い点と足りない点。2 文程度>\n" + ask_next,
        image=image,
    )
    print(f"--- 画家 r{k} ---\n{resp}\n", flush=True)
    d = sections(resp, ["CRITIQUE", "NEXT"])
    return d["CRITIQUE"], ("" if final else d["NEXT"])


def plan_next(dialogue, image, nxt, labels, k, out, failed=""):
    avoid = f"The previous attempt ({failed}) was REJECTED by the judge for making the drawing worse; avoid that.\n" \
        if failed else ""
    plan, _ = best_plan(
        f"{IMAGE_NOTE}\nThis is a robot artist's pen drawing in progress for \"{dialogue['theme']}\".\n"
        f"Full concept (Japanese): {dialogue['idea']}\n"
        f"Elements ALREADY on the canvas (do NOT draw these again, do not trace over them): {labels}\n"
        f"In the NEXT SESSION the artist will ADD (Japanese): {nxt}\n"
        "Look at the drawing and place the new strokes so they fit the existing content. Only add new elements "
        "or new details; keep the drawing clean.\n" + avoid + STROKE_RULES,
        dialogue["theme"], image, out, f"r{k}",
    )
    plan.update(round=k)
    return plan


# ---------------------------------------------------------------- セッション

def draw(out, k, sim_gpu):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(sim_gpu)}
    log = out / f"draw_r{k}.log"
    with log.open("w") as f:
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "pen_robot.py"), str(out), "--round", str(k)],
                           env=env, stdout=f, stderr=subprocess.STDOUT)
    lines = [ln for ln in log.read_text().splitlines() if ln.startswith(("round", "max tracking"))]
    print("\n".join(lines), flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"pen_robot failed, see {log}")


def cmd_session(theme, out, max_rounds, target, sim_gpu):
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces.json").unlink(missing_ok=True)
    dialogue = first_plan(theme, out)
    accepted_img, accepted_score, best_round = None, 0.0, 0
    prev_next = dialogue["first"]  # 直前の回に描こうとしたもの（取り消し時の説明用）
    k = 1
    while True:
        print(f"===== session {k} =====", flush=True)
        if k > 1:  # 採用済みの状態から描く（取り消された回の線は traces に入っていない）
            shutil.copy(out / f"traces_r{best_round}.json", out / "traces.json")
        draw(out, k, sim_gpu)
        shutil.copy(out / "traces.json", out / f"traces_r{k}.json")
        img = out / f"artwork_r{k}.png"

        j = judge(theme, img, before=accepted_img)
        # 基準は採用時に固定した点数。「前の絵」をその都度採点し直すと点がぶれて判定が揺れるので使わない
        baseline = None if accepted_img is None else accepted_score
        j["baseline"] = baseline
        accepted = baseline is None or j["after"] >= baseline
        if accepted:
            accepted_img, accepted_score, best_round = img, j["after"], k
        print(f"[judge r{k}] baseline={baseline} after={j['after']} -> {'採用' if accepted else '取り消し'}: "
              f"{j['comment']}", flush=True)

        final = accepted_score >= target or k >= max_rounds
        labels = drawn_labels(json.loads((out / f"traces_r{best_round}.json").read_text()))
        if accepted:
            critique, nxt = critique_and_next(dialogue, accepted_img, k, labels, final)
        else:
            critique = say(
                f"お題「{theme}」の作品で、{k} 回目に描き足したら審査員に「悪化した」と判定され、"
                f"その回は取り消されました（審査員のコメント: {j['comment']}）。"
                "これを受けての感想を 1〜2 文で述べてください。",
                image=img, max_tokens=200)
            failed = f"{prev_next} → 審査員: {j['comment']}"
            nxt = "" if final else critique_and_next(dialogue, accepted_img, best_round, labels, False, failed)[1]
        rec = {"round": k, "accepted": accepted, "judge": j, "score": accepted_score, "critique": critique,
               "next": nxt, "final": final, "best_round": best_round}
        (out / f"review_r{k}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        print(json.dumps({k: rec[k] for k in ("accepted", "score", "critique", "next")}, ensure_ascii=False),
              flush=True)
        if final:
            break
        plan = plan_next(dialogue, accepted_img, nxt, labels, k + 1, out, "" if accepted else failed)
        prev_next = nxt
        (out / f"plan_r{k + 1}.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
        k += 1
    summary = {"rounds": k, "title": dialogue["title"], "best_round": best_round, "score": accepted_score,
               "target": target}
    (out / "session.json").write_text(json.dumps(summary, ensure_ascii=False))
    print(f"finished after {k} sessions: best round {best_round}, score {accepted_score}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("theme")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=5, help="最大のセッション数")
    p.add_argument("--target", type=float, default=8.0, help="審査員の点数がこれ以上になったら終了")
    p.add_argument("--sim-gpu", default="0", help="pen_robot を動かす GPU")
    args = p.parse_args()
    cmd_session(args.theme, args.out, args.rounds, args.target, args.sim_gpu)


if __name__ == "__main__":
    main()
