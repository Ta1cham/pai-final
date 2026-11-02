"""1 作品分（複数セッション）の動画を組み立てる（ffmpeg + ASS 字幕）。

構成: お題 → 第一声 → 構想 → [第 k 回 制作（倍速）→ 途中経過と自己批評・次に足すもの] × k → 完成

    uv run python painter/make_video.py outputs/pen_sea
"""

import argparse
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

W, H, FPS = 1280, 720, 50
BG = "0x1a1f2b"
FONT = "Noto Sans CJK JP"
IMG = 620  # 批評画面の作品サイズ（左側）
TEXT_L = IMG + 80  # 右側テキストの左端

ASS_HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: JA,{FONT},36,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,1,2,60,60,40,1
Style: EN,{FONT},22,&H0090E0FF,&H00FFFFFF,&H00000000,&H80000000,0,1,0,0,100,100,0,0,1,2,0,2,80,80,40,1
Style: HEAD,{FONT},44,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,1,8,60,60,40,1
Style: SMALL,{FONT},26,&H00C0C0C0,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,9,30,30,20,1
Style: SIDEH,{FONT},34,&H0060D0FF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,7,{TEXT_L},40,50,1
Style: SIDE,{FONT},28,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,7,{TEXT_L},40,110,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ts(sec):
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def wrap_ja(text, width=30):
    # 日本語は空白で折り返せないので文字数で区切る
    lines = textwrap.wrap(text.replace("\n", " "), width=width, break_long_words=True)
    if len(lines) > 1 and len(lines[-1]) <= 3:  # 「。」だけの行を作らない
        last = lines.pop()
        lines[-1] += last
    return r"\N".join(lines)


def wrap_en(text, width=90):
    return r"\N".join(textwrap.wrap(text.replace("\n", " "), width=width))


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout)


class Builder:
    def __init__(self, tmp):
        self.tmp, self.segs = tmp, []

    def _ass(self, dur, events):
        path = self.tmp / f"s{len(self.segs)}.ass"
        lines = [ASS_HEADER] + [f"Dialogue: 0,{ts(0)},{ts(dur)},{st},,0,0,{mv},,{tx}\n" for st, tx, mv in events]
        path.write_text("".join(lines), encoding="utf-8")
        return path

    def _encode(self, inp, vf, dur, events):
        out = self.tmp / f"s{len(self.segs)}.mp4"
        vf = f"{vf},setsar=1,subtitles={self._ass(dur, events)}"
        run(["ffmpeg", "-y", *inp, "-vf", vf, "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "fast", "-crf", "20", "-an", str(out)])
        self.segs.append(out)

    def card(self, dur, events):
        """単色背景＋文字。"""
        self._encode(["-f", "lavfi", "-t", f"{dur}", "-i", f"color=c={BG}:s={W}x{H}:r={FPS}"], "null", dur, events)

    def image(self, path, dur, events, side=False):
        """静止画。side=True なら作品を左に置き、右側を文字の場所にする。"""
        if side:
            vf = f"scale={IMG}:{IMG},pad={W}:{H}:40:{(H - IMG) // 2}:color={BG}"
        else:
            vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={BG}"
        self._encode(["-loop", "1", "-t", f"{dur}", "-i", str(path)], vf, dur, events)

    def video(self, path, speed, events):
        dur = duration(path) / speed
        self._encode(["-i", str(path)], f"setpts=PTS/{speed}", dur, events)

    def concat(self, out):
        lst = self.tmp / "list.txt"
        lst.write_text("".join(f"file '{s}'\n" for s in self.segs))
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])


def read_time(text):
    return max(4.0, min(14.0, len(text) / 8))  # 日本語 8 文字/秒を目安に


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    d = args.run_dir
    dia = json.loads((d / "dialogue.json").read_text())
    title = dia["title"]
    rounds = sorted(int(f.stem.split("_r")[1]) for f in d.glob("artwork_r*.png"))
    out = args.out or d / "final.mp4"

    with tempfile.TemporaryDirectory() as t:
        b = Builder(Path(t))
        b.card(3.0, [("HEAD", "お題", 220), ("JA", f"「{dia['theme']}」", 320)])
        ja = wrap_ja(dia.get("reaction") or "")
        b.card(read_time(ja), [("HEAD", "画家ロボ", 60), ("JA", ja, 300 - 20 * ja.count(r"\N"))])
        ja = wrap_ja(dia.get("idea") or "", 34)
        b.card(read_time(ja), [("HEAD", f"構想『{title}』", 60), ("JA", ja, 120)])

        best = rounds[-1]
        for k in rounds:
            cands = d / f"r{k}_cands.png"
            if cands.exists():
                info = json.loads((d / f"r{k}_cands.json").read_text())
                sc = " / ".join(f"{x:g}" for x in info["scores"])
                b.image(cands, 3.5, [("HEAD", f"第{k}回の計画: 候補 {len(info['scores'])} 案", 30),
                                     ("JA", f"プレビューの審査点 {sc} → 最高点の案を描く", 0)])
            b.video(d / f"painting_r{k}.mp4", args.speed, [
                ("SMALL", f"{args.speed:g}倍速", 0), ("JA", f"第{k}回 制作中…", 0)])
            rv_path = d / f"review_r{k}.json"
            if not rv_path.exists():
                continue
            rv = json.loads(rv_path.read_text())
            best = rv.get("best_round", k)
            jd = rv["judge"]
            if jd.get("baseline") is not None:
                verdict = f"審査員: {jd['after']:g} 点（基準 {jd['baseline']:g} 点）"
            else:
                verdict = f"審査員: {jd['after']:g} 点"
            verdict += r"\N" + wrap_ja(jd.get("comment") or "", 17)
            if rv["accepted"]:
                head = f"第{k}回 採用"
            else:
                head = f"第{k}回 {{\\c&H5050FF&}}取り消し"
            text = r"{\c&HA0A0A0&}" + verdict + r"{\c&HFFFFFF&}\N\N" + wrap_ja(rv.get("critique") or "", 17)
            nxt = rv.get("next") or ""
            if nxt and not rv.get("final"):
                text += r"\N\N{\c&H60D0FF&}次に描くもの:{\c&HFFFFFF&}\N" + wrap_ja(nxt, 17)
            b.image(d / f"artwork_r{k}.png", read_time(verdict + (rv.get("critique") or "") + nxt), [
                ("SIDEH", head, 0), ("SIDE", text, 0)], side=True)

        sess = json.loads((d / "session.json").read_text()) if (d / "session.json").exists() else {}
        score = f"審査員 {sess['score']:g} 点" if "score" in sess else ""
        b.image(d / f"artwork_r{best}.png", 5.0,
                [("SIDEH", "完成", 0), ("SIDE", f"『{wrap_ja(title, 15)}』\\N\\N{score}", 0)], side=True)
        b.concat(out)
    print(f"saved: {out.resolve()}")


if __name__ == "__main__":
    main()
