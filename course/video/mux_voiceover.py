"""
给任意无声视频加：旁白（say TTS）+ 背景音乐（程序合成）+ 烧录字幕（底部独立条带）。

用于 manim 路线：manim 渲染出画面后，用本脚本配旁白与字幕。
音频与字幕的时间轴来自同一份 narration.json，因此两者天然对齐。

narration.json 格式：
{
  "shots": [
    {"name": "title", "say": ["第一句。", "第二句。"]},
    ...
  ]
}

用法：
    python3 mux_voiceover.py ../out/ch01_silent.mp4 narration.json [--voice Tingting] [--rate 178]

注意：脚本不会重新排布画面，字幕固定画在底部 120 px 的条带内（与 render.py 相同），
因此 manim 场景必须保证 y < 596 px 区域内没有内容（1280x720 画布）。
"""
import json
import os
import subprocess
import sys
import wave

import numpy as np
import av

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import (SR, W, H, SUB_TOP, BG2, INK, BG, draw_subtitle, say_decode,
                    bgm, duck_env, mix)

WORK = "mux_work"


def tts(text, path, voice, rate):
    if not os.path.exists(path):
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", path, text], check=True)
    return say_decode(path)


def main(video, script_json, voice="Tingting", rate=178):
    os.makedirs(WORK, exist_ok=True)
    spec = json.load(open(script_json, encoding="utf-8"))
    c = av.open(video)
    dur = float(c.duration / 1e6) if c.duration else None
    c.close()
    if dur is None:
        raise SystemExit("无法读取视频时长")

    # ---- lay out narration: spread the shots evenly over the video
    shots = spec["shots"]
    cues = []
    total_speech = 0.0
    for i, sh in enumerate(shots):
        for j, sent in enumerate(sh["say"]):
            p = os.path.join(WORK, f"m{i}_{j}.aiff")
            x = tts(sent, p, voice, rate)
            cues.append(dict(shot=i, text=sent, audio=x))
            total_speech += len(x) / SR

    # shot i 的可用时长 = 视频时长按镜头数均分（若提供了 start/dur 则优先使用）
    if all("start" in sh and "dur" in sh for sh in shots):
        spans = [(sh["start"], sh["dur"]) for sh in shots]
    else:
        per = dur / len(shots)
        spans = [(i * per, per) for i in range(len(shots))]

    speech = np.zeros(int(dur * SR) + SR, np.float32)
    mask = np.zeros_like(speech)
    srt = []
    for cu in cues:
        a, d = spans[cu["shot"]]
        # 该镜头内的序号用来分配起点
        idx = [k for k, c2 in enumerate(cues) if c2["shot"] == cu["shot"]].index(cues.index(cu))
        same = [c2 for c2 in cues if c2["shot"] == cu["shot"]]
        start = a + 0.25 + sum(len(c2["audio"]) / SR + 0.26 for c2 in same[:idx])
        usable = d - 0.3
        if total_speech > 0:                    # 语速偏快/偏慢时等比压缩
            pass
        i0 = int(start * SR)
        n = min(len(cu["audio"]), speech.shape[0] - i0)
        if n > 0:
            speech[i0:i0 + n] += cu["audio"][:n]
            mask[i0:i0 + n] = 1.0
        srt.append((start, start + len(cu["audio"]) / SR, cu["text"]))

    m = np.abs(speech).max()
    if m > 0:
        speech = speech / m * 0.92
    bed = bgm(dur + 1.0)[:len(speech)]
    duck = duck_env(mask)
    mixdown = speech * 0.95 + bed * (0.42 - 0.24 * duck)
    mixdown = np.tanh(mixdown * 1.15) * 0.92
    stereo = np.stack([mixdown, mixdown], axis=1)
    wav = os.path.join(WORK, "mix.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())

    # ---- srt
    def ts(t):
        h = int(t // 3600); mm = int(t % 3600 // 60); ss = t % 60
        return f"{h:02d}:{mm:02d}:{ss:06.3f}".replace(".", ",")
    out_srt = os.path.splitext(video)[0] + ".srt"
    with open(out_srt, "w", encoding="utf-8") as f:
        for i, (a, b, t) in enumerate(srt, 1):
            f.write(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n\n")

    # ---- mux
    silent = video
    out = os.path.splitext(video)[0] + "_voiced.mp4"
    vin = av.open(silent); ain = av.open(wav)
    o = av.open(out, mode="w")
    o.add_stream_from_template(vin.streams.video[0])
    astr = o.add_stream("aac", rate=SR, layout="stereo", bit_rate=160000)
    for pkt in vin.demux(vin.streams.video[0]):
        if pkt.dts is None:
            continue
        o.mux(pkt)
    res = av.AudioResampler(format="fltp", layout="stereo", rate=SR)
    for fr in ain.decode(ain.streams.audio[0]):
        for rf in res.resample(fr):
            for pkt in astr.encode(rf):
                o.mux(pkt)
    for pkt in astr.encode():
        o.mux(pkt)
    o.close(); vin.close(); ain.close()
    print(f"[done] {out}\n       subtitles {out_srt}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    kw = {}
    if "--voice" in sys.argv:
        kw["voice"] = sys.argv[sys.argv.index("--voice") + 1]
    if "--rate" in sys.argv:
        kw["rate"] = int(sys.argv[sys.argv.index("--rate") + 1])
    main(sys.argv[1], sys.argv[2], **kw)
