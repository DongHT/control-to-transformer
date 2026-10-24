"""
Tiny "3b1b-style" video engine: Pillow frames -> H.264 (PyAV) + TTS (say) + synth BGM.

Design rules enforced by the engine
-----------------------------------
* 1280x720 @ 30 fps, drawn at 2x and downsampled (cheap anti-aliasing).
* SAFE AREA: nothing but the subtitle band may be drawn below y = 596 px.
  The band [600, 720] is filled with a slightly different tone and holds one
  subtitle line at a time -> subtitles never cover the picture.
* Every text run registers its bounding box; `Canvas.check()` reports any
  overlapping pairs, so text collisions are caught mechanically, not by eye.
* No combining diacritics (no "theta-dot"): all derivatives are written as
  d/dt, which removes the main source of garbled glyphs.
"""
import os
import sys
import hashlib
import json
import math
import wave
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mathrender as mr

import av

# ------------------------------------------------------------------ constants
W, H = 1280, 720
FPS = 30
SS = 2                       # supersampling factor
SAFE_BOTTOM = 640            # content may come this low
SUB_TOP = 600                # kept for compatibility
BG = (10, 15, 24)
BG2 = (13, 19, 30)
INK = (232, 240, 248)
DIM = (150, 166, 183)
FAINT = (96, 112, 130)
CY = (47, 211, 192)
AM = (242, 169, 59)
MG = (224, 97, 158)
BL = (106, 169, 245)
GR = (123, 214, 107)

FONT_PATH = "/System/Library/Fonts/Songti.ttc"
FACES = {"regular": 6, "bold": 1, "light": 3, "black": 0}
_font_cache = {}


def font(size, weight="regular"):
    key = (size, weight)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONT_PATH, int(size), index=FACES[weight])
    return _font_cache[key]


def mix(c, bg, a):
    """fake alpha: blend colour c towards bg by a in [0,1]."""
    a = max(0.0, min(1.0, a))
    return tuple(int(round(c[i] * a + bg[i] * (1 - a))) for i in range(3))


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def appear(t, t0, dur=0.6):
    if t < t0:
        return 0.0
    return smooth((t - t0) / dur)


def lerp(a, b, u):
    return a + (b - a) * u


# ------------------------------------------------------------------ canvas
class Canvas:
    def __init__(self, ss=SS, bg=BG):
        self.ss = ss
        self.W, self.H = W, H
        self.bg = bg
        self.img = Image.new("RGB", (W * ss, H * ss), bg)
        self.d = ImageDraw.Draw(self.img)
        self.boxes = []          # (x0,y0,x1,y1,tag)
        self.reserved = []       # graphics areas that text must not cover
        self.dot_grid()

    # ---------------- background
    def dot_grid(self):
        step = 40
        for x in range(0, W, step):
            for y in range(0, H, step):
                self.d.point([(x * self.ss, y * self.ss)], fill=mix((255, 255, 255), self.bg, 0.045))

    def reserve(self, x0, y0, x1, y1, tag="gfx"):
        self.reserved.append((x0, y0, x1, y1, tag))

    # ---------------- primitives (design units, y down)
    def _p(self, x, y):
        return (x * self.ss, y * self.ss)

    def line(self, p0, p1, color, width=2, alpha=1.0, dash=None):
        c = mix(color, self.bg, alpha)
        if dash:
            (x0, y0), (x1, y1) = p0, p1
            L = math.hypot(x1 - x0, y1 - y0)
            n = max(1, int(L / dash))
            for i in range(n):
                u0, u1 = i / n, (i + 0.55) / n
                self.d.line([self._p(lerp(x0, x1, u0), lerp(y0, y1, u0)),
                             self._p(lerp(x0, x1, u1), lerp(y0, y1, u1))],
                            fill=c, width=int(width * self.ss))
        else:
            self.d.line([self._p(*p0), self._p(*p1)], fill=c, width=int(width * self.ss))

    def rect(self, x0, y0, x1, y1, color, width=2, alpha=1.0, fill_alpha=0.0, radius=0):
        c = mix(color, self.bg, alpha)
        fill = mix(color, self.bg, fill_alpha) if fill_alpha > 0 else None
        if radius:
            self.d.rounded_rectangle([self._p(x0, y0), self._p(x1, y1)], radius=radius * self.ss,
                                     outline=c, width=int(width * self.ss), fill=fill)
        else:
            self.d.rectangle([self._p(x0, y0), self._p(x1, y1)], outline=c,
                             width=int(width * self.ss), fill=fill)

    def circle(self, cx, cy, r, color, width=2, alpha=1.0, fill_alpha=0.0):
        c = mix(color, self.bg, alpha)
        fill = mix(color, self.bg, fill_alpha) if fill_alpha > 0 else None
        self.d.ellipse([self._p(cx - r, cy - r), self._p(cx + r, cy + r)],
                       outline=c, width=int(width * self.ss), fill=fill)

    def arrow(self, p0, p1, color, width=2, alpha=1.0, head=11):
        self.line(p0, p1, color, width, alpha)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        for s in (+1, -1):
            a = ang + s * 2.55
            self.line(p1, (p1[0] + head * math.cos(a), p1[1] + head * math.sin(a)), color, width, alpha)

    # ---------------- rich text: supports _{...} and ^{...}
    def text(self, x, y, s, size=32, color=INK, anchor="la", weight="regular",
             alpha=1.0, check=True, tag=""):
        ss = self.ss
        parts = parse_runs(s, size * ss, weight)      # widths in device px
        total = sum(p[1] for p in parts) / ss          # design px
        if anchor[0] == "c":
            cx = x - total / 2
        elif anchor[0] == "r":
            cx = x - total
        else:
            cx = x
        col = mix(color, self.bg, alpha)
        y0, y1 = 1e9, -1e9
        x0 = cx
        for txt, wdt_dev, dy_dev, sz_dev, wt in parts:
            if txt:
                px, py = cx * ss, y * ss + dy_dev
                self.d.text((px, py), txt, font=font(sz_dev, wt), fill=col, anchor="ls")
                bb = self.d.textbbox((px, py), txt, font=font(sz_dev, wt), anchor="ls")
                y0 = min(y0, bb[1] / ss)
                y1 = max(y1, bb[3] / ss)
            cx += wdt_dev / ss
        box = (x0, y0, x0 + total, y1)
        if check:
            self.boxes.append((box[0], box[1], box[2], box[3], tag or s[:18]))
        return box

    def text_w(self, s, size=32, weight="regular"):
        return sum(p[1] for p in parse_runs(s, size * self.ss, weight)) / self.ss

    # ---------------- checks
    def check(self, tol=2.0):
        bad = []
        b = [x for x in self.boxes]
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                a, c = b[i], b[j]
                ix = min(a[2], c[2]) - max(a[0], c[0])
                iy = min(a[3], c[3]) - max(a[1], c[1])
                if ix > tol and iy > tol:
                    bad.append((a[4], c[4], round(ix, 1), round(iy, 1)))
        for t in self.boxes:
            for r in self.reserved:
                ix = min(t[2], r[2]) - max(t[0], r[0])
                iy = min(t[3], r[3]) - max(t[1], r[1])
                if ix > tol and iy > tol:
                    bad.append((t[4], "reserved:" + r[4], round(ix, 1), round(iy, 1)))
        return bad

    def finish(self):
        img = self.img.resize((W, H), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)


SUBFONT = "/System/Library/Fonts/STHeiti Medium.ttc"


def _sub_font(size):
    key = ("sub", size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(SUBFONT, int(size), index=0)
    return _font_cache[key]


def _sub_tokens(text):
    """Split a subtitle into plain runs and $...$ inline-LaTeX runs."""
    out, i = [], 0
    while i < len(text):
        j = text.find("$", i)
        if j < 0:
            out.append(("t", text[i:])); break
        if j > i:
            out.append(("t", text[i:j]))
        k = text.find("$", j + 1)
        if k < 0:
            out.append(("t", text[j:])); break
        out.append(("m", text[j + 1:k]))
        i = k + 1
    return [t for t in out if t[1]]


def _measure(tok, size, ss):
    kind, val = tok
    if kind == "t":
        return _sub_font(size * ss).getlength(val) / ss
    return mr.render(val, int(size * ss * 0.98), "#ffffff").width / ss


def draw_subtitle(cv, text, alpha=0.72):
    """Semi-transparent subtitle: true alpha compositing, no backing band."""
    ss = cv.ss
    size = 32
    maxw = W - 120
    toks = _sub_tokens(text)

    lines, cur, curw = [], [], 0.0
    for kind, val in toks:
        if kind == "m":
            w = _measure((kind, val), size, ss)
            if curw + w > maxw and cur:
                lines.append(cur); cur, curw = [], 0.0
            cur.append((kind, val, w)); curw += w
        else:
            piece = ""
            for ch in val:
                cw = _measure(("t", ch), size, ss)
                if curw + cw > maxw and (cur or piece):
                    if piece:
                        cur.append(("t", piece, _measure(("t", piece), size, ss)))
                    lines.append(cur); cur, curw, piece = [], 0.0, ""
                piece += ch
                curw += cw
            if piece:
                cur.append(("t", piece, _measure(("t", piece), size, ss)))
    if cur:
        lines.append(cur)
    lines = lines[:2]

    # draw onto an RGBA layer, then composite once -> real transparency
    top = 636
    layer = Image.new("RGBA", (W * ss, (H - top) * ss), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(layer)
    f = _sub_font(size * ss)
    y = 700 if len(lines) == 1 else 668
    a_main = int(255 * alpha)
    for line in lines:
        lw = sum(t[2] for t in line)
        x = (W - lw) / 2
        for kind, val, w in line:
            if kind == "t":
                for ddy, aa in ((3, 0.30), (1, 0.40)):
                    d2.text((x * ss, (y - top) * ss + ddy * ss), val, font=f,
                            fill=(0, 0, 0, int(255 * aa)), anchor="ls")
                d2.text((x * ss, (y - top) * ss), val, font=f,
                        fill=(240, 246, 252, a_main), anchor="ls")
            else:
                img = mr.render(val, int(size * ss * 0.98), "#f0f6fc")
                bl = mr.baseline_of(val, int(size * ss * 0.98), "#f0f6fc")
                px_ = int(x * ss)
                py_ = int((y - top) * ss - img.height * bl)
                sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
                sh.putalpha(img.split()[3].point(lambda v: int(v * 0.35)))
                layer.alpha_composite(sh, (px_ + 2, py_ + 3))
                m = img.copy()
                m.putalpha(img.split()[3].point(lambda v: int(v * alpha)))
                layer.alpha_composite(m, (px_, py_))
            x += w
        y += size + 8
    region = cv.img.crop((0, top * ss, W * ss, H * ss)).convert("RGBA")
    merged = Image.alpha_composite(region, layer)
    cv.img.paste(merged.convert("RGB"), (0, top * ss))


def parse_runs(s, size, weight="regular"):
    """Return [(text, width, dy, size, weight)] for a string with _{..}/^{..}."""
    out = []
    i = 0
    while i < len(s):
        if s[i] in "_^" and i + 1 < len(s) and s[i + 1] == "{":
            j = s.index("}", i + 2)
            body = s[i + 2:j]
            sz = size * 0.66
            dy = +size * 0.20 if s[i] == "_" else -size * 0.44
            w = font(sz, weight).getlength(body)
            out.append((body, w, dy, sz, weight))
            i = j + 1
        else:
            j = i
            while j < len(s) and not (s[j] in "_^" and j + 1 < len(s) and s[j + 1] == "{"):
                j += 1
            body = s[i:j]
            out.append((body, font(size, weight).getlength(body), 0, size, weight))
            i = j
    return out


# ------------------------------------------------------------------ audio
SR = 44100


EDGE = os.path.expanduser("~/Library/Python/3.9/bin/edge-tts")

_SPEAK = [(r"\theta", "theta"), (r"\omega", "omega"), (r"\pi", "pi"),
          (r"\sin", "sin"), (r"\cos", "cos"), (r"\pm", "正负"),
          (r"\%", "百分之"), (r"\circ", "度"), (r"\,", " "), (r"\ ", " ")]


def _speakable(s):
    """Strip LaTeX so the TTS never reads '$' or backslashes."""
    if "$" not in s:
        return s
    import re as _re
    def one(m):
        t = m.group(1)
        for a, b in _SPEAK:
            t = t.replace(a, b)
        t = _re.sub(r"\\[a-zA-Z]+", "", t)
        t = t.replace("{", "").replace("}", "").replace("^", "").replace("_", "")
        return t
    return _re.sub(r"\$([^$]*)\$", one, s)


def synth(text, path, voice, rate=178, engine="edge"):
    """engine='edge' -> Microsoft neural voice (mp3); 'say' -> macOS TTS (aiff)."""
    if engine == "edge":
        r = f"{rate:+d}%" if isinstance(rate, int) else str(rate)
        res = subprocess.run([EDGE, "--voice", voice, "--rate=" + r,
                              "--text", text, "--write-media", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0 or not os.path.exists(path):
            raise RuntimeError("edge-tts failed: " + res.stderr.decode()[:300])
    else:
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", path, text], check=True)


def say_to_array(text, path, voice="Tingting", rate=178, engine="say"):
    synth(text, path, voice, rate, engine)
    c = av.open(path)
    st = c.streams.audio[0]
    chunks = []
    for fr in c.decode(st):
        a = fr.to_ndarray()
        if a.ndim == 2:
            a = a.mean(axis=0)
        chunks.append(a.astype(np.float32))
    c.close()
    x = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    sr = st.rate
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    m = np.abs(x).max()
    if m > 1e-6:
        x = x / m * 0.85
    return x


def bgm(duration, seed=3):
    """Slow ambient pad: Am - F - C - G, soft attack, gentle low-pass."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    chords = [([220.00, 261.63, 329.63], 4.0),      # Am
              ([174.61, 220.00, 261.63], 4.0),      # F
              ([261.63, 329.63, 392.00], 4.0),      # C
              ([196.00, 246.94, 293.66], 4.0)]      # G
    out = np.zeros(n, np.float32)
    pos = 0.0
    while pos < duration:
        for freqs, dur in chords:
            i0 = int(pos * SR)
            i1 = min(n, int((pos + dur) * SR))
            if i0 >= n:
                break
            seg = np.arange(i1 - i0) / SR
            env = np.minimum(1.0, seg / 1.6) * np.clip((dur - seg) / 1.8, 0, 1)
            env = np.clip(env, 0, 1) ** 1.5
            for k, f in enumerate(freqs):
                det = 1 + 0.0015 * math.sin(2 * math.pi * 0.07 * pos + k)
                out[i0:i1] += 0.12 * env * np.sin(2 * np.pi * f * det * seg)
            pos += dur
    y = lfilter([0.16], [1, -0.84], out)            # one-pole low pass
    y = y / (np.abs(y).max() + 1e-9)
    return (y * 0.19).astype(np.float32)


def duck_env(speech_mask, sr=SR, block_s=0.05, smooth_blocks=9):
    """Cheap smoothed activity envelope (blocked + tiny kernel + interp)."""
    b = int(block_s * sr)
    nb = int(np.ceil(len(speech_mask) / b))
    pad = np.zeros(nb * b, np.float32)
    pad[:len(speech_mask)] = speech_mask
    coarse = pad.reshape(nb, b).mean(axis=1)
    k = np.hanning(smooth_blocks)
    k /= k.sum()
    sm = np.convolve(coarse, k, mode="same")
    x_old = np.arange(nb) * b
    x_new = np.arange(len(speech_mask))
    return np.interp(x_new, x_old, sm).astype(np.float32)


# ------------------------------------------------------------------ pipeline
def build(video_spec, out_mp4, work="tmp_video", voice="Tingting", rate=178,
          check_frames=6, bench=None, engine="say"):
    os.makedirs(work, exist_ok=True)
    shots = video_spec["shots"]
    audio_dir = os.path.join(work, "tts")
    os.makedirs(audio_dir, exist_ok=True)

    # ---- 1. synthesize narration, work out the timeline
    timeline = []
    t = 0.35
    total = 0.35
    for si, sh in enumerate(shots):
        cues = []
        for ci, sent in enumerate(sh["say"]):
            if isinstance(sent, dict):
                shown, spoken = sent["say"], sent.get("tts") or _speakable(sent["say"])
            else:
                shown, spoken = sent, _speakable(sent)
            if "$" in spoken or "\\" in spoken:
                print(f"[warn] LaTeX leaked into TTS: {spoken[:40]}")
            ext = ".mp3" if engine == "edge" else ".aiff"
            # cache key = the SPOKEN TEXT (not the shot index!) -- otherwise every
            # chapter reuses the previous chapter's audio, which is exactly the bug
            # that shipped in ch03..ch10 of the first release.
            key = hashlib.sha1(f"{spoken}|{voice}|{rate}|{engine}".encode()).hexdigest()[:16]
            path = os.path.join(audio_dir, f"v_{key}{ext}")
            if not os.path.exists(path):
                synth(spoken, path, voice, rate, engine)
            x = say_to_array_cached(path)
            dur = len(x) / SR
            cues.append(dict(text=shown, audio=x, dur=dur, start=t))
            t += dur + 0.26
        sh["_cues"] = cues
        sh["_start"] = timeline[-1]["end"] if timeline else 0.35
        sh["_dur"] = max(sh.get("min_dur", 3.0), (t - sh["_start"]) + sh.get("tail", 0.7))
        timeline.append(dict(start=sh["_start"], end=sh["_start"] + sh["_dur"], shot=si))
        total = sh["_start"] + sh["_dur"]
        t = total + 0.30
    total += 0.9

    print(f"[timeline] {len(shots)} shots, total {total:.1f}s")
    for sh in shots:
        print(f"   {sh['name']:14s} start {sh['_start']:6.1f}  dur {sh['_dur']:5.1f}")

    # ---- 2. audio mix
    speech = np.zeros(int(total * SR) + SR, np.float32)
    mask = np.zeros_like(speech)
    for sh in shots:
        for c in sh["_cues"]:
            i0 = int(c["start"] * SR)
            speech[i0:i0 + len(c["audio"])] += c["audio"]
            mask[i0:i0 + len(c["audio"])] = 1.0
    m = np.abs(speech).max()
    if m > 0:
        speech /= m
        speech *= 0.92
    bed = bgm(total + 1.0)
    bed = bed[:len(speech)]
    duck = duck_env(mask)
    gain = 0.62 - 0.26 * duck
    mixdown = speech * 0.95 + bed * gain
    # gentle limiter + fades
    mixdown = np.tanh(mixdown * 1.15) * 0.92
    fade = int(0.6 * SR)
    mixdown[:fade] *= np.linspace(0, 1, fade)
    mixdown[-fade:] *= np.linspace(1, 0, fade)
    stereo = np.stack([mixdown, mixdown], axis=1)
    wav_path = os.path.join(work, "mix.wav")
    pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("[audio] wrote", wav_path, f"{len(mixdown)/SR:.1f}s")

    # ---- 3. srt
    srt_path = os.path.splitext(out_mp4)[0] + ".srt"
    write_srt(shots, srt_path, maxchars=46)

    # ---- 4. frames -> silent video
    nframes = int(round(total * FPS))
    if bench:
        nframes = min(nframes, bench)
    silent = os.path.join(work, "silent.mp4")
    problems = []
    container = av.open(silent, mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height = W, H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "21", "preset": "medium", "x264-params": "keyint=60"}
    for k in range(nframes):
        tt = k / FPS
        cv = Canvas()
        active = None
        for si, sh in enumerate(shots):
            if sh["_start"] <= tt < sh["_start"] + sh["_dur"] or \
               (si == len(shots) - 1 and tt >= sh["_start"]):
                local = tt - sh["_start"]
                active = sh
                sh["draw"](cv, local, sh["_dur"])
                break
        if active is not None:
            cue = None
            for c in active["_cues"]:
                if c["start"] - 0.05 <= tt < c["start"] + c["dur"] + 0.30:
                    cue = c
            if cue is not None:
                draw_subtitle(cv, cue["text"])
        if k % 15 == 0 or k < check_frames * 10:
            problems += [(round(tt, 2),) + tuple(b) for b in cv.check()]
        frame = cv.finish()
        vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
        vf.pts = k
        for pkt in stream.encode(vf):
            container.mux(pkt)
        if k % 300 == 0:
            print(f"[video] {k}/{nframes}  t={tt:5.1f}s  collisions={len(problems)}", flush=True)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()

    # ---- 5. mux audio
    vin = av.open(silent)
    ain = av.open(wav_path)
    out = av.open(out_mp4, mode="w")
    vs = out.add_stream_from_template(vin.streams.video[0])
    as_ = out.add_stream("aac", rate=SR, layout="stereo", bit_rate=160000)
    for pkt in vin.demux(vs):
        if pkt.dts is None:
            continue
        out.mux(pkt)
    res = av.AudioResampler(format="fltp", layout="stereo", rate=SR)
    pts = 0
    for fr in ain.decode(ain.streams.audio[0]):
        for rf in res.resample(fr):
            for pkt in as_.encode(rf):
                out.mux(pkt)
    for pkt in as_.encode():
        out.mux(pkt)
    out.close()
    vin.close()
    ain.close()

    print(f"[done] {out_mp4}  {os.path.getsize(out_mp4)/1e6:.1f} MB  "
          f"{nframes} frames  collisions={len(problems)}")
    if problems:
        print("[collisions] first 10:", problems[:10])
    return dict(total=total, frames=nframes, collisions=problems, srt=srt_path)


_tts_cache = {}


def say_to_array_cached(path):
    if path not in _tts_cache:
        _tts_cache[path] = say_decode(path)
    return _tts_cache[path]


def say_decode(path):
    c = av.open(path)
    st = c.streams.audio[0]
    chunks = []
    for fr in c.decode(st):
        a = fr.to_ndarray()
        if a.ndim == 2:
            a = a.mean(axis=0)
        chunks.append(a.astype(np.float32))
    sr = st.rate
    c.close()
    x = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    m = np.abs(x).max()
    if m > 1e-6:
        x = x / m * 0.85
    return x


def _plain(s):
    """Strip $...$ delimiters for the .srt file (players show raw text)."""
    import re as _re

    def repl(m):
        t = m.group(1)
        t = _re.sub(r"\\mathrm\{([^}]*)\}", r"\1", t)
        t = t.replace("\\,", " ").replace("\\ ", " ").replace("\\!", "")
        t = t.replace("\\%", "%").replace("\\&", "&").replace("\\$", "$")
        t = t.replace("\\{", "{").replace("\\}", "}")
        t = t.replace("\\theta", "θ").replace("\\omega", "ω").replace("\\pi", "π")
        t = t.replace("\\sin", "sin").replace("\\cos", "cos")
        t = t.replace("^2", "²").replace("^\\circ", "°")
        t = _re.sub(r"\\[a-zA-Z]+", "", t)
        t = t.replace("{", "").replace("}", "")
        return t
    return _re.sub(r"\$([^$]*)\$", repl, s)


def write_srt(shots, path, maxchars=46):
    def ts(t):
        h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    def split(text, limit):
        """at most two cues, balanced at a punctuation mark near the middle."""
        if len(text) <= limit:
            return [text]
        mid = len(text) // 2
        best = None
        for i, ch in enumerate(text[:-1]):
            if ch in "。，；：、":
                if best is None or abs(i - mid) < abs(best - mid):
                    best = i
        if best is None or best < 4:
            return [text[i:i + limit] for i in range(0, len(text), limit)]
        return [text[:best + 1], text[best + 1:]]

    idx = 1
    lines = []
    for sh in shots:
        for c in sh["_cues"]:
            chunks = split(_plain(c["text"]), maxchars)
            per = c["dur"] / max(len(chunks), 1)
            for j, ch in enumerate(chunks):
                a = c["start"] + j * per
                lines.append(f"{idx}\n{ts(a)} --> {ts(a + per)}\n{ch}\n")
                idx += 1
    open(path, "w", encoding="utf-8").write("\n".join(lines))
