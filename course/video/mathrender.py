"""
LaTeX 公式渲染器：KaTeX + 无头 Chrome -> 带透明通道的 PNG（带缓存）。

为什么不用 LaTeX 本体：本机没有 TeX 引擎，且 TeX Live 体积过大无法下载。
KaTeX 是纯前端 LaTeX 排版器，字体与度量都是 Knuth 系的，视觉上与 LaTeX 一致，
并且支持 bmatrix / frac / dot 等本项目需要的语法。
"""
import hashlib
import re
import json
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets", "katex")
CACHE = os.path.join(HERE, "assets", "mathcache")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="file://{css}">
<style>
  html,body{{margin:0;padding:0;background:transparent;}}
  .wrap{{display:inline-block;padding:8px 10px;color:{color};font-size:{size}px;}}
</style>
<script src="file://{js}"></script>
</head><body>
<div class="wrap" id="f"></div><span id="mk" style="display:inline-block;width:0;height:0;vertical-align:baseline"></span>
<script>
  katex.render({tex}, document.getElementById('f'),
               {{displayMode:true, throwOnError:false, output:'html'}});
  var r = document.getElementById('f').getBoundingClientRect();
  var b = document.getElementById('mk').getBoundingClientRect();
  document.body.setAttribute('data-geom',
      [r.left, r.top, r.width, r.height, (b.top - r.top)].join(','));
</script>
</body></html>"""


def _key(tex, size, color):
    h = hashlib.sha1(f"{tex}|{size}|{color}".encode()).hexdigest()[:16]
    return os.path.join(CACHE, f"m_{h}.png")


def _baseline_frac(png_path, geom):
    """baseline position as a fraction of the alpha-cropped image height."""
    img = Image.open(png_path).convert("RGBA")
    a = np.asarray(img)[:, :, 3]
    ys, xs = np.where(a > 6)
    if len(ys) == 0:
        return 0.75
    top = ys.min()
    h = ys.max() - top + 1
    bl = geom[4] + geom[1] - top          # page coords
    return float(np.clip(bl / h, 0.2, 1.15))


def render(tex, size=40, color="#EAF2F8"):
    """Return a PIL RGBA image of the formula (cropped to its ink)."""
    os.makedirs(CACHE, exist_ok=True)
    path = _key(tex, size, color)
    meta = path.replace(".png", ".json")
    if os.path.exists(path) and os.path.exists(meta):
        return Image.open(path).convert("RGBA")

    html = HTML.format(css=os.path.join(ASSETS, "katex.min.css"),
                       js=os.path.join(ASSETS, "katex.min.js"),
                       color=color, size=size, tex=json.dumps(tex))
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                     encoding="utf-8") as f:
        f.write(html)
        tmp_html = f.name
    shot = tmp_html.replace(".html", ".png")

    dom = subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
                          "--allow-file-access-from-files",
                          "--default-background-color=00000000",
                          "--window-size=2000,700", f"--screenshot={shot}",
                          "--dump-dom", "--virtual-time-budget=1500", "file://" + tmp_html],
                         capture_output=True, text=True, timeout=60).stdout
    m = re.search(r'data-geom="([-\d.,]+)"', dom or "")
    geom = [float(v) for v in m.group(1).split(",")] if m else [0, 0, 0, 0, 0.72]
    img = Image.open(shot).convert("RGBA")
    os.unlink(tmp_html)
    os.unlink(shot)
    a = np.asarray(img)[:, :, 3]
    ys, xs = np.where(a > 6)
    if len(xs) == 0:
        return Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    ink_top, ink_h = int(ys.min()), int(ys.max() - ys.min() + 1)
    frac = float(np.clip((geom[1] + geom[4] - ink_top) / ink_h, 0.2, 1.15))
    img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    img.save(path)
    json.dump(dict(baseline=frac, ink_top=ink_top, ink_h=ink_h), open(meta, "w"))
    return img


def baseline_of(tex, size, color="#EAF2F8"):
    render(tex, size, color)                    # ensures cache
    p = _key(tex, size, color).replace(".png", ".json")
    try:
        return json.load(open(p))["baseline"]
    except Exception:
        return 0.72


def paste(canvas, x, y, tex, size=40, color="#EAF2F8", anchor="ls", ss=2, baseline=0.72, alpha=1.0):
    """Paste a formula onto the supersampled canvas.

    anchor: 'ls' = left baseline, 'ms' = middle baseline, 'rs' = right baseline,
            'lc'/'mc'/'rc' = vertical centre instead of baseline.
    """
    img = render(tex, size * ss, color)
    if alpha < 0.999:
        a = img.getchannel("A").point(lambda v: int(v * max(0.0, alpha)))
        img.putalpha(a)
    w = img.width
    dy = int(img.height * baseline_of(tex, size * ss, color))
    # 与 Canvas.text 一致：第一个字符 m/c 都表示水平居中
    if anchor[0] in ("c", "m"):
        px = x * ss - w // 2
    elif anchor[0] == "r":
        px = x * ss - w
    else:
        px = int(x * ss)
    if anchor[1] == "c":
        py = int(y * ss) - img.height // 2
    else:
        py = int(y * ss) - dy
    canvas.img.alpha_composite(img, (px, py)) if canvas.img.mode == "RGBA" else \
        canvas.img.paste(img, (px, py), img)
    box = (px / ss, py / ss, (px + img.width) / ss, (py + img.height) / ss)
    return box
