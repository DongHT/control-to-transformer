"""扫描所有场景里用到的 LaTeX 公式，找出 KaTeX 渲染失败（红色）或缺字的那些。"""
import importlib
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scenes"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "course", "video"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "course", "video", "scenes"))

import mathrender as mr           # noqa: E402

SEEN = {}


def record(tex, size, color):
    SEEN.setdefault((tex, size, color), 0)
    SEEN[(tex, size, color)] += 1


def main():
    import kit
    # patch the helpers so every formula used by a scene is registered
    orig_kit = kit.math_

    def patched(cv, x, y, tex, size=26, color=kit.INK, anchor="ls", check=True, tag=None):
        record(tex, size, "#%02x%02x%02x" % color if isinstance(color, tuple) else color)
        return orig_kit(cv, x, y, tex, size, color, anchor, check, tag)

    kit.math_ = patched
    for m in ("ch01", "ch02", "ch03", "ch04", "ch05", "ch06",
              "ch07", "ch08", "ch09", "ch10", "chA"):
        mod = importlib.import_module(m)
        if hasattr(mod, "math_"):
            mod.math_ = patched
        # replay a few frames so every shot draws
        from render import Canvas
        for sh in mod.SHOTS:
            for tt in (0.8, 0.5 * sh.get("min_dur", 10), sh.get("min_dur", 10) - 0.5):
                cv = Canvas()
                try:
                    sh["draw"](cv, tt, sh.get("min_dur", 10))
                except Exception as e:
                    print(f"[draw-error] {m}/{sh['name']}: {e}")

    print(f"collected {len(SEEN)} unique (tex,size,color) formulas\n")

    bad = []
    for (tex, size, color), n in sorted(SEEN.items(), key=lambda kv: -kv[1]):
        try:
            img = mr.render(tex, size, color)
        except Exception as e:
            bad.append((tex, size, "exception: " + str(e)[:80]))
            continue
        a = np.asarray(img.convert("RGB")).astype(int)
        if a.size == 0:
            bad.append((tex, size, "empty image"))
            continue
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        # KaTeX renders parse errors with colour #cc0000
        red = ((r > 150) & (g < 70) & (b < 70)).sum()
        if red > 20:
            bad.append((tex, size, f"parse-error red px={red}"))

    if not bad:
        print("no KaTeX failures detected")
        return
    print(f"!! {len(bad)} problematic formulas:\n")
    for tex, size, why in bad:
        print(f"  [{why}]  size={size}\n    {tex[:160]}")


if __name__ == "__main__":
    main()
