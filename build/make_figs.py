"""生成第 2 讲讲义用的图（PNG，供 xelatex \\includegraphics）。"""
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import plant as pl
import controllers as C
from dataset import DT
from plant import P, wrap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "..", "course", "latex", "figures")
os.makedirs(FIG, exist_ok=True)

FONT = "/System/Library/Fonts/Songti.ttc"
INK, DIM, CY, AM, MG = (28, 34, 44), (90, 104, 120), (24, 160, 148), (200, 130, 20), (190, 60, 110)


def f(size, idx=6):
    return ImageFont.truetype(FONT, size, index=idx)


def nice_ticks(lo, hi, n=4):
    span = hi - lo
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    start = np.ceil(lo / step) * step
    out = []
    v = start
    while v <= hi + 1e-9:
        out.append(round(float(v), 6)); v += step
    return out


def vtext(img, xy, text, fnt, fill):
    """vertical (rotated 90°) text, centred at xy."""
    tmp = Image.new("RGBA", (10, 10))
    w = int(fnt.getlength(text)) + 20
    h = int(fnt.size * 1.6)
    tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((w / 2, h / 2), text, font=fnt, fill=fill, anchor="mm")
    tmp = tmp.rotate(90, expand=True)
    img.paste(tmp, (int(xy[0] - tmp.width / 2), int(xy[1] - tmp.height / 2)), tmp)


def basin_fig(data, path):
    W, H = 1400, 900
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ths = np.array(data["ths"]); ws = np.array(data["ws"])
    ok = np.array(data["ok"]).astype(bool)
    L, R, T, B = 150, 70, 80, 110
    x0, x1 = ths.min(), ths.max()
    y0, y1 = ws.min(), ws.max()
    px = lambda th: L + (th - x0) / (x1 - x0) * (W - L - R)
    py = lambda w: H - B - (w - y0) / (y1 - y0) * (H - T - B)
    # cells
    for j in range(len(ws) - 1):
        for i in range(len(ths) - 1):
            good = ok[j, i]
            col = (206, 236, 232) if good else (246, 226, 232)
            d.rectangle([px(ths[i]), py(ws[j + 1]), px(ths[i + 1]), py(ws[j])], fill=col)
    d.rectangle([L, T, W - R, H - B], outline=(120, 132, 148), width=2)
    for th in (-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2):
        d.line([px(th), H - B, px(th), H - B + 8], fill=(120, 132, 148), width=2)
        d.text((px(th), H - B + 14), f"{th:g}", font=f(26), fill=DIM, anchor="ma")
    for w in (-4, -2, 0, 2, 4):
        d.line([L - 8, py(w), L, py(w)], fill=(120, 132, 148), width=2)
        d.text((L - 14, py(w)), f"{w:g}", font=f(26), fill=DIM, anchor="rm")
    d.text(((L + W - R) / 2, H - 34), "初始倾角 θ0 [rad]", font=f(30), fill=INK, anchor="mm")
    vtext(img, (52, (T + H - B) / 2), "初始角速度 ω0 [rad/s]", f(30), INK)
    lx, ly = L + 24, T + 22
    d.rectangle([lx, ly, lx + 26, ly + 18], fill=(206, 236, 232), outline=(160, 180, 190))
    d.text((lx + 36, ly + 9), "可拉回直立", font=f(26), fill=DIM, anchor="lm")
    lx2 = lx + 190
    d.rectangle([lx2, ly, lx2 + 26, ly + 18], fill=(246, 226, 232), outline=(190, 170, 180))
    d.text((lx2 + 36, ly + 9), "失败（撞墙或回不来）", font=f(26), fill=DIM, anchor="lm")
    img.save(path)
    print("wrote", path)


def response_fig(K, path):
    """two panels: theta(t) for 0.1 / 0.25 rad, and the force."""
    runs = {}
    for a in (0.1, 0.25):
        x = np.array([0.0, a, 0.0, 0.0])
        ts, th, us = [], [], []
        for k in range(200):
            u = float(np.clip(-K @ x, -P["u_max"], P["u_max"]))
            ts.append(k * DT); th.append(np.degrees(x[1])); us.append(u)
            x = pl.rk4(x, u, DT)
        runs[a] = (np.array(ts), np.array(th), np.array(us))

    W, H = 1400, 760
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    # panel 1: theta
    L, R, T = 110, 50, 60
    h1, h2 = 300, 240
    for pi, (ylab, key, ylim) in enumerate([("θ [°]", 1, None), ("u [N]", 2, (-12.5, 12.5))]):
        top = T + pi * 360
        bot = top + (h1 if pi == 0 else h2)
        d.rectangle([L, top, W - R, bot], outline=(140, 150, 165), width=2)
        d.text((L, top - 34), ylab, font=f(28), fill=INK)
        if key == 1:
            ymin, ymax = 0, 16
        else:
            ymin, ymax = ylim
        py = lambda v: bot - (v - ymin) / (ymax - ymin) * (bot - top)
        px = lambda t: L + t / 1.6 * (W - L - R)
        for v in nice_ticks(ymin, ymax, 4):
            d.line([L - 8, py(v), L, py(v)], fill=(140, 150, 165), width=2)
            d.text((L - 14, py(v)), f"{v:g}", font=f(24), fill=DIM, anchor="rm")
        for t in (0, 0.4, 0.8, 1.2, 1.6):
            d.line([px(t), bot, px(t), bot + 8], fill=(140, 150, 165), width=2)
            d.text((px(t), bot + 12), f"{t:g}", font=f(24), fill=DIM, anchor="ma")
        for a, col in ((0.1, CY), (0.25, MG)):
            ts, th, us = runs[a]
            ys = th if key == 1 else us
            pts = [(px(ts[i]), py(min(max(ys[i], ymin), ymax))) for i in range(len(ts))
                   if ts[i] <= 1.6]
            d.line(pts, fill=col, width=3)
        if pi == 0:
            d.text((W - R - 10, top + 12), "θ0 = 0.1 rad", font=f(26), fill=CY, anchor="ra")
            d.text((W - R - 10, top + 46), "θ0 = 0.25 rad", font=f(26), fill=MG, anchor="ra")
    d.text(((L + W - R) / 2, H - 40), "时间 t [s]", font=f(30), fill=INK, anchor="mm")
    img.save(path)
    print("wrote", path)


def main():
    data = json.load(open(os.path.join(OUT, "ch02.json")))
    basin_fig(data["basin"], os.path.join(FIG, "basin.png"))
    K = np.array(data["design"]["K"])
    response_fig(K, os.path.join(FIG, "response.png"))


if __name__ == "__main__":
    main()
