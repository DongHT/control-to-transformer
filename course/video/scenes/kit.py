"""各讲共用的绘图组件（第 2 讲起使用）。"""
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from render import (INK, DIM, FAINT, CY, AM, MG, BL, GR, BG, appear, smooth, lerp, W, H)
import mathrender as mr

MC, MP, L, G, UMAX = 1.0, 0.25, 0.6, 9.81, 12.0
SCALE = 190.0
RAIL = 1.4


def math_(cv, x, y, tex, size=26, color=INK, anchor="ls", check=True, tag=None):
    box = mr.paste(cv, x, y, tex, size=size, color="#%02x%02x%02x" % color,
                   anchor=anchor, ss=cv.ss)
    if check:
        cv.boxes.append((box[0], box[1], box[2], box[3], tag or tex[:16]))
    return box


def math_w(tex, size):
    return mr.render(tex, size * 2, "#ffffff").width / 2.0


def title(cv, text, sub=None):
    cv.text(90, 86, text, size=40, color=INK, weight="bold")
    cv.line((90, 108), (150, 108), CY, width=4)
    if sub:
        cv.text(90, 138, sub, size=21, color=FAINT)


def chip(cv, x, y, text, color, a=1.0, size=21):
    w = cv.text_w(text, size) + 28
    cv.rect(x, y - size * 0.95, x + w, y + 10, color, width=1,
            alpha=0.55 * a, fill_alpha=0.10 * a, radius=6)
    cv.text(x + 14, y, text, size=size, color=color, alpha=a)
    return w


def axes(cv, x0, y0, x1, y1, a=1.0, xlabel=None, ylabel=None, color=(92, 110, 132)):
    cv.arrow((x0, y1), (x1 + 14, y1), color, width=1.4, alpha=0.85 * a)
    cv.arrow((x0, y1), (x0, y0 - 14), color, width=1.4, alpha=0.85 * a)
    if xlabel:
        cv.text(x1 + 18, y1 + 6, xlabel, size=17, color=DIM, alpha=a)
    if ylabel:
        cv.text(x0 + 9, y0 - 6, ylabel, size=17, color=DIM, alpha=a)


def cartpole(cv, x, y0, th, p=0.0, u=0.0, a=1.0, show_rail=True, scale=None, rail=None):
    sc = scale if scale else SCALE
    rl = rail if rail else RAIL
    if show_rail:
        cv.line((x - rl * sc, y0), (x + rl * sc, y0), (110, 130, 150), width=3, alpha=0.8 * a)
        for s in (-1, 1):
            cv.line((x + s * rl * sc, y0 - 26), (x + s * rl * sc, y0 + 8),
                    MG, width=2, alpha=0.8 * a, dash=5)
    cx = x + p * sc
    cw, ch = 66, 34
    ytop = y0 - ch - 6
    cv.rect(cx - cw / 2, ytop, cx + cw / 2, y0 - 6, (60, 90, 118), width=2,
            alpha=a, fill_alpha=0.55 * a, radius=5)
    for s in (-1, 1):
        cv.circle(cx + s * cw * 0.28, y0 - 4, 6, BL, width=1.5, alpha=0.9 * a)
    px, py = cx, ytop
    tx = px + math.sin(th) * L * sc
    ty = py - math.cos(th) * L * sc
    cv.line((px, py), (tx, ty), CY, width=5, alpha=a)
    cv.circle(tx, ty, 9, CY, width=2, alpha=a, fill_alpha=0.35 * a)
    cv.circle(px, py, 3.2, INK, width=0, alpha=a, fill_alpha=a)
    cv.line((px, py), (px, py - L * sc * 1.35), (110, 130, 150), width=1, alpha=0.5 * a, dash=5)
    if abs(u) > 0.2:
        ln = min(1.0, abs(u) / UMAX) * 86 * (1 if u > 0 else -1)
        cv.arrow((cx, y0 - ch / 2 - 6), (cx + ln, y0 - ch / 2 - 6), AM, width=2.5, alpha=a)
    return cx, ytop


def curvy(cv, pts, color, width=2.0, alpha=1.0, dash=None):
    """draw a polyline; pts in design px."""
    for i in range(len(pts) - 1):
        cv.line(pts[i], pts[i + 1], color, width=width, alpha=alpha, dash=dash)


def basin_image(data, w, h, ss=2):
    """pre-render the (theta, omega) success map as an RGBA image."""
    from PIL import Image, ImageDraw
    ths = np.array(data["ths"]); ws = np.array(data["ws"])
    ok = np.array(data["ok"]).astype(bool)
    img = Image.new("RGBA", (int(w * ss), int(h * ss)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    nw, nth = ok.shape
    cw = w * ss / (nth - 1)
    chh = h * ss / (nw - 1)
    good = (int(0.16 * 255), int(0.80 * 255), int(0.74 * 255), 78)
    bad = (int(0.80 * 255), int(0.28 * 255), int(0.52 * 255), 30)
    for j in range(nw - 1):
        for i in range(nth - 1):
            col = good if ok[j, i] else bad
            d.rectangle([i * cw, (nw - 2 - j) * chh, (i + 1) * cw + 1, (nw - 1 - j) * chh + 1],
                        fill=col)
    return img
