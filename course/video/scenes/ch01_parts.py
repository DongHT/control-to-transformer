"""第 1 讲：建模 —— 场景定义（画面 + 旁白 + 字幕）。

所有文案同时用于：TTS 旁白、底部字幕、以及（在 manim 版里）同一份脚本。
"""
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from render import (Canvas, W, H, SAFE_BOTTOM, INK, DIM, FAINT, CY, AM, MG, BL, GR,
                    BG, BG2, mix, ease, smooth, appear, lerp, font)

MC, MP, L, G, UMAX = 1.0, 0.25, 0.6, 9.81, 12.0
RAIL = 1.4          # metres shown on each side in the diagram
SCALE = 190.0        # px per metre


# ----------------------------------------------------------------- helpers
def title(cv, text, sub=None):
    cv.text(90, 86, text, size=40, color=INK, weight="bold")
    cv.line((90, 108), (150, 108), CY, width=4)
    if sub:
        cv.text(90, 136, sub, size=22, color=FAINT)


def chip(cv, x, y, text, color, a=1.0, size=22):
    w = cv.text_w(text, size) + 28
    cv.rect(x, y - size * 0.95, x + w, y + 10, color, width=1,
            alpha=0.55 * a, fill_alpha=0.10 * a, radius=6)
    cv.text(x + 14, y, text, size=size, color=color, alpha=a)
    return w


def matrix(cv, x, y, rows, colw=86, rowh=40, size=24, color=INK, a=1.0, ncol=None):
    ncol = ncol or len(rows[0])
    w = ncol * colw
    h = len(rows) * rowh
    cv.line((x - 12, y - 6), (x - 12, y + h), color, width=2, alpha=0.8 * a)
    cv.line((x - 12, y - 6), (x - 2, y - 6), color, width=2, alpha=0.8 * a)
    cv.line((x - 12, y + h), (x - 2, y + h), color, width=2, alpha=0.8 * a)
    cv.line((x + w + 12, y - 6), (x + w + 12, y + h), color, width=2, alpha=0.8 * a)
    cv.line((x + w + 12, y - 6), (x + w + 2, y - 6), color, width=2, alpha=0.8 * a)
    cv.line((x + w + 12, y + h), (x + w + 2, y + h), color, width=2, alpha=0.8 * a)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if cell:
                cv.text(x + colw * (j + 0.5), y + rowh * (i + 1) - 8,
                        str(cell), size=size, color=color, anchor="ma",
                        alpha=a, check=False)


def cartpole(cv, x, y0, th, p=0.0, u=0.0, a=1.0, labels=False, show_rail=True):
    """Draw the cart-pole at (x = rail centre) / (y0 = rail line)."""
    if show_rail:
        cv.line((x - RAIL * SCALE, y0), (x + RAIL * SCALE, y0), (110, 130, 150), width=3, alpha=0.8 * a)
        for s in (-1, 1):
            cv.line((x + s * RAIL * SCALE, y0 - 26), (x + s * RAIL * SCALE, y0 + 8),
                    MG, width=1, alpha=0.5 * a, dash=5)
    cx = x + p * SCALE
    cw, ch = 66, 34
    ytop = y0 - ch - 6
    cv.rect(cx - cw / 2, ytop, cx + cw / 2, y0 - 6, (60, 90, 118), width=2,
            alpha=a, fill_alpha=0.55 * a, radius=5)
    for s in (-1, 1):
        cv.circle(cx + s * cw * 0.28, y0 - 4, 6, BL, width=1.5, alpha=0.9 * a)
    # pole
    px, py = cx, ytop
    tx = px + math.sin(th) * L * SCALE
    ty = py - math.cos(th) * L * SCALE
    cv.line((px, py), (tx, ty), CY, width=5, alpha=a)
    cv.circle(tx, ty, 9, CY, width=2, alpha=a, fill_alpha=0.35 * a)
    cv.circle(px, py, 3.2, INK, width=0, alpha=a, fill_alpha=a)
    # vertical reference
    cv.line((px, py), (px, py - L * SCALE * 1.35), (110, 130, 150), width=1, alpha=0.55 * a, dash=5)
    # theta arc (approximate with a polyline)
    r = L * SCALE * 0.55
    pts = []
    for i in range(13):
        ang = math.radians(90 - i * (math.degrees(th) / 12))
        pts.append((px + r * math.cos(ang), py - r * math.sin(ang)))
    for i in range(len(pts) - 1):
        cv.line(pts[i], pts[i + 1], AM, width=2, alpha=a)
    if abs(th) > 0.06:
        mid = th / 2
        cv.text(px + (r + 20) * math.sin(mid), py - (r + 20) * math.cos(mid) + 8,
                "θ", size=26, color=AM, alpha=a, check=labels)
    if abs(u) > 0.2:
        ln = min(1.0, abs(u) / UMAX) * 90 * (1 if u > 0 else -1)
        cv.arrow((cx, y0 - ch / 2 - 6), (cx + ln, y0 - ch / 2 - 6), AM, width=2.5, alpha=a)
    if labels:
        cv.text(cx, ytop + ch / 2 - 4, "m", size=20, color=INK, anchor="ma", alpha=a)
        cv.text(tx + 14, ty - 4, "m", size=20, color=CY, alpha=a)
        cv.line((px, y0 + 16), (px + p * SCALE, y0 + 16), BL, width=1.5, alpha=0.8 * a)
    return cx, ytop


def plot_axes(cv, x0, y0, x1, y1, color=(70, 88, 106), a=0.7):
    cv.line((x0, y1), (x1, y1), color, width=1.5, alpha=a)
    cv.line((x0, y1), (x0, y0), color, width=1.5, alpha=a)
