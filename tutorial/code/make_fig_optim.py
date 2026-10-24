"""生成 fig_optim.png：左=二维病态曲面轨迹，右=真实模型损失曲线。"""
import json, math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
D = json.load(open(os.path.join(OUT, "optim.json")))
F = "/System/Library/Fonts/Songti.ttc"
INK, DIM, CY, AM, BL, GR, MG = (28,34,44), (90,104,120), (24,160,148), (200,130,20), (60,110,180), (40,150,90), (190,60,110)
GRAY = (120,130,145)
def f(s, i=6): return ImageFont.truetype(F, s, index=i)

W, H = 1400, 620
img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
d.text((40, 18), "优化器对比：左为二维病态曲面（κ=50）上的真实轨迹，右为本项目模型上的训练损失",
       font=f(20, 1), fill=INK)

# ---------- 左：等高线 + 轨迹
X0, Y0, WW, HH = 70, 80, 540, 440
d.rectangle([X0, Y0, X0+WW, Y0+HH], outline=(170,178,190), width=1)
sx = lambda x: X0 + (x + 1.2) / 2.4 * WW
sy = lambda y: Y0 + HH - (y + 2.0) / 4.0 * HH
K = D["kappa"]
for lv in (0.02, 0.08, 0.25, 0.7, 1.8, 4.5, 10.0):
    pts = []
    for k in range(97):
        th = 2*math.pi*k/96
        pts.append((sx(math.sqrt(2*lv)*math.cos(th)), sy(math.sqrt(2*lv/K)*math.sin(th))))
    d.line(pts, fill=(215, 220, 228), width=1)
cols = {"sgd": GRAY, "momentum": BL, "rmsprop": AM, "adam": CY}
for k, c in cols.items():
    pts = [(sx(p[0]), sy(p[1])) for p in D["toy"][k]["path"]]
    d.line(pts, fill=c, width=3)
    d.ellipse([pts[0][0]-4, pts[0][1]-4, pts[0][0]+4, pts[0][1]+4], fill=c)
d.text((X0+10, Y0+HH+10), "起点 (1,1)，60 步", font=f(15), fill=DIM)
# 图例放到左图右上角（椭圆之外，不与轨迹冲突）
lx = X0 + WW - 176
d.text((lx, Y0 + 12), "60 步后损失", font=f(14), fill=DIM)
for i, (k, c) in enumerate(cols.items()):
    y = Y0 + 40 + i * 34
    d.line([lx, y + 7, lx + 24, y + 7], fill=c, width=3)
    d.text((lx + 32, y), f"{k}", font=f(14), fill=INK)
    d.text((lx + 32, y + 16), f"{D['toy'][k]['loss'][-1]:.1e}", font=f(13), fill=DIM)

# ---------- 右：真实模型曲线
X1, Y1, W2, H2 = 700, 80, 640, 440
d.rectangle([X1, Y1, X1+W2, Y1+H2], outline=(170,178,190), width=1)
lo, hi = 1e-2, 100
px = lambda ep: X1 + 30 + ep / 2.0 * (W2 - 70)
py = lambda v: Y1 + H2 - 30 - (math.log10(min(max(v, lo), hi)) - math.log10(lo)) / (math.log10(hi)-math.log10(lo)) * (H2 - 60)
for v in (1e-2, 1e-1, 1, 10, 100):
    d.line([X1+20, py(v), X1+W2-20, py(v)], fill=(232,236,242), width=1)
    d.text((X1+18, py(v)), f"{v:g}", font=f(13), fill=DIM, anchor="rm")
names = [("adam", CY), ("rmsprop", AM), ("sgd", GRAY), ("momentum", BL), ("sgd_big", MG)]
for nm, c in names:
    L = D["real"][nm]["loss"]
    pts = [(px(i), py(v)) for i, v in enumerate(L)]
    d.line(pts, fill=c, width=3 if nm == "adam" else 2)
    for x, y in pts: d.ellipse([x-3, y-3, x+3, y+3], fill=c)
for i in range(3):
    d.text((px(i), Y1+H2-16), str(i+1), font=f(14), fill=DIM, anchor="ma")
d.text((X1+W2/2, Y1+H2+4), "epoch", font=f(15), fill=DIM, anchor="ma")
lx = X1 + W2 - 250
for i, (nm, c) in enumerate(names):
    lab = {"adam":"Adam/AdamW", "rmsprop":"RMSProp", "sgd":"SGD 1e-3",
           "momentum":"Momentum 1e-3", "sgd_big":"SGD 2e-2（发散）"}[nm]
    y = Y1 + 40 + i * 30
    d.line([lx, y, lx+22, y], fill=c, width=3)
    d.text((lx+30, y-8), lab, font=f(14), fill=DIM)
img.save(os.path.join(OUT, "fig_optim.png"))
print("wrote fig_optim.png")
