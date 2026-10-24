"""第 4 讲：能量整定 —— 6 个镜头。"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kit import (math_, math_w, title, chip, axes, cartpole, curvy, MC, MP, L, G, UMAX)
from render import (Canvas, INK, DIM, FAINT, CY, AM, MG, BL, GR, BG,
                    appear, smooth, lerp, W, H)

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "..", "..", "build", "out", "ch03_06.json")))["ch04"]
T = np.array(D["t"]); E = np.array(D["E"]); P_ = np.array(D["p"])
SWEEP = D["kE_sweep"]


# ============================================================ S1
def s1(cv, t, dur):
    cv.text(640, 252, "倒立摆控制　·　从拉格朗日到 Transformer", size=25, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 352, "第 4 讲　能量整定", size=68, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 414, "把「摆起来」变成一个标量", size=31, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((500, 450), (780, 450), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 512, r"E=\tfrac{1}{2}m_p l^2\dot\theta^2+m_p g l(\cos\theta-1)", 24, DIM,
              anchor="mc")
    cv.text(90, 620, "课程讲义 · ch04　　代码 · controllers.py: EnergyFBL", size=19, color=FAINT,
            alpha=appear(t, 3.0, 0.8))
    cartpole(cv, 640, 600, math.pi - 0.15 + 0.05 * math.sin(2.2 * t), a=0.12 * appear(t, 0.5, 1.2),
             show_rail=False)


# ============================================================ S2
def s2(cv, t, dur):
    title(cv, "能量通道：一对共轭量")
    a = appear(t, 0.3, 0.8)
    if a > 0:
        math_(cv, 110, 236, r"\dot E=-m_p l\,(\dot\theta\cos\theta)\,v-b_p\dot\theta^2", 27, INK)
    a2 = appear(t, 1.8, 0.8)
    if a2 > 0:
        cv.rect(110, 280, 700, 360, CY, width=1.2, alpha=0.7 * a2, fill_alpha=0.06 * a2, radius=8)
        cv.text(132, 312, "v 与 θ̇cosθ 同号 → 能量上升；反号 → 下降", size=21, color=CY, alpha=a2)
    a3 = appear(t, 3.2, 0.8)
    if a3 > 0:
        cv.text(110, 420, "所以“摆起来”不再是一个轨迹问题，", size=23, color=INK, alpha=a3)
        cv.text(110, 456, "而是“控制这一对量的相位”。", size=23, color=INK, alpha=a3)
    a4 = appear(t, 4.6, 0.9)
    if a4 > 0:
        axes(cv, 800, 240, 1140, 430, a4, "t [s]", "E [J]")
        cv.line((800, 336), (1140, 336), (90, 108, 130), width=1.2, alpha=0.7 * a4, dash=6)
        cv.text(1146, 330, "0", size=16, color=FAINT, alpha=a4)
        curvy(cv, [(800 + i / 60 * 340, 336 + 90 * (1 - math.cos(2 * math.pi * i / 60)) / 2)
                   for i in range(61)], CY, 2.2, 0.8 * a4)
        cv.text(970, 470, "能量在小车里来回倒", size=18, color=DIM, anchor="ma", alpha=a4)


# ============================================================ S3
def s3(cv, t, dur):
    title(cv, "控制律与 Lyapunov 证明")
    a = appear(t, 0.3, 0.8)
    if a > 0:
        cv.text(110, 162, "控制律（加速度域）", size=19, color=FAINT, alpha=a)
        math_(cv, 110, 224, r"v=k_E(E-E_d)\,\mathrm{sat}\!\left(\frac{\dot\theta\cos\theta}{\phi}\right)"
                            r"-k_x p-k_d\dot p", 23, INK)
    a2 = appear(t, 2.0, 0.8)
    if a2 > 0:
        cv.text(110, 300, "代入能量通道（忽略阻尼与回中）", size=19, color=FAINT, alpha=a2)
        math_(cv, 110, 344, r"\dot E=-m_p l k_E(E-E_d)\,|\dot\theta\cos\theta|\,"
                            r"\big|\mathrm{sat}(\cdot)\big|", 21, INK)
    a3 = appear(t, 3.8, 0.8)
    if a3 > 0:
        cv.rect(110, 400, 700, 500, CY, width=1.4, alpha=0.8 * a3, fill_alpha=0.07 * a3, radius=8)
        math_(cv, 132, 444, r"W=\tfrac{1}{2}(E-E_d)^2\ \Rightarrow\ \dot W\le0", 22, CY)
        cv.text(132, 484, "W 单调不增 → E → E_d", size=19, color=DIM, alpha=a3)
    a4 = appear(t, 5.2, 0.8)
    cv.text(760, 420, "注意：摆不动的时候", size=20, color=AM, alpha=a4)
    cv.text(760, 452, "|θ̇cosθ| = 0，泵不进能量", size=20, color=AM, alpha=a4)


# ============================================================ S4
def s4(cv, t, dur):
    title(cv, "实测：一次完整的摆起")
    a = appear(t, 0.3, 0.8)
    X0, Y0, X1, Y1 = 120, 220, 780, 520
    axes(cv, X0, Y0, X1, Y1, a, "t [s]", "E [J]")
    sx = lambda tv: X0 + tv / 12.0 * (X1 - X0)
    sy = lambda v: Y1 - (v + 3.2) / 3.6 * (Y1 - Y0)
    for v in (-3, -2, -1, 0):
        cv.line((X0 - 5, sy(v)), (X0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8 * a)
        cv.text(X0 + 9, sy(v) + 6, str(v), size=15, color=FAINT, alpha=a)
        cv.line((X0, sy(v)), (X1, sy(v)), (78, 96, 116), width=1, alpha=0.35 * a, dash=6)
    for tv in (0, 3, 6, 9, 12):
        cv.line((sx(tv), Y1), (sx(tv), Y1 + 5), (92, 110, 132), width=1.2, alpha=0.8 * a)
        cv.text(sx(tv), Y1 + 26, str(tv), size=15, color=FAINT, anchor="ma", alpha=a)
    cv.text(X0 + 10, sy(0) - 10, "E = 0（直立）", size=16, color=CY, alpha=a)
    prog = min(1.0, max(0.0, (t - 1.0) / 4.0))
    n = max(2, int(len(T) * prog))
    curvy(cv, [(sx(T[i]), sy(E[i])) for i in range(n)], CY, 2.2, a)
    if prog >= 1:
        cv.circle(sx(1.98), sy(0.03), 4.5, AM, width=0, alpha=a, fill_alpha=a)
        cv.text(sx(2.3), sy(0.9), "1.98 s 到顶", size=18, color=AM, alpha=a)

    cards = [("−2.93 J", "下垂初始化", 2.0, MG), ("+0.026 J", "能量峰值（略过冲）", 3.0, AM),
             ("< 1e−3 rad", "稳态角度", 4.0, CY)]
    for txt, sub, t0, col in cards:
        aa = appear(t, t0, 0.6)
        if aa <= 0:
            continue
        y = 250 + (t0 - 2.0) * 90
        cv.text(840, y, txt, size=30, color=col, weight="bold", alpha=aa)
        cv.text(840, y + 26, sub, size=18, color=FAINT, alpha=aa)


# ============================================================ S5
def s5(cv, t, dur):
    title(cv, "k_E：泵多快，撞多狠")
    a = appear(t, 0.3, 0.8)
    cv.text(110, 190, "k_E", size=19, color=FAINT, alpha=a)
    cv.text(230, 190, "平均摆起时间", size=19, color=FAINT, alpha=a)
    cv.text(520, 190, "平均 |p|max", size=19, color=FAINT, alpha=a)
    for i, r in enumerate(SWEEP):
        aa = appear(t, 0.6 + 0.42 * i, 0.5)
        if aa <= 0:
            continue
        y = 236 + i * 52
        good = r["t_up"] < 15
        hit = r["p_max"] > 2.4
        col = GR if (good and not hit) else (MG if hit else DIM)
        cv.text(110, y, f"{r['kE']:.1f}", size=22, color=INK, alpha=aa)
        cv.text(230, y, (f"{r['t_up']:.2f} s" if good else "> 15 s"), size=22, color=col, alpha=aa)
        cv.text(520, y, f"{r['p_max']:.2f} m", size=22, color=(MG if hit else INK), alpha=aa)
        if abs(r["kE"] - 6.0) < 0.01:
            cv.rect(96, y - 26, 660, y + 12, CY, width=1.2, alpha=0.7 * aa,
                    fill_alpha=0.07 * aa, radius=6)
            cv.text(806, y, "← 本课选择", size=18, color=CY, alpha=aa, anchor="ra")

    a2 = appear(t, 3.4, 0.8)
    cv.text(836, 300, "太小：泵不动", size=21, color=DIM, alpha=a2)
    cv.text(836, 340, "太大：撞墙", size=21, color=MG, alpha=a2)
    cv.text(836, 400, "k_E = 6 是“刚好能在", size=21, color=INK, alpha=a2)
    cv.text(836, 432, "导轨内完成一次摆起”", size=21, color=INK, alpha=a2)


# ============================================================ S6
def s6(cv, t, dur):
    title(cv, "本讲小结")
    items = [("能量通道", "Ė = −m l (θ̇cosθ) v − b_p θ̇²"),
             ("Lyapunov", "W = ½(E−E_d)² 单调不增 → E → E_d"),
             ("两个不变集", "直立（目标）与下垂（陷阱）"),
             ("回中要慢", "k_x 太弱会跑、太强会撞墙"),
             ("k_E 要刚好", "6.0：1.98 s 到顶，|p|max = 2.37 m"),
             ("下一步", "到顶后能量会过冲，需要局部稳定器接手")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.66 * i, 0.55)
        if aa <= 0:
            continue
        y = 190 + i * 56
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=22, color=CY, alpha=aa, weight="bold")
        cv.text(290, y, v, size=21, color=INK, alpha=aa)
    a = appear(t, 4.6, 0.8)
    cv.text(130, 580, "下一讲：滑模 —— 把鲁棒性做进结构", size=25, color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "上一讲把力换成了加速度。这一讲用它做一件看起来很难的事：让摆自己甩上去。",
        "办法是换掉目标：不再规定轨迹，只规定一个标量——能量。",
        "能量到零，摆杆就必然经过竖直位置。",
    ], draw=s1, min_dur=12.0),
    dict(name="channel", say=[
        "先看能量对时间的导数。",
        {"say": "它等于 $-m_p l$ 乘上 $\\dot\\theta\\cos\\theta$，再乘上加速度指令，减去阻尼耗散。",
         "tts": "它等于负的 m p l，乘上 theta 点乘 cos theta，再乘上加速度指令，减去阻尼耗散。"},
        {"say": "所以 $v$ 和 $\\dot\\theta\\cos\\theta$ 是一对共轭量：同号就泵能量，反号就抽能量。",
         "tts": "所以 v 和 theta 点乘 cos theta 是一对共轭量：同号就泵能量，反号就抽能量。"},
        "摆起来，从此不是轨迹问题，而是相位问题。",
    ], draw=s2, min_dur=19.0),
    dict(name="law", say=[
        "控制律就是让加速度跟着这对量的符号走，再叠加一个小车回中项。",
        {"say": "取 $W=\\tfrac{1}{2}(E-E_d)^2$，它的导数恒小于等于零。",
         "tts": "取 W 等于二分之一乘以能量误差的平方，它的导数恒小于等于零。"},
        {"say": "所以能量一定收敛到目标值。但注意：摆不动的时候，这个泵是无效的。",
         "tts": "所以能量一定收敛到目标值。但注意：摆不动的时候，这个泵是无效的。"},
    ], draw=s3, min_dur=19.0),
    dict(name="result", say=[
        "看一次真实的摆起过程。",
        {"say": "从下垂出发，能量是负的二点九三焦耳；一路爬到零附近，用时 $1.98$ 秒。",
         "tts": "从下垂出发，能量是负的二点九三焦耳；一路爬到零附近，用时一点九八秒。"},
        {"say": "注意峰值是正的 $0.026$ 焦耳：能量控制会把摆抛过头一点点。",
         "tts": "注意峰值是正的零点零二六焦耳：能量控制会把摆抛过头一点点。"},
        "这正是为什么必须有一个局部稳定器在后面接住它。",
    ], draw=s4, min_dur=19.0),
    dict(name="gain", say=[
        "泵的增益不能随便给。",
        "给得太小，十五秒都甩不上去；给得太大，车先撞墙。",
        {"say": "$k_E=6$ 是刚好能在导轨内完成一次摆起的临界点。",
         "tts": "k E 等于六，是刚好能在导轨内完成一次摆起的临界点。"},
        "这也是这一讲最花时间调的一个数。",
    ], draw=s5, min_dur=17.0),
    dict(name="summary", say=[
        "总结：一个标量、一个 Lyapunov 函数、一条平滑的能量曲线。",
        "但摆起来只是前半程，到了顶上它停不住。",
        "下一讲解决另一半问题：怎么顶住扰动。",
    ], draw=s6, min_dur=14.0),
]
