"""第 3 讲：部分反馈线性化 —— 6 个镜头。"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kit import (math_, math_w, title, chip, axes, cartpole, curvy,
                 MC, MP, L, G, UMAX, SCALE, RAIL)
from render import (Canvas, INK, DIM, FAINT, CY, AM, MG, BL, GR, BG,
                    appear, smooth, lerp, W, H)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "..", "..", "..", "build", "out", "ch03_06.json")))["ch03"]


# ============================================================ S1
def s1(cv, t, dur):
    cv.text(640, 252, "倒立摆控制　·　从拉格朗日到 Transformer", size=25, color=CY,
            anchor="ma", alpha=appear(t, 0.2, 0.9))
    cv.text(640, 352, "第 3 讲　反馈线性化", size=66, color=INK, anchor="ma",
            weight="bold", alpha=appear(t, 0.9, 0.9))
    cv.text(640, 414, "把「力」变成「加速度」", size=31, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((500, 450), (780, 450), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 512, r"\ddot p = v \quad\Longleftarrow\quad u = f^{-1}(x,v)",
              25, DIM, anchor="mc")
    cv.text(90, 620, "课程讲义 · ch03　　代码 · controllers.py", size=19, color=FAINT,
            alpha=appear(t, 3.0, 0.8))
    cartpole(cv, 640, 600, 0.05 * math.sin(0.8 * t), a=0.12 * appear(t, 0.5, 1.2), show_rail=False)


# ============================================================ S2
def s2(cv, t, dur):
    title(cv, "同一台机器，两种输入坐标")
    a = appear(t, 0.3, 0.8)
    # left: force domain
    cv.rect(96, 180, 560, 470, (70, 86, 104), width=1.2, alpha=0.8 * a, radius=8)
    cv.text(120, 214, "力域：u 直接进系统", size=22, color=AM, alpha=a)
    math_(cv, 120, 268, r"(m_c+m_p\sin^2\theta)\,\ddot p = u", 20, INK)
    math_(cv, 120, 302, r"\quad -b_c\dot p + m_p l\sin\theta\,\dot\theta^2", 19, INK)
    math_(cv, 120, 332, r"\quad +b_p\cos\theta\,\dot\theta - m_p g\sin\theta\cos\theta", 19, INK)
    cv.text(120, 390, "摩擦 / 科氏 / 重力耦合全混在一起", size=19, color=DIM, alpha=a)
    cv.text(120, 424, "每换一次控制律就得重推一遍", size=19, color=DIM, alpha=a)

    a2 = appear(t, 2.0, 0.8)
    cv.arrow((596, 326), (652, 326), CY, width=2.4, alpha=a2)
    cv.text(624, 300, "反解", size=17, color=CY, anchor="ma", alpha=a2)

    a3 = appear(t, 3.0, 0.8)
    cv.rect(688, 180, 1184, 470, (70, 86, 104), width=1.2, alpha=0.8 * a3, radius=8)
    cv.text(712, 214, "加速度域：v 是干净输入", size=22, color=CY, alpha=a3)
    math_(cv, 712, 268, r"\ddot p = v", 26, INK)
    math_(cv, 712, 330, r"\ddot\theta=\frac{g\sin\theta-v\cos\theta}{l}"
                        r"-\frac{b_p\dot\theta}{m_p l^2}", 22, INK)
    cv.text(712, 420, "小车通道变成二重积分器", size=19, color=DIM, alpha=a3)
    cv.text(712, 454, "摆杆通道只留下一个非线性方程", size=19, color=DIM, alpha=a3)


# ============================================================ S3
def s3(cv, t, dur):
    title(cv, "反解：一个精确的换元")
    a = appear(t, 0.3, 0.8)
    if a > 0:
        cv.text(110, 190, "把“指定的加速度”翻译成“需要的力”", size=21, color=FAINT, alpha=a)
        math_(cv, 110, 250, r"u=\big(m_c+m_p\sin^2\theta\big)\,v", 24, INK)
        math_(cv, 110, 300, r"\qquad+\,b_c\dot p-m_p l\sin\theta\,\dot\theta^2", 21, INK)
        math_(cv, 110, 336, r"\qquad-\,b_p\cos\theta\,\dot\theta+m_p g\sin\theta\cos\theta", 21, INK)
    a2 = appear(t, 2.4, 0.8)
    if a2 > 0:
        cv.rect(96, 384, 640, 470, CY, width=1.2, alpha=0.7 * a2, fill_alpha=0.06 * a2, radius=8)
        cv.text(120, 416, "分母恒正 → 对任意 v 都有唯一解", size=21, color=CY, alpha=a2)
        math_(cv, 120, 452, r"m_c+m_p\sin^2\theta\in[1.00,\ 1.25]", 19, INK)
    a3 = appear(t, 4.0, 0.8)
    if a3 > 0:
        cv.text(700, 250, "数值验证（200 组随机状态）", size=20, color=FAINT, alpha=a3)
        cv.text(700, 300, "9.2e−4", size=40, color=AM, weight="bold", alpha=a3)
        cv.text(700, 330, "最大误差 |p̈ − v|（m/s²）", size=18, color=FAINT, alpha=a3)
        cv.text(700, 400, "2.2e−4", size=32, color=INK, alpha=a3)
        cv.text(700, 428, "平均误差，约 0.02%", size=18, color=FAINT, alpha=a3)
        cv.text(700, 490, "余下误差全部来自 ±12 N 限幅", size=19, color=DIM, alpha=a3)


# ============================================================ S4
def s4(cv, t, dur):
    title(cv, "换完之后：一个线性，一个非线性")
    a = appear(t, 0.3, 0.8)
    cv.rect(96, 186, 620, 330, CY, width=1.4, alpha=0.8 * a, fill_alpha=0.06 * a, radius=8)
    cv.text(120, 218, "小车通道（线性）", size=22, color=CY, alpha=a)
    math_(cv, 120, 276, r"\ddot p=v", 30, INK)
    cv.text(400, 268, "二重积分器", size=19, color=DIM, alpha=a)

    a2 = appear(t, 2.0, 0.8)
    cv.rect(96, 372, 760, 520, AM, width=1.4, alpha=0.8 * a2, fill_alpha=0.06 * a2, radius=8)
    cv.text(120, 404, "摆杆通道（仍然非线性）", size=22, color=AM, alpha=a2)
    math_(cv, 120, 470, r"\ddot\theta=\frac{g\sin\theta-v\cos\theta}{l}"
                        r"-\frac{b_p\dot\theta}{m_p l^2}", 24, INK)

    a3 = appear(t, 3.6, 0.8)
    cv.text(810, 250, "为什么要这么拆？", size=21, color=INK, alpha=a3)
    cv.text(810, 292, "能量法、滑模法都只需要", size=19, color=DIM, alpha=a3)
    cv.text(810, 322, "处理这一个方程", size=19, color=DIM, alpha=a3)
    cv.text(810, 380, "摩擦、科氏、重力耦合", size=19, color=DIM, alpha=a3)
    cv.text(810, 410, "已被内环吃掉了", size=19, color=DIM, alpha=a3)


# ============================================================ S5
def s5(cv, t, dur):
    title(cv, "奇异点：摆杆水平时，小车说了不算")
    a = appear(t, 0.3, 0.8)
    rows = [(0, "1.00"), (30, "0.87"), (60, "0.50"), (80, "0.17"), (90, "0.00")]
    for i, (deg, val) in enumerate(rows):
        aa = appear(t, 0.6 + 0.55 * i, 0.5)
        if aa <= 0:
            continue
        y = 210 + i * 62
        cv.text(110, y, f"{deg}°", size=22, color=INK, alpha=aa)
        math_(cv, 200, y, val, 20, INK)
        frac = float(val)
        cv.rect(300, y - 16, 300 + 420 * frac, y + 4,
                CY if frac > 0.4 else (AM if frac > 0.05 else MG), width=1,
                alpha=0.85 * aa, fill_alpha=0.30 * aa, radius=3)
        if i == 4:
            cv.text(740, y, "← 完全失效", size=19, color=MG, alpha=aa)
    a2 = appear(t, 4.0, 0.8)
    cv.text(110, 560, "v 对 θ̈ 的系数就是 cosθ：θ → ±90° 时它趋于 0", size=21, color=DIM, alpha=a2)
    a3 = appear(t, 5.2, 0.7)
    chip(cv, 110, 612, "实现里把 |cosθ| 钳位到 ≥ 0.25，并在大角度时绕开这条通道", AM, a3, 20)


# ============================================================ S6
def s6(cv, t, dur):
    title(cv, "本讲小结")
    items = [("精确换元", "内环由 v 反解出 u：分母恒正、全域可逆"),
             ("拆成两层", "内环吃非线性，外环看到 p̈ = v"),
             ("精度", "误差 ~1e−4，全部来自 ±12 N 限幅"),
             ("奇异点", "cosθ → 0 时摆杆通道失效，需显式处理"),
             ("实际价值", "同一能量律：直接调力 0/3，加反解 3/3")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.72 * i, 0.6)
        if aa <= 0:
            continue
        y = 200 + i * 62
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=24, color=CY, alpha=aa, weight="bold")
        cv.text(300, y, v, size=22, color=INK, alpha=aa)
    a = appear(t, 4.4, 0.8)
    cv.text(130, 552, "下一讲：能量整定 —— 把“甩起来”变成一个标量", size=25, color=AM, alpha=a)
    cv.text(130, 596, "作业：逐项去掉干扰项，看能量泵成功率下降多少", size=19, color=FAINT, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "上一讲我们知道了局部最优的边界。这一讲换一个坐标。",
        "把“我能施加多大力”，换成“我能让小车产生多大加速度”。",
        "这个换元是精确的，也是后面所有控制律的前提。",
    ], draw=s1, min_dur=11.0),
    dict(name="why", say=[
        "看同一台机器的两种写法。",
        "力域里，摩擦、科氏项、重力耦合全都混在一个方程里，换一个控制律就要重推一遍。",
        "加速度域里，小车通道直接变成二重积分器，摆杆通道只剩下一个非线性方程。",
    ], draw=s2, min_dur=17.0),
    dict(name="inverse", say=[
        "反解就是一句话：把指定的加速度翻译成需要的力。",
        {"say": "等效质量乘上指令，再加上所有需要抵消的干扰项。",
         "tts": "等效质量乘上指令，再加上所有需要抵消的干扰项。"},
        {"say": "分母恒正，所以对任意指令都有唯一解；数值上误差只有 $9.2\\times10^{-4}$，全部来自限幅。",
         "tts": "分母恒正，所以对任意指令都有唯一解；数值上误差只有万分之九，全部来自限幅。"},
    ], draw=s3, min_dur=17.0),
    dict(name="after", say=[
        "换完之后系统被拆成两层。",
        "外层看到的是小车通道的二重积分器，这是最好对付的对象。",
        "摆杆通道保留全部非线性，但只剩下重力项和加速度指令的耦合。",
        "能量法和滑模法，从此都只需要对付这一个方程。",
    ], draw=s4, min_dur=16.0),
    dict(name="singular", say=[
        "但换坐标有一个代价：奇异点。",
        "指令对摆杆角加速度的系数，就是余弦。",
        {"say": "摆杆越接近水平，小车的权限越小；到 $90$ 度时完全失效。",
         "tts": "摆杆越接近水平，小车的权限越小；到九十度时完全失效。"},
        "所以实现里把余弦钳位，并在大角度时绕开这条通道。",
    ], draw=s5, min_dur=18.0),
    dict(name="summary", say=[
        "总结这一讲：一次精确的换元，换来一个干净的输入通道。",
        {"say": "同一个能量律，直接调力一个都上不去，加了反解 3 个全部成功。",
         "tts": "同一个能量律，直接调力一个都上不去，加了反解三个全部成功。"},
        "下一讲，我们就用这个干净通道，把摆甩起来。",
    ], draw=s6, min_dur=14.0),
]
