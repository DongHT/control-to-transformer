"""第 6 讲：切换与监督 —— 5 个镜头。"""
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
D = json.load(open(os.path.join(HERE, "..", "..", "..", "build", "out", "ch03_06.json")))["ch06"]
SW = D["th_sweep"]


# ============================================================ S1
def s1(cv, t, dur):
    cv.text(640, 252, "倒立摆控制　·　从拉格朗日到 Transformer", size=25, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 352, "第 6 讲　切换与监督", size=66, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 414, "把两种局部真理缝起来", size=31, color=DIM, anchor="ma", alpha=appear(t, 1.7, 0.9))
    cv.line((500, 450), (780, 450), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 512, r"\text{swing}(x)\ \xrightarrow{\ |\theta|<0.40\ }\ "
                            r"\text{stabilize}(x)", 22, DIM, anchor="mc")
    cv.text(90, 620, "课程讲义 · ch06　　代码 · controllers.py: Hybrid", size=19, color=FAINT,
            alpha=appear(t, 3.0, 0.8))
    cartpole(cv, 640, 600, 0.06 * math.sin(1.1 * t), a=0.12 * appear(t, 0.5, 1.2), show_rail=False)


# ============================================================ S2
def s2(cv, t, dur):
    title(cv, "两个控制器，两种真理")
    a = appear(t, 0.3, 0.8)
    cv.rect(96, 186, 600, 400, AM, width=1.2, alpha=0.75 * a, fill_alpha=0.05 * a, radius=8)
    cv.text(120, 218, "能量整定（第 4 讲）", size=22, color=AM, alpha=a)
    cv.text(120, 262, "擅长：大偏差，从任意角度甩到顶", size=19, color=INK, alpha=a)
    cv.text(120, 296, "不会：到顶后停不住（能量过冲）", size=19, color=INK, alpha=a)
    math_(cv, 120, 350, r"E\to 0", 22, DIM)

    a2 = appear(t, 2.0, 0.8)
    cv.rect(96, 430, 600, 600, CY, width=1.2, alpha=0.75 * a2, fill_alpha=0.05 * a2, radius=8)
    cv.text(120, 462, "局部稳定器（第 2 / 5 讲）", size=22, color=CY, alpha=a2)
    cv.text(120, 506, "擅长：小偏差，压到 1e−3 rad", size=19, color=INK, alpha=a2)
    cv.text(120, 540, "不会：角度一大模型就失效", size=19, color=INK, alpha=a2)
    math_(cv, 120, 578, r"x\to 0", 22, DIM)

    a3 = appear(t, 3.6, 0.8)
    cv.text(660, 260, "目标函数不同 →", size=22, color=INK, alpha=a3)
    cv.text(660, 300, "没有任何连续反馈律", size=22, color=INK, alpha=a3)
    cv.text(660, 334, "能同时做到两件事", size=22, color=INK, alpha=a3)
    a4 = appear(t, 4.8, 0.8)
    chip(cv, 660, 420, "必须显式切换", MG, a4, 24)
    cv.text(660, 500, "问题只剩：在哪切、怎么切、切完抖不抖", size=19, color=DIM, alpha=a4)


# ============================================================ S3
def s3(cv, t, dur):
    title(cv, "捕获窗口：多大才接得住")
    a = appear(t, 0.3, 0.8)
    cv.text(110, 190, "捕获阈值 θ_in", size=19, color=FAINT, alpha=a)
    cv.text(420, 190, "成功回合", size=19, color=FAINT, alpha=a)
    cv.text(700, 190, "结果", size=19, color=FAINT, alpha=a)
    for i, r in enumerate(SW):
        aa = appear(t, 0.6 + 0.5 * i, 0.5)
        if aa <= 0:
            continue
        y = 240 + i * 62
        ok = r["ok"] == r["n"]
        col = GR if ok else MG
        cv.text(110, y, f"{r['th_in']:.2f} rad", size=23, color=INK, alpha=aa)
        cv.text(420, y, f"{r['ok']}/{r['n']}", size=23, color=col, alpha=aa)
        cv.text(700, y, "全部成功" if ok else "全部失败（窗口太窄）", size=21, color=col, alpha=aa)
    a2 = appear(t, 3.6, 0.9)
    cv.rect(96, 500, 1000, 600, MG, width=1.1, alpha=0.65 * a2, fill_alpha=0.05 * a2, radius=8)
    cv.text(120, 532, "窗口越窄看起来越“安全”，实际越糟：", size=20, color=MG, alpha=a2)
    cv.text(120, 566, "摆过顶时速度不为 0，窄窗口可能只有一个采样周期，直接看漏", size=19, color=INK,
            alpha=a2)


# ============================================================ S4
def s4(cv, t, dur):
    title(cv, "迟滞与切换瞬态")
    a = appear(t, 0.3, 0.8)
    cv.text(110, 190, "迟滞：进出用不同阈值", size=21, color=INK, alpha=a)
    math_(cv, 130, 240, r"|\theta|<0.40,\ |\dot\theta|<2.5\ \Rightarrow\ \text{稳摆}", 21, CY)
    math_(cv, 130, 286, r"|\theta|>0.80\ \text{或}\ |\dot\theta|>3.5\ \Rightarrow\ \text{摆起}", 21, AM)
    a2 = appear(t, 2.0, 0.8)
    cv.text(110, 356, "没迟滞会怎样：模式在临界点高频切换 → 力抖振", size=19, color=DIM, alpha=a2)

    a3 = appear(t, 3.4, 0.9)
    cv.text(110, 440, "切换瞬态", size=21, color=INK, alpha=a3)
    cv.text(110, 486, "11.0 N", size=44, color=MG, weight="bold", alpha=a3)
    cv.text(110, 518, "最大单步 Δu（能量+LQR）", size=18, color=FAINT, alpha=a3)

    a4 = appear(t, 5.0, 0.8)
    cv.text(520, 440, "这一帧，指令从满幅正转到满幅负", size=20, color=DIM, alpha=a4)
    cv.text(520, 480, "真机会有电流冲击；可用速率限制器压平", size=20, color=DIM, alpha=a4)
    cv.text(520, 530, "（第 10 讲会看到：模仿学习把它压到 5.4 N）", size=19, color=CY, alpha=a4)

    a5 = appear(t, 6.4, 0.8)
    cv.text(110, 600, "监督器 5 条清单：能采到 · 落在 attractor 内 · 有迟滞 · 限速率 · 能回退",
            size=19, color=FAINT, alpha=a5)


# ============================================================ S5
def s5(cv, t, dur):
    title(cv, "本讲小结")
    items = [("两种真理", "能量管全局、稳定器管局部"),
             ("捕获窗口", "0.40 rad：0.25 会看漏，0.60 接近边界"),
             ("迟滞", "0.40/0.80 与 2.5/3.5，抗抖振"),
             ("切换瞬态", "Δu ≈ 11 N，需要速率限制"),
             ("清单", "能采到 · 在吸引域内 · 迟滞 · 限幅 · 可回退")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.72 * i, 0.6)
        if aa <= 0:
            continue
        y = 200 + i * 62
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=24, color=CY, alpha=aa, weight="bold")
        cv.text(290, y, v, size=22, color=INK, alpha=aa)
    a = appear(t, 4.4, 0.8)
    cv.text(130, 552, "下一讲：不再手写控制律 —— 从数据里学一个", size=25, color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "上一讲解决了“停得住”，但那是从已经接近直立的状态开始。",
        "这一讲把会甩的和会停的缝在一起。",
        "难点不在两边，而在接缝。",
    ], draw=s1, min_dur=11.0),
    dict(name="two", say=[
        "两个控制器的目标函数根本不同。",
        "能量整定擅长把摆从任意位置甩上来，但它到顶后停不住。",
        "局部稳定器能压到千分之一弧度，但角度一大模型就失效。",
        "所以没有任何一个连续反馈律能同时做到——必须显式切换。",
    ], draw=s2, min_dur=18.0),
    dict(name="window", say=[
        "第一个问题：在哪切。",
        {"say": "扫一遍捕获阈值：$0.25$ 弧度全部失败，$0.35$ 以上全部成功。",
         "tts": "扫一遍捕获阈值：零点二五弧度全部失败，零点三五以上全部成功。"},
        "窗口越窄看起来越安全，实际越糟：摆过顶时速度不为 0，窄窗口可能只有一个采样周期。",
    ], draw=s3, min_dur=17.0),
    dict(name="hyst", say=[
        "第二个问题：怎么切才不抖。",
        "进出用不同阈值，中间留一条迟滞带。",
        {"say": "第 3 个问题：切换那一帧，力会跳变 $11$ 牛顿。",
         "tts": "第三个问题：切换那一帧，力会跳变十一牛顿。"},
        "这在真机上是一次电流冲击，需要速率限制器压平。",
    ], draw=s4, min_dur=18.0),
    dict(name="summary", say=[
        "总结：两种真理、一个窗口、一条迟滞带、5 条清单。",
        "从下一讲开始，我们不再手写控制律，而是从数据里学一个。",
    ], draw=s5, min_dur=13.0),
]
