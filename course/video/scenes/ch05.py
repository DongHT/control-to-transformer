"""第 5 讲：滑模控制 —— 6 个镜头。"""
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
D = json.load(open(os.path.join(HERE, "..", "..", "..", "build", "out", "ch03_06.json")))["ch05"]
CE = D["curves"]["elqr"]; CS = D["curves"]["esmc"]


# ============================================================ S1
def s1(cv, t, dur):
    cv.text(640, 252, "倒立摆控制　·　从拉格朗日到 Transformer", size=25, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 352, "第 5 讲　滑模控制", size=68, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 414, "把鲁棒性做进结构", size=31, color=DIM, anchor="ma", alpha=appear(t, 1.7, 0.9))
    cv.line((500, 450), (780, 450), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 512, r"s=\dot\theta+\lambda(\theta-\theta_{\mathrm{ref}}),\quad s\to0",
              25, DIM, anchor="mc")
    cv.text(90, 620, "课程讲义 · ch05　　代码 · controllers.py: SMC", size=19, color=FAINT,
            alpha=appear(t, 3.0, 0.8))
    cartpole(cv, 640, 600, 0.05 * math.sin(1.6 * t), a=0.12 * appear(t, 0.5, 1.2), show_rail=False)


# ============================================================ S2
def s2(cv, t, dur):
    title(cv, "滑模面：把二维问题化成一维")
    a = appear(t, 0.3, 0.8)
    X0, Y0, X1, Y1 = 110, 210, 640, 540
    axes(cv, X0, Y0, X1, Y1, a, "θ [rad]", "θ̇ [rad/s]")
    sx = lambda v: X0 + (v + 1.0) / 2.0 * (X1 - X0)
    sy = lambda v: Y1 - (v + 4.0) / 8.0 * (Y1 - Y0)
    for v in (-4, -2, 0, 2, 4):
        cv.line((X0 - 5, sy(v)), (X0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8 * a)
        cv.text(X0 + 9, sy(v) + 6, str(v), size=15, color=FAINT, alpha=a)
    for v in (-1, 0, 1):
        cv.line((sx(v), Y1), (sx(v), Y1 + 5), (92, 110, 132), width=1.2, alpha=0.8 * a)
        cv.text(sx(v), Y1 + 26, str(v), size=15, color=FAINT, anchor="ma", alpha=a)
    # s = 0  →  theta_dot = -lambda * theta
    lam = 6.0
    curvy(cv, [(sx(v), sy(max(-4, min(4, -lam * v)))) for v in np.linspace(-1, 1, 80)],
          AM, 2.4, a)
    cv.text(sx(0.55), sy(-3.2), "s = 0", size=20, color=AM, alpha=a)
    # a trajectory converging onto the surface
    p = min(1.0, max(0.0, (t - 1.2) / 3.0))
    pts = []
    for i in range(int(60 * p)):
        u = i / 59
        th = 0.9 * math.exp(-4.2 * u)
        w = -6.0 * 0.9 * math.exp(-4.2 * u) + 0.35 * math.sin(9 * u) * math.exp(-3 * u)
        pts.append((sx(th), sy(max(-4, min(4, w)))))
    curvy(cv, pts, CY, 2.2, a)
    if p >= 1:
        cv.circle(pts[-1][0], pts[-1][1], 5, CY, width=0, alpha=a, fill_alpha=a)

    a2 = appear(t, 2.6, 0.8)
    cv.rect(690, 230, 1180, 330, AM, width=1.2, alpha=0.7 * a2, fill_alpha=0.06 * a2, radius=8)
    cv.text(712, 264, "只要停在 s = 0 上，θ 就按 e^(−λt) 衰减", size=20, color=AM, alpha=a2)
    a3 = appear(t, 4.0, 0.8)
    cv.text(690, 400, "二维跟踪 → 一维镇定", size=23, color=INK, alpha=a3)
    cv.text(690, 440, "这就是滑模的核心动作", size=20, color=DIM, alpha=a3)


# ============================================================ S3
def s3(cv, t, dur):
    title(cv, "到达律：怎么把状态按到滑模面上")
    a = appear(t, 0.3, 0.8)
    if a > 0:
        math_(cv, 110, 236, r"\dot s=-\eta\,\mathrm{sat}(s/\phi)-k_s s", 26, INK)
        cv.text(110, 300, "η：抗扰强度　　φ：边界层宽度", size=21, color=DIM, alpha=a)
    a2 = appear(t, 1.8, 0.9)
    if a2 > 0:
        # sat curve
        X0, Y0, X1, Y1 = 110, 380, 560, 560
        axes(cv, X0, Y0, X1, Y1, a2, "s", "sat(s/φ)")
        sx = lambda v: X0 + (v + 1.0) / 2.0 * (X1 - X0)
        sy = lambda v: Y1 - (v + 1.1) / 2.2 * (Y1 - Y0)
        curvy(cv, [(sx(v), sy(max(-1.05, min(1.05, v / 0.25)))) for v in np.linspace(-1, 1, 300)],
              CY, 2.4, a2)
        cv.line((sx(-0.25), Y1), (sx(-0.25), Y0), (90, 108, 130), width=1, alpha=0.6 * a2, dash=5)
        cv.line((sx(0.25), Y1), (sx(0.25), Y0), (90, 108, 130), width=1, alpha=0.6 * a2, dash=5)
        cv.text(sx(0), Y0 + 6, "边界层 |s| < φ", size=17, color=FAINT, anchor="ma", alpha=a2)
    a3 = appear(t, 3.6, 0.8)
    cv.text(640, 400, "φ → 0：经典滑模，强鲁棒 + 抖振", size=20, color=MG, alpha=a3)
    cv.text(640, 440, "φ 适中：线性区内平滑，区外强鲁棒", size=20, color=CY, alpha=a3)
    cv.text(640, 480, "本课取 φ = 0.25", size=20, color=DIM, alpha=a3)
    a4 = appear(t, 5.0, 0.8)
    chip(cv, 640, 548, "1/cosθ 必须钳位到 |cosθ| ≥ 0.25", AM, a4, 19)


# ============================================================ S4
def s4(cv, t, dur):
    title(cv, "抗扰实测：4 N 持续侧风")
    a = appear(t, 0.3, 0.8)
    X0, Y0, X1, Y1 = 110, 230, 760, 480
    axes(cv, X0, Y0, X1, Y1, a, "t [s]", "θ [rad]")
    sx = lambda tv: X0 + tv / 12.0 * (X1 - X0)
    sy = lambda v: Y1 - (v + 0.04) / 0.09 * (Y1 - Y0)
    for v in (-0.02, 0.0, 0.02):
        cv.line((X0 - 5, sy(v)), (X0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8 * a)
        cv.text(X0 + 9, sy(v) + 6, f"{v:.2f}", size=15, color=FAINT, alpha=a)
    for tv in (0, 4, 8, 12):
        cv.line((sx(tv), Y1), (sx(tv), Y1 + 5), (92, 110, 132), width=1.2, alpha=0.8 * a)
        cv.text(sx(tv), Y1 + 26, str(tv), size=15, color=FAINT, anchor="ma", alpha=a)
    cv.rect(sx(4), Y0, sx(8), Y1, AM, width=0, alpha=0.10 * a, fill_alpha=0.10 * a)
    cv.text(sx(6), Y0 + 18, "风 4 N", size=16, color=AM, anchor="ma", alpha=a)
    prog = min(1.0, max(0.0, (t - 1.0) / 3.5))
    n = max(2, int(len(CE["t"]) * prog))
    curvy(cv, [(sx(CE["t"][i]), sy(max(-0.04, min(0.045, CE["th"][i])))) for i in range(n)],
          BL, 2.2, a)
    curvy(cv, [(sx(CS["t"][i]), sy(max(-0.04, min(0.045, CS["th"][i])))) for i in range(n)],
          CY, 2.2, a)
    cv.text(sx(9.4), sy(0.030), "LQR", size=18, color=BL, alpha=a)
    cv.text(sx(9.4), sy(0.012), "SMC", size=18, color=CY, alpha=a)

    rows = [("0.0121 rad", "LQR 稳态误差", BL, 2.6), ("0.0053 rad", "SMC 稳态误差（2.3× 更好）", CY, 3.4),
            ("+7%", "代价：力 RMS 略高", AM, 4.2)]
    for txt, sub, col, t0 in rows:
        aa = appear(t, t0, 0.6)
        if aa <= 0:
            continue
        y = 250 + (t0 - 2.6) * 92
        cv.text(830, y, txt, size=30, color=col, weight="bold", alpha=aa)
        cv.text(830, y + 26, sub, size=18, color=FAINT, alpha=aa)


# ============================================================ S5
def s5(cv, t, dur):
    title(cv, "一个真实的坑：小车回中")
    a = appear(t, 0.3, 0.8)
    cv.rect(96, 190, 600, 340, MG, width=1.2, alpha=0.75 * a, fill_alpha=0.05 * a, radius=8)
    cv.text(120, 222, "× 直接把 −k_x p − k_d·(dp/dt) 加进 v", size=21, color=MG, alpha=a)
    cv.text(120, 268, "两个回路抢同一个输入：", size=19, color=INK, alpha=a)
    cv.text(120, 300, "摆杆回路要压 s，回中回路要反向", size=19, color=INK, alpha=a)
    cv.text(120, 332, "实测：被推出边界层 → 极限环 → 发散", size=19, color=MG, alpha=a)

    a2 = appear(t, 2.4, 0.8)
    cv.rect(96, 380, 700, 540, CY, width=1.2, alpha=0.75 * a2, fill_alpha=0.05 * a2, radius=8)
    cv.text(120, 412, "√ 倾斜参考：让摆杆回路去追一个偏置", size=21, color=CY, alpha=a2)
    math_(cv, 120, 462, r"\theta_{\mathrm{ref}}=-\mathrm{clip}(k_p p+k_d\dot p,\ \pm0.25)", 22, INK)
    cv.text(120, 512, "摆杆仍是主角，回中只是缓慢的参考漂移", size=19, color=DIM, alpha=a2)

    a3 = appear(t, 4.0, 0.8)
    cv.text(760, 300, "多输入系统里这叫", size=20, color=INK, alpha=a3)
    cv.text(760, 336, "级联 / 时标分离", size=24, color=AM, weight="bold", alpha=a3)
    cv.text(760, 400, "在单输入欠驱动系统里，", size=20, color=DIM, alpha=a3)
    cv.text(760, 432, "这是唯一不冲突的回中方式", size=20, color=DIM, alpha=a3)


# ============================================================ S6
def s6(cv, t, dur):
    title(cv, "本讲小结")
    items = [("滑模面", "s = θ̇ + λ(θ−θ_ref)：二维化一维"),
             ("到达律", "η 换鲁棒，φ 换平滑"),
             ("钳位", "1/cosθ 必须限制，否则奇异"),
             ("回中", "用倾斜参考，不能直接叠加"),
             ("实测", "稳态误差 0.0121 → 0.0053 rad，力 +7%")]
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
    cv.text(130, 552, "下一讲：切换与监督 —— 把两种局部真理缝起来", size=25, color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "上一讲让摆甩起来了，但它到了顶上停不住。这一讲解决“停得住”。",
        "LQR 相信模型，滑模不相信——它先规定一条必须到达的线。",
        "代价是抖振，收益是抗扰。",
    ], draw=s1, min_dur=12.0),
    dict(name="surface", say=[
        "看相平面。横轴是角度，纵轴是角速度。",
        {"say": "取 $s=\\dot\\theta+\\lambda\\theta$，令它为零，就得到一条过原点的直线。",
         "tts": "取 s 等于 theta 点加上 lambda 乘 theta，令它为零，就得到一条过原点的直线。"},
        "只要状态停在这条线上，角度就会按指数衰减。二维跟踪问题化成了一维。",
    ], draw=s2, min_dur=18.0),
    dict(name="reach", say=[
        "接下来是怎么把状态“按”上去。",
        {"say": "到达律用饱和函数：边界层内线性、外面强力，$\\eta$ 管抗扰，$\\phi$ 管平滑。",
         "tts": "到达律用饱和函数：边界层内线性、外面强力，eta 管抗扰，phi 管平滑。"},
        {"say": "还有一个工程细节：指令里有 $1/\\cos\\theta$，必须钳位，否则摆杆水平时会爆炸。",
         "tts": "还有一个工程细节：指令里有 cos theta 分之一，必须钳位，否则摆杆水平时会爆炸。"},
    ], draw=s3, min_dur=19.0),
    dict(name="dist", say=[
        "拿持续侧风来检验它。",
        {"say": "第 $4$ 到第 $8$ 秒加 $4$ 牛顿的风，LQR 的稳态误差是 $0.0121$ 弧度。",
         "tts": "第四到第八秒加四牛顿的风，LQR 的稳态误差是零点零一二一弧度。"},
        {"say": "滑模做到 $0.0053$，好了 $2.3$ 倍。代价是力大了 $7\\%$。",
         "tts": "滑模做到零点零零五三，好了二点三倍。代价是力大了百分之七。"},
        "鲁棒性不是免费的，它是用控制能量换来的。",
    ], draw=s4, min_dur=19.0),
    dict(name="pitfall", say=[
        "这一讲还有一个必须讲的坑：小车回中。",
        "如果直接把回中项加到加速度指令上，两个回路会抢同一个输入。",
        "实测会发散。正确做法是给摆杆回路一个极小的倾斜参考。",
    ], draw=s5, min_dur=17.0),
    dict(name="summary", say=[
        "总结：滑模把跟踪化成一维镇定，用 eta 换鲁棒，用 phi 换平滑。",
        "但摆起和稳摆依然是两个控制器——下一讲把它们缝起来。",
    ], draw=s6, min_dur=13.0),
]
