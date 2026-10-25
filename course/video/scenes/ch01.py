"""第 1 讲：建模 —— 8 个镜头（LaTeX 公式版）。"""
import math
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from ch01_parts import *          # noqa: F401,F403
from render import (Canvas, INK, DIM, FAINT, CY, AM, MG, BL, GR, BG,
                    appear, smooth, lerp, ease, W, H)
import mathrender as mr

MC, MP, L, G, UMAX = 1.0, 0.25, 0.6, 9.81, 12.0
SCALE = 190.0
RAIL = 1.4


# ----------------------------------------------------------------- 公式组件
from kit import math_ as _kit_math, title as _kit_title   # 与其它讲统一：居中 + 淡入


def math_(cv, x, y, tex, size=26, color=INK, anchor="ls", check=True, tag=None):
    return _kit_math(cv, x, y, tex, size, color, anchor, check, tag)


def title(cv, text, sub=None):
    return _kit_title(cv, text, sub)


def _unused_math_(cv, x, y, tex, size=26, color=INK, anchor="ls", check=True, tag=None):
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


def cartpole(cv, x, y0, th, p=0.0, u=0.0, a=1.0, show_rail=True):
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
    px, py = cx, ytop
    tx = px + math.sin(th) * L * SCALE
    ty = py - math.cos(th) * L * SCALE
    cv.line((px, py), (tx, ty), CY, width=5, alpha=a)
    cv.circle(tx, ty, 9, CY, width=2, alpha=a, fill_alpha=0.35 * a)
    cv.circle(px, py, 3.2, INK, width=0, alpha=a, fill_alpha=a)
    cv.line((px, py), (px, py - L * SCALE * 1.35), (110, 130, 150), width=1, alpha=0.55 * a, dash=5)
    r = L * SCALE * 0.55
    pts = []
    for i in range(13):
        ang = math.radians(90 - i * (math.degrees(th) / 12))
        pts.append((px + r * math.cos(ang), py - r * math.sin(ang)))
    for i in range(len(pts) - 1):
        cv.line(pts[i], pts[i + 1], AM, width=2, alpha=a)
    if abs(th) > 0.06:
        mid = th / 2
        cv.text(px + (r + 18) * math.sin(mid), py - (r + 18) * math.cos(mid) + 8,
                "θ", size=24, color=AM, alpha=a, check=False)
    if abs(u) > 0.2:
        ln = min(1.0, abs(u) / UMAX) * 90 * (1 if u > 0 else -1)
        cv.arrow((cx, y0 - ch / 2 - 6), (cx + ln, y0 - ch / 2 - 6), AM, width=2.5, alpha=a)
    return cx, ytop


def _free_fall():
    x = np.array([0.0, 2.5, 0.0, 0.0])

    def f(y):
        sn, cs = np.sin(y[1]), np.cos(y[1])
        den = MC + MP * sn * sn
        a = (-0.12 * y[2] + MP * L * sn * y[3] ** 2 + 0.004 * cs * y[3]
             - MP * G * sn * cs) / den
        b = (G * sn - cs * a - 0.004 * y[3] / (MP * L)) / L
        return np.array([y[2], y[3], a, b])

    out = []
    h = 0.01
    for _ in range(800):
        k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append((x[0], x[1]))
    return np.array(out)


FALL = _free_fall()


def _tilt_traj(th0_deg=1.0, T=0.8):
    """real response of the plant released from a 1-degree tilt (u = 0)."""
    x = np.array([0.0, math.radians(th0_deg), 0.0, 0.0])
    def f(y):
        sn, cs = np.sin(y[1]), np.cos(y[1])
        den = MC + MP * sn * sn
        a = (-0.12 * y[2] + MP * L * sn * y[3] ** 2 + 0.004 * cs * y[3]
             - MP * G * sn * cs) / den
        b = (G * sn - cs * a - 0.004 * y[3] / (MP * L)) / L
        return np.array([y[2], y[3], a, b])
    out, h = [(0.0, th0_deg)], 0.01
    for k in range(int(T / h)):
        k1 = f(x); k2 = f(x + 0.5*h*k1); k3 = f(x + 0.5*h*k2); k4 = f(x + h*k3)
        x = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
        if (k + 1) % 1 == 0:
            out.append(((k + 1) * h, math.degrees(x[1])))
    return np.array(out)


TILT = _tilt_traj()


# ============================================================ S1
def s1(cv, t, dur):
    cv.text(640, 252, "倒立摆控制　·　从拉格朗日到 Transformer",
            size=26, color=CY, anchor="ma", alpha=appear(t, 0.2, 0.9))
    cv.text(640, 352, "第 1 讲　建模", size=74, color=INK, anchor="ma",
            weight="bold", alpha=appear(t, 0.9, 0.9))
    cv.text(640, 414, "把一台机器写成方程", size=33, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((500, 450), (780, 450), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    a = appear(t, 2.8, 0.9)
    math_(cv, 640, 520, r"\dot{x}=f(x,u),\qquad x=(p,\ \theta,\ v,\ \omega)",
          24, DIM, anchor="mc", check=False) if a > 0 else None
    cv.text(90, 618, "课程讲义 · ch01　　代码 · build/plant.py", size=21,
            color=FAINT, alpha=appear(t, 3.2, 0.8))


# ============================================================ S2
def s2(cv, t, dur):
    title(cv, "系统与符号：每个量都标在图上")
    idx = min(int(t / 0.012), len(FALL) - 1)
    pp, th = FALL[idx]
    a = appear(t, 0.2, 0.8)

    OX, OY = 300, 470          # 导轨原点（小车零位）
    SC = 110.0                 # px / m
    rail = 2.4
    cartx = OX + pp * SC

    # 导轨
    cv.line((OX - rail * SC, OY), (OX + rail * SC, OY), (110, 130, 150), width=3, alpha=0.85 * a)
    for sgn in (-1, 1):
        x = OX + sgn * rail * SC
        cv.line((x, OY - 34), (x, OY + 10), MG, width=2, alpha=0.8 * a)
    cv.text(OX + rail * SC + 8, OY + 6, "x_max = 2.4 m", size=15, color=MG, alpha=a)
    # 轨道摩擦 b_c


    # 小车
    cw, chh = 84, 40
    cv.rect(cartx - cw / 2, OY - chh - 6, cartx + cw / 2, OY - 6, (60, 90, 118), width=2,
            alpha=a, fill_alpha=0.55 * a, radius=5)
    cv.text(cartx, OY - chh - 16, "m_c = 1.0 kg", size=16, color=INK, anchor="ma", alpha=a)

    # 摆杆与质点
    tx = cartx + math.sin(th) * L * SC
    ty = OY - chh - 6 - math.cos(th) * L * SC
    cv.line((cartx, OY - chh - 6), (tx, ty), CY, width=6, alpha=a)
    cv.circle(tx, ty, 11, CY, width=2, alpha=a, fill_alpha=0.45 * a)
    cv.text(tx + 16, ty - 6, "m_p = 0.25 kg（质点）", size=15, color=CY, alpha=a)
    # 杆长标注
    mxp, myp = (cartx + tx) / 2 + 26, (OY - chh - 6 + ty) / 2
    cv.text(mxp, myp, "l = 0.60 m", size=15, color=CY, alpha=a)

    # 角度弧与 θ
    r = 74
    cv.line((cartx, OY - chh - 6), (cartx, OY - chh - 6 - r), (110, 130, 150), width=1,
            alpha=0.6 * a, dash=5)
    arc = [(cartx + r * 0.55 * math.sin(math.pi / 2 * k / 12),
            OY - chh - 6 - r * 0.55 * math.cos(math.pi / 2 * k / 12)) for k in range(13)]
    cv_line = ""
    for k in range(len(arc) - 1):
        pass
    import math as _m
    pts = [(cartx + 0.55 * r * _m.sin(th * k / 12), OY - chh - 6 - 0.55 * r * _m.cos(th * k / 12))
           for k in range(13)]
    for k in range(len(pts) - 1):
        cv.line(pts[k], pts[k + 1], AM, width=1.6, alpha=0.9 * a)
    cv.text(cartx + 34, OY - chh - 6 - 58, "θ", size=22, color=AM, alpha=a)

    # 位置 p（原点 → 小车）
    y0 = OY + 46
    cv.line((OX, y0), (cartx, y0), BL, width=2, alpha=a)
    cv.arrow((cartx - 12, y0), (cartx, y0), BL, width=2, alpha=a)
    cv.text((OX + cartx) / 2, y0 + 20, "p", size=20, color=BL, anchor="ma", alpha=a)
    cv.line((OX, OY - 4), (OX, y0 + 4), (110, 130, 150), width=1, alpha=0.6 * a, dash=5)
    cv.text(OX - 8, y0 + 20, "0", size=15, color=FAINT, anchor="ra", alpha=a)

    # 力 u
    cv.arrow((cartx + cw / 2 + 4, OY - chh / 2 - 6), (cartx + cw / 2 + 70, OY - chh / 2 - 6),
             MG, width=2.6, alpha=a)
    cv.text(cartx + cw / 2 + 76, OY - chh / 2 - 10, "u  |u| ≤ 12 N", size=15, color=MG, alpha=a)

    # 重力
    cv.arrow((tx, ty + 14), (tx, ty + 62), (150, 160, 172), width=1.8, alpha=0.9 * a)
    cv.text(tx + 6, ty + 74, "g = 9.81 m/s²", size=14, color=(150, 160, 172), alpha=a)

    # 转轴摩擦


    a2 = appear(t, 4.6, 0.8)
    cv.text(90, 596, "阻尼：b_c = 0.12 N·s/m（导轨）　b_p = 0.004 N·m·s（转轴）", size=15,
            color=FAINT, alpha=a2)
    a3 = appear(t, 5.6, 0.8)
    cv.text(90, 640, "状态 x = (p, θ, v, ω)，v = ṗ，ω = θ̇；参数取值见下一页", size=17,
            color=DIM, alpha=a3)


def s_params(cv, t, dur):
    title(cv, "全部参数：符号、取值与含义")
    head = [("符号", 110), ("取值", 330), ("单位", 500), ("含义", 650)]
    for txt, x in head:
        cv.text(x, 186, txt, size=16, color=FAINT, alpha=appear(t, 0.3, 0.6))
    rows = [(r"m_c", "1.0", "kg", "小车质量", CY),
            (r"m_p", "0.25", "kg", "摆杆质点质量", CY),
            (r"l", "0.60", "m", "转轴到质心距离", CY),
            (r"g", "9.81", "m/s²", "重力加速度", DIM),
            (r"b_c", "0.12", "N·s/m", "导轨粘性阻尼", BL),
            (r"b_p", "0.004", "N·m·s/rad", "转轴粘性阻尼", BL),
            (r"u_{\max}", "12", "N", "执行器饱和", MG),
            (r"x_{\max}", "2.4", "m", "导轨半长", MG),
            (r"\Delta t", "0.02", "s", "控制周期（50 Hz）", AM)]
    for i, (tex, val, unit, mean, col) in enumerate(rows):
        aa = appear(t, 0.5 + 0.42 * i, 0.5)
        if aa <= 0:
            continue
        y = 228 + i * 44
        math_(cv, 120, y, tex, 20, col)
        cv.text(330, y, val, size=20, color=INK, alpha=aa)
        cv.text(500, y, unit, size=18, color=FAINT, alpha=aa)
        cv.text(650, y, mean, size=19, color=INK, alpha=aa)
        cv.line((110, y + 14), (1150, y + 14), (60, 76, 94), width=1, alpha=0.35 * aa)
    a = appear(t, 4.6, 0.8)
    chip(cv, 110, 648, "这 9 个数字决定了后面所有实验；改参数只需改 build/plant.py 里的 P", CY, a, 17)


def s_coords(cv, t, dur):
    title(cv, "符号约定：正方向与单位")
    rows = [(r"p", "小车位置，向右为正；原点取导轨中点", BL),
            (r"\theta", "摆杆与竖直向上的夹角，右侧为正", AM),
            (r"u", "作用在小车上的水平力，向右为正", MG),
            (r"v=\dot p,\ \omega=\dot\theta", "平动速度与角速度", CY)]
    for i, (tex, txt, col) in enumerate(rows):
        aa = appear(t, 0.4 + 0.5 * i, 0.55)
        if aa <= 0:
            continue
        y = 200 + i * 66
        math_(cv, 110, y, tex, 21, col)
        cv.text(360, y, txt, size=19, color=INK, alpha=aa)

    a = appear(t, 2.8, 0.8)
    cv.text(110, 500, "两条容易搞反的约定", size=19, color=INK, alpha=a)
    a2 = appear(t, 3.4, 0.8)
    cv.text(130, 540, "① θ = 0 是竖直向上（不稳定平衡），不是向下：所以 sinθ 项的符号与教科书相反", size=16,
            color=DIM, alpha=a2)
    cv.text(130, 572, "② 力 u 向右为正，但让摆“向右倾”需要把车向右推：u 与 θ 同号时摆更快倒下", size=16,
            color=DIM, alpha=a2)
    a3 = appear(t, 4.8, 0.8)
    math_(cv, 110, 632, r"x=(p,\ \theta,\ v,\ \omega)^{\top}\in\mathbb{R}^{4},"
                        r"\qquad \dot x=f(x,u)", 19, DIM)


# ============================================================ S3
def s3(cv, t, dur):
    title(cv, "一个输入，两个自由度")
    a0 = appear(t, 0.3, 0.9)
    if a0 > 0:
        math_(cv, 200, 300, r"\Delta(x)=", 26, INK, anchor="mc")
        box = math_(cv, 330, 300,
                    r"\begin{bmatrix} \dfrac{1}{m_c+m_p\sin^2\theta} \\[8pt] "
                    r"\dfrac{-\cos\theta}{l\,(m_c+m_p\sin^2\theta)} \end{bmatrix}",
                    20, INK, anchor="lc")
    a1 = appear(t, 2.4, 0.8)
    if a1 > 0 and a0 > 0:
        cv.rect(box[0] - 14, box[1] - 12, box[2] + 14, box[3] + 12, MG,
                width=1.6, alpha=0.7 * a1, radius=8)
        cv.text(box[2] + 34, 252, "列向量", size=24, color=MG, alpha=a1)
        cv.text(box[2] + 34, 292, "秩 = 1", size=34, color=MG, weight="bold", alpha=a1)
    a2 = appear(t, 4.2, 0.9)
    cv.text(200, 452, "→ 对任何 u，都无法同时指定 p 和 θ 的二阶导", size=26,
            color=INK, alpha=a2)
    a3 = appear(t, 6.0, 0.6)
    chip(cv, 200, 566, "欠驱动", MG, a3, 24)
    chip(cv, 358, 566, "本质非线性", AM, appear(t, 6.4, 0.6), 24)
    chip(cv, 552, 566, "开环不稳定", CY, appear(t, 6.8, 0.6), 24)


# ============================================================ S4
def s4(cv, t, dur):
    title(cv, "拉格朗日方程：3 行写出模型")
    x0, y0 = 110, 232
    a = appear(t, 0.3, 0.7)
    if a > 0:
        A = math_(cv, x0, y0, r"T=\tfrac{1}{2}(m_c+m_p)\,v^2+", 23, INK)
        bx = A[2] + 6
        B = math_(cv, bx, y0, r"m_p l\cos\theta\,v\,\omega", 23, AM)
        C = math_(cv, B[2] + 6, y0, r"+\tfrac{1}{2}m_p l^2\omega^2", 23, INK)
        a2 = appear(t, 1.6, 0.7)
        if a2 > 0:                       # highlight BEHIND the term, then repaint it
            cv.rect(B[0] - 10, B[1] - 8, B[2] + 10, B[3] + 8, AM, width=1.4,
                    alpha=0.75 * a2, fill_alpha=0.10 * a2, radius=6)
            math_(cv, bx, y0, r"m_p l\cos\theta\,v\,\omega", 23, AM, check=False)
    a3 = appear(t, 2.8, 0.7)
    if a3 > 0:
        math_(cv, x0, 336, r"V=m_p g l\cos\theta", 23, INK)
    a4 = appear(t, 3.6, 0.7)
    if a4 > 0:
        math_(cv, x0, 424, r"R=\tfrac{1}{2}b_c v^2+\tfrac{1}{2}b_p\omega^2", 23, INK)
    a5 = appear(t, 4.6, 0.8)
    if a5 > 0:
        math_(cv, 806, 262, r"\frac{d}{dt}\frac{\partial L}{\partial \dot q_i}"
                            r"-\frac{\partial L}{\partial q_i}"
                            r"+\frac{\partial R}{\partial \dot q_i}=Q_i", 20, DIM, anchor="ms")
    a6 = appear(t, 5.2, 0.7)
    if a6 > 0:
        math_(cv, 806, 392, r"L=T-V", 21, DIM, anchor="ms")
    a7 = appear(t, 6.0, 0.7)
    if a7 > 0:
        chip(cv, 760, 520, "两行方程 · 全是非线性", CY, a7, 23)


# ============================================================ S5
def s5(cv, t, dur):
    title(cv, "两个耦合方程 → 解出加速度")
    a = appear(t, 0.3, 0.8)
    if a > 0:
        cv.text(110, 172, "小车", size=21, color=FAINT, alpha=a)
        math_(cv, 160, 212, r"(m_c+m_p)\,\ddot p+m_p l\cos\theta\,\ddot\theta"
                           r"-m_p l\sin\theta\,\omega^2=u-b_c v", 19, INK)
        cv.text(110, 252, "摆杆", size=21, color=FAINT, alpha=a)
        math_(cv, 160, 286, r"m_p l\cos\theta\,\ddot p+m_p l^2\ddot\theta"
                           r"+m_p g l\sin\theta=-b_p\omega", 19, INK)
    a2 = appear(t, 1.8, 0.7)
    if a2 > 0:
        cv.arrow((110, 330), (110, 372), CY, width=2.2)
    a3 = appear(t, 2.4, 0.9)
    if a3 > 0:
        math_(cv, 110, 448,
              r"\ddot p=\frac{u-b_c v+m_p l\sin\theta\,\omega^2"
              r"+b_p\cos\theta\,\omega-m_p g\sin\theta\cos\theta}"
              r"{m_c+m_p\sin^2\theta}", 21, INK)
        math_(cv, 110, 546, r"\ddot\theta=\frac{g\sin\theta-\ddot p\cos\theta"
                            r"-b_p\omega/(m_p l)}{l}", 21, INK)
    a4 = appear(t, 4.6, 0.8)
    if a4 > 0:
        w = math_w(r"m_c+m_p\sin^2\theta\in[\,1.00,\ 1.25\,]", 20)
        cv.rect(736, 600, 736 + w + 24, 648, AM, width=1.2, alpha=0.6 * a4,
                fill_alpha=0.07 * a4, radius=6)
        math_(cv, 748, 632, r"m_c+m_p\sin^2\theta\in[\,1.00,\ 1.25\,]", 20, AM)
        cv.text(110, 640, "分母恒正 → 小车通道永远可以精确反解", size=24, color=DIM, alpha=a4)


# ============================================================ S6
def s6(cv, t, dur):
    title(cv, "能量：把“摆起来”压缩成一个标量")
    a = appear(t, 0.3, 0.9)
    if a > 0:
        math_(cv, 110, 226, r"E=\tfrac{1}{2}m_p l^2\omega^2+m_p g l\,(\cos\theta-1)",
              25, INK)
    a2 = appear(t, 1.8, 0.7)
    if a2 > 0:
        cv.circle(118, 300, 5, CY, width=0, alpha=a2, fill_alpha=a2)
        math_(cv, 140, 312, r"E=0", 22, CY)
        cv.text(238, 312, "→ 静止于竖直向上", size=24, color=DIM, alpha=a2)
    a3 = appear(t, 3.0, 0.7)
    if a3 > 0:
        cv.circle(118, 356, 5, MG, width=0, alpha=a3, fill_alpha=a3)
        math_(cv, 140, 372, r"E=-2.943\ \mathrm{J}", 22, MG)
        cv.text(370, 372, "→ 静止下垂", size=24, color=DIM, alpha=a3)
    a4 = appear(t, 4.4, 0.8)
    cv.text(110, 470, "目标从“轨迹”换成一个标量：把 E 推到 0", size=25, color=DIM, alpha=a4)

    if a > 0:
        x0, y0, x1, y1 = 762, 238, 1136, 468
        emin = -2.943
        px = lambda th: x0 + (th + math.pi) / (2 * math.pi) * (x1 - x0)
        py = lambda E: y1 - (E - emin) / (0 - emin) * (y1 - y0)
        ax = (92, 110, 132)
        cv.arrow((x0, y1), (x1 + 14, y1), ax, width=1.4, alpha=0.85)
        cv.arrow((x0, y1), (x0, y0 - 14), ax, width=1.4, alpha=0.85)
        for th, lab in ((-math.pi, "−π"), (0, "0"), (math.pi, "π")):
            cv.line((px(th), y1), (px(th), y1 + 5), ax, width=1.2, alpha=0.8)
            cv.text(px(th), y1 + 26, lab, size=17, color=FAINT, anchor="ma")
        for E, lab in ((0.0, "0"), (emin, "−2.943")):
            cv.line((x0 - 5, py(E)), (x0, py(E)), ax, width=1.2, alpha=0.8)
            cv.text(x0 + 9, py(E) + 6, lab, size=16, color=FAINT)
            cv.line((x0, py(E)), (x1, py(E)), (78, 96, 116), width=1, alpha=0.45, dash=6)
        cv.text(x1 + 18, y1 + 6, "θ [rad]", size=17, color=DIM)
        cv.text(x0 + 9, y0 - 6, "E [J]", size=17, color=DIM)
        N = 600
        pts = [(px(-math.pi + 2 * math.pi * i / N),
                py(emin * (1 - math.cos(-math.pi + 2 * math.pi * i / N)) / 2)) for i in range(N + 1)]
        for i in range(len(pts) - 1):
            cv.line(pts[i], pts[i + 1], CY, width=2, alpha=0.8 * a)
        thc = 2.4 * math.sin(0.9 * t)
        cv.circle(px(thc), py(emin * (1 - math.cos(thc)) / 2), 6, AM, width=0, alpha=a, fill_alpha=a)


# ============================================================ S7
def s7(cv, t, dur):
    title(cv, "线性化：一个正根说明一切")
    a = appear(t, 0.3, 0.8)
    if a > 0:
        math_(cv, 130, 180, r"A=", 26, DIM, anchor="ms")
        math_(cv, 190, 300,
              r"\begin{bmatrix} 0 & 0 & 1 & 0\\ 0 & 0 & 0 & 1\\ "
              r"0 & -2.4525 & -0.12 & 0.004\\ 0 & 16.35 & 0.20 & -0.0511 \end{bmatrix}",
              19, INK, anchor="lc")
    a2 = appear(t, 1.6, 0.7)
    if a2 > 0:
        math_(cv, 640, 180, r"B=", 26, DIM, anchor="ms")
        math_(cv, 700, 300, r"\begin{bmatrix} 0\\ 0\\ 1\\ -1.667 \end{bmatrix}", 19,
              INK, anchor="lc")
    a3 = appear(t, 3.0, 0.8)
    if a3 > 0:
        p1 = math_(cv, 120, 470, r"\lambda=0,\quad", 22, INK)
        qx = p1[2] + 6
        p2 = math_(cv, qx, 470, r"+4.0034", 22, AM)
        p3 = math_(cv, p2[2] + 6, 470, r",\quad -4.0845,\quad -0.09", 22, INK)
        cv.rect(p2[0] - 8, p2[1] - 7, p2[2] + 8, p2[3] + 7,
                AM, width=1.4, alpha=0.7 * a3, fill_alpha=0.06 * a3, radius=5)
        math_(cv, qx, 470, r"+4.0034", 22, AM, check=False)
    a4 = appear(t, 4.6, 0.8)
    cv.text(120, 540, "时间常数　1 / 4.00 = 0.25 s", size=25, color=AM, alpha=a4)
    cv.text(120, 582, "1° 倾角，0.5 s 后 → 4.8°", size=23, color=DIM, alpha=a4)

    a5 = appear(t, 3.6, 0.9)
    if a5 > 0:
        x0, y0, x1, y1 = 762, 404, 1136, 596
        ax = (92, 110, 132)
        TMAX, YMAX = 0.8, 14.0
        px = lambda tt: x0 + tt / TMAX * (x1 - x0)
        py = lambda dg: y1 - dg / YMAX * (y1 - y0)
        cv.arrow((x0, y1), (x1 + 14, y1), ax, width=1.4, alpha=0.85 * a5)
        cv.arrow((x0, y1), (x0, y0 - 14), ax, width=1.4, alpha=0.85 * a5)
        for tt in (0.0, 0.25, 0.5, 0.75):
            cv.line((px(tt), y1), (px(tt), y1 + 5), ax, width=1.2, alpha=0.8 * a5)
            cv.text(px(tt), y1 + 26, "%.2f" % tt if tt else "0", size=15,
                    color=FAINT, anchor="ma")
        for dg in (0, 5, 10):
            cv.line((x0 - 5, py(dg)), (x0, py(dg)), ax, width=1.2, alpha=0.8 * a5)
            cv.text(x0 + 9, py(dg) + 6, str(dg), size=15, color=FAINT)
            cv.line((x0, py(dg)), (x1, py(dg)), (78, 96, 116), width=1, alpha=0.4, dash=6)
        cv.text(x1 + 18, y1 + 6, "t [s]", size=17, color=DIM, alpha=a5)
        cv.text(x0 + 9, y0 - 6, "θ [°]", size=17, color=DIM, alpha=a5)
        idx = TILT[:, 0] <= TMAX
        pts = [(px(tt), py(dg)) for tt, dg in TILT[idx]]
        for i in range(len(pts) - 1):
            cv.line(pts[i], pts[i + 1], AM, width=2.2, alpha=0.9 * a5)
        j = int(np.argmin(np.abs(TILT[:, 0] - 0.5)))
        cv.circle(px(TILT[j, 0]), py(TILT[j, 1]), 5, AM, width=0, alpha=a5, fill_alpha=a5)


# ============================================================ S8
def s8(cv, t, dur):
    title(cv, "本讲小结")
    items = [("运动方程", "两行耦合方程，非线性全在 sinθ、cosθ、ω²"),
             ("分母恒正", "m_c + m_p sin²θ ∈ [1.00, 1.25] → 小车通道可反解"),
             ("失去权限", "cosθ 是摆杆通道的系数 → 水平位置无法直接控制"),
             ("一个标量", "能量 E 把“甩起来”变成“把 E 推到 0”"),
             ("一个正根", "+4.00 → 时间常数 0.25 s → 局部不稳定")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.7 * i, 0.6)
        if aa <= 0:
            continue
        y = 200 + i * 62
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=25, color=CY, alpha=aa, weight="bold")
        cv.text(292, y, v, size=24, color=INK, alpha=aa)
    a = appear(t, 4.6, 0.8)
    cv.text(130, 556, "下一讲：LQR 与它的能力边界", size=27, color=AM, alpha=a)
    cv.text(130, 602, "作业：复现推导，并用 3 个自检验证你的仿真器", size=21, color=FAINT, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "欢迎来到这门课。我们从最基础的一步开始：把一台机器写成方程。",
        "整门课只用一台小车倒立摆：一个只能左右推的小车，上面立着一根杆。",
        "后面的每一讲，我们都会回到这一讲推出来的这两行方程。",
    ], draw=s1, min_dur=11.0),
    dict(name="system", say=[
        {"say": "$p$ 是小车位置，向右为正。", "tts": "p 是小车位置，向右为正。"},
        {"say": "$\\theta$ 是摆杆与竖直向上方向的夹角，向右倾为正。注意它从竖直向上量起。",
         "tts": "西塔是摆杆与竖直向上方向的夹角，向右倾为正。注意它从竖直向上量起。"},
        {"say": "$l$ 是转轴到摆杆质心的距离（$0.60\\ \\mathrm{m}$），$u$ 是作用在小车上的水平力。",
         "tts": "l 是转轴到摆杆质心的距离，零点六米；u 是作用在小车上的水平力。"},
        "假设摆杆是一个质点，暂时忽略电机动态和传动柔性。",
    ], draw=s2, min_dur=22.0),
    dict(name="params", say=[
        {"say": "把参数一次列全：小车质量 $1.0$ 千克，摆杆质点 $0.25$ 千克，杆长 $0.6$ 米。",
         "tts": "把参数一次列全：小车质量一点零千克，摆杆质点零点二五千克，杆长零点六米。"},
        {"say": "重力加速度 $9.81$；两处粘性阻尼：导轨 $0.12$，转轴 $0.004$。",
         "tts": "重力加速度九点八一；两处粘性阻尼：导轨零点一二，转轴零点零零四。"},
        {"say": "执行器上限 $12$ 牛顿，导轨半长 $2.4$ 米，控制周期 $20$ 毫秒。",
         "tts": "执行器上限十二牛顿，导轨半长二点四米，控制周期二十毫秒。"},
        "这九个数字决定了后面所有实验的可比性。",
    ], draw=s_params, min_dur=19.0),
    dict(name="coords", say=[
        "再强调两条容易搞反的约定。",
        {"say": "第一，$\theta$ 等于零是竖直向上，也就是不稳定平衡；这和我们习惯的“垂下为零”相反。",
         "tts": "第一，theta 等于零是竖直向上，也就是不稳定平衡；这和我们习惯的垂下为零相反。"},
        "第二，力向右为正，但让摆向右倾需要把车向右推——所以 u 与 θ 同号时，摆反而更快倒下。",
        "这两条弄对，后面的符号才不会错。",
    ], draw=s_coords, min_dur=18.0),
    dict(name="underactuated", say=[
        {"say": "为什么说它难？因为只有 1 个输入，却有 2 个自由度。",
         "tts": "为什么说它难？因为只有一个输入，却有两个自由度。"},
        {"say": "把输出取成小车位置和摆角，各求 2 次导数，你会看到解耦矩阵其实是一个列向量。",
         "tts": "把输出取成小车位置和摆角，各求两次导数，你会看到解耦矩阵其实是一个列向量。"},
        {"say": "列向量的秩至多是 1，所以任何力，都不可能同时指定小车加速度和摆杆角加速度。",
         "tts": "列向量的秩至多是一，所以任何力，都不可能同时指定小车加速度和摆杆角加速度。"},
        "这就是欠驱动：你的控制权限，天生不够用。",
    ], draw=s3, min_dur=17.0),
    dict(name="lagrange", say=[
        {"say": "写出动能、势能和耗散函数，只需要 3 行。",
         "tts": "写出动能、势能和耗散函数，只需要三行。"},
        {"say": "动能里有一项 $m_p l\\cos\\theta\\,v\\,\\omega$：推小车会改变摆杆的运动，反过来也一样。",
         "tts": "动能里有一项交叉项，是速度 v 与角速度 omega 的乘积，再乘上 cos theta：推小车会改变摆杆的运动，反过来也一样。"},
        "这一项是耦合项。后面所有控制器的补偿设计，本质上都在处理它。",
        {"say": "代入欧拉-拉格朗日方程，就得到 2 个互相耦合的二阶方程。",
         "tts": "代入欧拉-拉格朗日方程，就得到两个互相耦合的二阶方程。"},
    ], draw=s4, min_dur=18.0),
    dict(name="solve", say=[
        {"say": "联立这 2 个方程，把加速度解出来。", "tts": "联立这两个方程，把加速度解出来。"},
        {"say": "小车加速度等于一个分数：分子是力、摩擦、耦合项和重力，分母是 $m_c+m_p\\sin^2\\theta$。",
         "tts": "小车加速度等于一个分数：分子是力、摩擦、耦合项和重力，分母是 m c 加上 m p sin 平方 theta。"},
        {"say": "注意这个分母：取值范围是 $1.00$ 到 $1.25$，永远为正。",
         "tts": "注意这个分母，它的取值范围是一点零零到一点二五，永远为正。"},
        {"say": "这意味着小车通道永远可以精确反解，第 3 讲会把这一点用到极致。",
         "tts": "这意味着小车通道永远可以精确反解，第三讲会把这一点用到极致。"},
    ], draw=s5, min_dur=18.0),
    dict(name="energy", say=[
        "接下来是这一讲最重要的工具：能量。",
        "定义摆杆相对直立位置的能量，它等于动能项加上重力势能项。",
        {"say": "$E=0$ 表示静止于竖直向上；$E=-2.943\\ \\mathrm{J}$ 表示静止下垂。",
         "tts": "能量等于零，表示静止于竖直向上；能量等于负二点九四焦耳，表示静止下垂。"},
        {"say": "于是把摆起来这个任务，就被压缩成了一个标量：把能量推到 $0$。",
         "tts": "于是把摆甩起来这个任务，就被压缩成了一个标量：把能量推到零。"},
    ], draw=s6, min_dur=17.0),
    dict(name="linearize", say=[
        "在直立点线性化，得到这两个矩阵。",
        {"say": "有两个项特别容易漏：摆杆也会拉小车，所以 $A$ 的第 3 行第 2 列不是 $0$；"
                "转轴摩擦也会通过小车加速度，间接进入摆角方程。",
         "tts": "有两个项特别容易漏：摆杆也会拉小车，所以 A 的第三杭第二列不是零；"
                "转轴摩擦也会通过小车加速度，间接进入摆角方程。"},
        {"say": "特征值里有一个正实根：$+4.00$。", "tts": "特征值里有一个正实根，正四点零零。"},
        {"say": "它的倒数是 $0.25\\ \\mathrm{s}$：偏差每 $0.25$ 秒放大 $2.7$ 倍。",
         "tts": "它的倒数是零点二五秒，表示偏差每四分之一秒放大二点七倍。"},
        {"say": "$1^\\circ$ 倾角，$0.5$ 秒后就变成 $4.8^\\circ$。",
         "tts": "一度倾角，半秒后就变成四点八度。"},
        "所以线性化可以精确描述局部行为，但不能预测大范围运动。",
    ], draw=s7, min_dur=24.0),
    dict(name="summary", say=[
        "这一讲我们得到了完整模型、一个能量标量，和一个不稳定特征根。",
        "还有 3 条结构性事实：分母恒正、余弦会失去权限、输入只有一个。",
        "下一讲，我们用最经典的方法先在局部把它稳住：LQR。",
        "然后你会发现，它的能力边界，正好就是这一讲所说的那个局部。",
    ], draw=s8, min_dur=20.0),
]
