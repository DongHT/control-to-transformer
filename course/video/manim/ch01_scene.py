"""
第 1 讲 manim 版场景（3b1b 风格，与 render.py 出的成片同脚本、同画面语言）。

本机无法运行 manim：macOS 上 manim 依赖 pycairo，而 pycairo 在 macOS 没有
wheel、需要系统 libcairo（本机无 brew）。因此本仓库的做法是：

    路径 A（本机可跑）  : render.py   —— Pillow 逐帧绘制 + PyAV 编码 + say 配音
    路径 B（manim 环境）: 本文件      —— manim 渲染画面，再用 mux_voiceover.py 加旁白/配乐/字幕

在有 manim 的机器上：
    pip install manim
    manim -qh ch01_scene.py Ch01 -o ch01_silent.mp4
    python3 ../mux_voiceover.py ../out/ch01_silent.mp4 ch01_narration.json

注：为避免 LaTeX 依赖（很多机器没装 TeX），公式一律用 Text + 数学符号排版，
字体统一 Songti SC，这也是 render.py 用的字体，保证两版画面一致。
"""
from manim import *          # noqa

FONT = "Songti SC"
CY, AM, MG, BL = "#2fd3c0", "#f2a93b", "#e0619e", "#6aa9f5"
INK, DIM, FAINT = "#e8f0f8", "#96a6b7", "#60708b"

# 与 render.py 完全一致的旁白（用于字幕/配音对齐）
NARRATION = [
    "欢迎来到这门课。我们从最基础的一步开始：把一台机器写成方程。",
    "整门课只用一台小车倒立摆：一个只能左右推的小车，上面立着一根杆。",
    "后面的每一讲，我们都会回到这一讲推出来的这两行方程。",
]


def T(s, size=30, color=INK, **kw):
    return Text(s, font=FONT, font_size=size, color=color, **kw)


class Ch01(Scene):
    def construct(self):
        self.camera.background_color = "#0a0f18"
        self.shot_title()
        self.shot_system()
        self.shot_underactuated()
        self.shot_lagrange()
        self.shot_solve()
        self.shot_energy()
        self.shot_linearize()
        self.shot_summary()

    # ---------------------------------------------------------------- 1
    def shot_title(self):
        kicker = T("倒立摆控制　·　从拉格朗日到 Transformer", 24, CY)
        title = T("第 1 讲　建模", 66, INK, weight=BOLD)
        sub = T("把一台机器写成方程", 30, DIM)
        rule = Line(LEFT * 1.1, RIGHT * 1.1, color=CY, stroke_width=2)
        foot = T("课程讲义 · ch01    代码 · build/plant.py", 18, FAINT).to_corner(DL, buff=0.6)
        grp = VGroup(kicker, title, sub, rule).arrange(DOWN, buff=0.42).move_to(UP * 0.35)
        self.play(FadeIn(kicker, shift=DOWN * 0.2), run_time=1.0)
        self.play(Write(title), run_time=1.4)
        self.play(FadeIn(sub, shift=UP * 0.15), Create(rule), run_time=0.9)
        self.play(FadeIn(foot), run_time=0.6)
        self.wait(6.0)

    # ---------------------------------------------------------------- 2
    def shot_system(self):
        heading = T("坐标约定与三条假设", 38, INK, weight=BOLD).to_corner(UL, buff=0.6)
        self.play(FadeIn(heading), run_time=0.6)

        theta = ValueTracker(2.5)
        rail = Line(LEFT * 3.4, RIGHT * 3.4, color=FAINT, stroke_width=2)
        cart = RoundedRectangle(width=1.05, height=0.5, corner_radius=0.08,
                               stroke_color=BL, fill_color="#1b2b3a", fill_opacity=0.7)
        pole = always_redraw(lambda: Line(
            cart.get_top(),
            cart.get_top() + rotate_vector(UP * 1.45, theta.get_value()),
            color=CY, stroke_width=6))
        bob = always_redraw(lambda: Dot(
            cart.get_top() + rotate_vector(UP * 1.45, theta.get_value()),
            radius=0.09, color=CY))
        vref = always_redraw(lambda: DashedLine(
            cart.get_top(), cart.get_top() + UP * 1.9, color=FAINT, stroke_width=1.5))
        rig = VGroup(rail, cart, vref, pole, bob).move_to(LEFT * 3.4 + DOWN * 1.2)
        rail.shift(UP * 0.25)
        cart.next_to(rail, UP, buff=0.02)

        self.play(Create(rail), FadeIn(cart), run_time=0.8)
        self.add(vref, pole, bob)
        self.play(FadeIn(T("无驱动：自由倒下", 18, FAINT).next_to(rig, UP, buff=0.45)))
        self.play(theta.animate.set_value(3.2), run_time=5.0, rate_func=linear)

        rows = [("p", "小车位置　向右为正", BL), ("θ", "从竖直向上量起", AM),
                ("l", "转轴到质心　0.60 m", CY), ("u", "水平力　|u| ≤ 12 N", MG)]
        lines = VGroup()
        for sym, txt, col in rows:
            lines.add(VGroup(T(sym, 28, col, weight=BOLD), T(txt, 22, INK)).arrange(RIGHT, buff=0.35))
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.34).to_edge(RIGHT, buff=0.9).shift(UP * 0.6)
        self.play(LaggedStartMap(FadeIn, lines, lag_ratio=0.35), run_time=2.2)
        self.play(FadeIn(T("v = dp/dt,　ω = dθ/dt", 20, DIM).next_to(lines, DOWN, buff=0.4)))
        self.wait(4.0)

    # ---------------------------------------------------------------- 3
    def shot_underactuated(self):
        heading = T("一个输入，两个自由度", 38, INK, weight=BOLD).to_corner(UL, buff=0.6)
        self.play(FadeIn(heading))
        vec = VGroup(
            T("Δ(x) =", 30, INK),
            VGroup(
                T("1 /(m_c + m_p sin²θ)", 21, INK),
                T("−cosθ /( l (m_c + m_p sin²θ) )", 21, INK),
            ).arrange(DOWN, buff=0.35).scale(1.0),
        ).arrange(RIGHT, buff=0.45).move_to(UP * 1.1)
        box = SurroundingRectangle(vec[1], color=MG, buff=0.22)
        self.play(FadeIn(vec[0]), Write(vec[1]), run_time=1.6)
        self.play(Create(box), run_time=0.6)
        self.play(FadeIn(T("列向量　秩 = 1", 26, MG).next_to(box, RIGHT, buff=0.6)))
        self.play(FadeIn(T("→ 对任何 u，都无法同时指定 p 的二阶导 和 θ 的二阶导", 24, INK)
                         .next_to(vec, DOWN, buff=0.75)))
        chips = VGroup(*[T(s, 22, c) for s, c in
                         [("欠驱动", MG), ("本质非线性", AM), ("开环不稳定", CY)]])
        chips.arrange(RIGHT, buff=0.5).to_edge(DOWN, buff=1.5)
        self.play(LaggedStartMap(FadeIn, chips, lag_ratio=0.3))
        self.wait(4.5)

    # ---------------------------------------------------------------- 4
    def shot_lagrange(self):
        heading = T("拉格朗日方程：三行写出模型", 38, INK, weight=BOLD).to_corner(UL, buff=0.6)
        self.play(FadeIn(heading))
        t_line = T("T = ½(m_c + m_p) v² + m_p l cosθ · v · ω + ½ m_p l² ω²", 27, INK)
        t_line.move_to(UP * 1.5).to_edge(LEFT, buff=0.9)
        self.play(Write(t_line), run_time=1.8)
        # 高亮耦合项：用一段与子串等宽的矩形近似
        hl = Rectangle(width=3.05, height=0.55, color=AM, stroke_width=1.6,
                       fill_color=AM, fill_opacity=0.08)
        hl.move_to(t_line).shift(RIGHT * 0.35)
        self.play(Create(hl), run_time=0.6)
        self.play(FadeIn(T("耦合项", 20, AM).next_to(hl, DOWN, buff=0.25)))
        v_line = T("V = m_p g l cosθ", 27, INK).next_to(t_line, DOWN, aligned_edge=LEFT, buff=0.75)
        r_line = T("R = ½ b_c v² + ½ b_p ω²", 27, INK).next_to(v_line, DOWN, aligned_edge=LEFT, buff=0.55)
        self.play(Write(VGroup(v_line, r_line)), run_time=1.6)
        el = T("d/dt (∂L/∂q̇) − ∂L/∂q + ∂R/∂q̇ = Q", 21, DIM).to_edge(RIGHT, buff=0.8).shift(UP * 0.9)
        self.play(FadeIn(el))
        self.wait(4.5)

    # ---------------------------------------------------------------- 5
    def shot_solve(self):
        heading = T("两个耦合方程 → 解出加速度", 38, INK, weight=BOLD).to_corner(UL, buff=0.6)
        self.play(FadeIn(heading))
        eq_a = VGroup(T("小车", 18, FAINT),
                      T("(m_c+m_p) p̈ + m_p l cosθ · θ̈", 21, INK),
                      T("  − m_p l sinθ · ω² = u − b_c v", 21, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        eq_b = VGroup(T("摆杆", 18, FAINT),
                      T("m_p l cosθ · p̈ + m_p l² θ̈", 21, INK),
                      T("  + m_p g l sinθ = − b_p ω", 21, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        left = VGroup(eq_a, eq_b).arrange(DOWN, aligned_edge=LEFT, buff=0.55).to_edge(LEFT, buff=0.9).shift(UP * 0.6)
        self.play(FadeIn(left), run_time=1.4)
        arrow = Arrow(left.get_right(), left.get_right() + RIGHT * 1.1, color=CY, buff=0.2)
        self.play(GrowArrow(arrow))
        res = VGroup(T("p̈ = [ u − b_c v + m_p l sinθ · ω²", 20, INK),
                     T("      + b_p cosθ · ω − m_p g sinθ cosθ ]", 20, INK),
                     T("      / ( m_c + m_p sin²θ )", 24, CY)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        res.next_to(arrow, RIGHT, buff=0.5).shift(UP * 0.75)
        self.play(FadeIn(res), run_time=1.4)
        self.play(FadeIn(T("m_c + m_p sin²θ ∈ [1.00, 1.25]", 24, AM).next_to(res, DOWN, buff=0.7)))
        self.wait(5.0)

    # ---------------------------------------------------------------- 6
    def shot_energy(self):
        heading = T("能量：把“摆起来”压缩成一个标量", 38, INK, weight=BOLD).to_corner(UL, buff=0.6)
        self.play(FadeIn(heading))
        formula = T("E = ½ m_p l² ω² + m_p g l ( cosθ − 1 )", 30, INK).next_to(heading, DOWN, buff=0.8).to_edge(LEFT, buff=0.9)
        self.play(Write(formula), run_time=1.6)
        zero = T("E = 0　↔　静止于竖直向上", 24, CY).next_to(formula, DOWN, aligned_edge=LEFT, buff=0.5)
        hang = T("E = −2.943 J　↔　静止下垂", 24, MG).next_to(zero, DOWN, aligned_edge=LEFT, buff=0.25)
        self.play(FadeIn(zero), FadeIn(hang), run_time=1.0)

        axes = Axes(x_range=[-PI, PI, PI / 2], y_range=[-3.0, 0.2, 1.0],
                    x_length=5.2, y_length=2.9, tips=False,
                    axis_config={"color": FAINT, "stroke_width": 1.2})
        axes.to_edge(RIGHT, buff=0.7).shift(DOWN * 0.3)
        curve = axes.plot(lambda x: -2.943 * (1 - np.cos(x)) / 2, x_range=[-PI, PI], color=CY)
        self.play(Create(axes), Create(curve), run_time=1.8)
        tr = ValueTracker(-PI)
        dot = always_redraw(lambda: Dot(axes.c2p(tr.get_value(),
                            -2.943 * (1 - np.cos(tr.get_value())) / 2), radius=0.07, color=AM))
        self.add(dot)
        self.play(tr.animate.set_value(PI), run_time=4.5, rate_func=linear)
        self.wait(2.5)

    # ---------------------------------------------------------------- 7
    def shot_linearize(self):
        heading = T("线性化：一个正根说明一切", 38, INK, weight=BOLD).to_corner(UL, buff=0.6)
        self.play(FadeIn(heading))
        elem = lambda v: Text(str(v), font=FONT, font_size=20, color=INK)
        A = Matrix([["0", "0", "1", "0"],
                    ["0", "0", "0", "1"],
                    ["0", "−2.45", "−0.12", "0.004"],
                    ["0", "16.35", "0.20", "−0.051"]],
                   element_to_mobject=elem, h_buff=1.0, v_buff=0.5)
        A.scale(0.9).to_edge(LEFT, buff=0.9).shift(UP * 0.2)
        self.play(FadeIn(A), run_time=1.2)

        B = Matrix([["0"], ["0"], ["1"], ["−1.667"]],
                   element_to_mobject=elem, h_buff=1.2, v_buff=0.5)
        B.scale(0.9).next_to(A, RIGHT, buff=0.8)
        self.play(FadeIn(B), run_time=0.9)

        eig = VGroup(T("特征值　λ = 0,", 23, INK),
                     T("+4.0034", 23, AM),
                     T(",　−4.0845,　−0.09", 23, INK)).arrange(RIGHT, buff=0.12)
        eig.next_to(A, DOWN, buff=0.7).to_edge(LEFT, buff=0.9)
        self.play(FadeIn(eig), run_time=1.0)
        hl = SurroundingRectangle(eig[1], color=AM, buff=0.12)
        self.play(Create(hl), run_time=0.5)

        tau = T("时间常数　1 / 4.00 = 0.25 s", 24, AM).next_to(eig, DOWN, aligned_edge=LEFT, buff=0.45)
        deg = T("1° 倾角，0.5 s 后 → 4.8°", 22, DIM).next_to(tau, DOWN, aligned_edge=LEFT, buff=0.2)
        self.play(FadeIn(tau), FadeIn(deg), run_time=0.8)

        axes = Axes(x_range=[0, 3, 1], y_range=[0, 8, 2],
                    x_length=4.4, y_length=2.6, tips=False,
                    axis_config={"color": FAINT, "stroke_width": 1.2})
        axes.to_edge(RIGHT, buff=0.8).shift(DOWN * 0.9)
        curve = axes.plot(lambda x: np.exp(2.4 * x / 3 * 2.1) - 1, x_range=[0, 3], color=AM)
        self.play(Create(axes), Create(curve), run_time=1.5)
        self.wait(4.0)

    # ---------------------------------------------------------------- 8
    def shot_summary(self):
        heading = T("本讲小结", 38, INK, weight=BOLD).to_corner(UL, buff=0.6)
        self.play(FadeIn(heading))
        items = [("运动方程", "两行耦合方程，非线性全在 sinθ、cosθ、ω²"),
                 ("分母恒正", "m_c + m_p sin²θ ∈ [1.00, 1.25] → 小车通道可反解"),
                 ("失去权限", "cosθ 是摆杆通道的系数 → 水平位置无法直接控制"),
                 ("一个标量", "能量 E 把“甩起来”变成“把 E 推到 0”"),
                 ("一个正根", "+4.00 → 时间常数 0.25 s → 局部不稳定")]
        rows = VGroup()
        for k, v in items:
            rows.add(VGroup(T("✓", 22, CY), T(k, 22, CY, weight=BOLD), T(v, 20, INK))
                    .arrange(RIGHT, buff=0.3, aligned_edge=DOWN))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.34).next_to(heading, DOWN, buff=0.7)
        rows.to_edge(LEFT, buff=0.9)
        self.play(LaggedStartMap(FadeIn, rows, lag_ratio=0.35), run_time=2.4)
        nxt = T("下一讲：LQR 与它的能力边界", 26, AM).to_edge(DOWN, buff=1.1).to_edge(LEFT, buff=0.9)
        self.play(FadeIn(nxt), run_time=0.8)
        self.wait(4.0)
