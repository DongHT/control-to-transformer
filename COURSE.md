# 倒立摆控制：从拉格朗日到 Transformer

**一部可以逐行验证的完整教程。** 同一台小车倒立摆，用三种方法论各做一遍：
经典非线性控制 → 模仿学习与强化学习 → 从零手写那个 Transformer。
每一步都有公式、代码、真实数字，以及配套视频。

```
建模 → LQR（划出能力边界）→ 反馈线性化 → 能量整定 → 滑模 → 切换
     → 数据与模仿 → Transformer → DAgger → PPO
     → 从零实现：前向 / 反向 / AdamW / 部署
```

---

## 一、从哪里开始

<table>
<tr><th>你的背景</th><th>推荐路径</th><th>时间</th></tr>
<tr><td>控制理论（熟悉 LQR/滑模）</td><td>第 1–6 章 → 第 7、9 章 → 第 11–17 章</td><td>约 3 小时</td></tr>
<tr><td>深度学习（熟悉 Transformer）</td><td>第 1 章 → 第 2 章（能力边界）→ 第 7–10 章</td><td>约 2 小时</td></tr>
<tr><td>只想搞懂 Transformer 实现</td><td>第 13（numpy 手写反向）、15（优化器）章 + 视频 14–17</td><td>约 1.5 小时</td></tr>
<tr><td>想动手做</td><td>`index.html` → `build/` 与 `tutorial/code/` 逐个脚本跑</td><td>半天</td></tr>
</table>

---

## 二、三种用法：看文字 / 看视频 / 动手

### ① 想看文字

**不想编译**：直接下载编译好的 PDF（97 页）——
[https://github.com/DongHT/control-to-transformer/releases/tag/v1.0](https://github.com/DongHT/control-to-transformer/releases/tag/v1.0)

**想自己编**（三篇 + 附录，97 页）：

```bash
cd book && xelatex main.tex && xelatex main.tex && open main.pdf
# 没有 TeX？用 tectonic：./tectonic -X compile main.tex
```

| 想看的范围 | 直接读源文件 |
|---|---|
| 全书（推荐） | `book/main.tex`（它 `\input` 了下面所有内容） |
| 第 1–10 讲 + 附录 A | `course/latex/chapters/ch01.tex` … `ch10.tex`、`chA.tex` |
| 第 11–17 集（Transformer 实现） | `tutorial/latex/tutorial.tex`（或 `book/chapters/p3ch01–07.tex`） |
| 课程大纲 | `course/README.md` |
| 一张纸的速览 | `COURSE.md`（本文件） |

### ② 想看视频

20 集，一共 **61 分 43 秒**。**按编号顺序看**即可，编号就是依赖顺序：

| 顺序 | 文件 | 时长 | 看完你应该知道 |
|---|---|---|---|
| 1 | `course/out/ch01_建模.mp4` | 3:53 | 方程从哪来、为什么难 |
| 2 | `course/out/ch02_线性基线.mp4` | 3:12 | LQR 的能力边界（25% 成功率） |
| 3 | `course/out/ch03_反馈线性化.mp4` | 2:06 | 把力换成加速度 |
| 4 | `course/out/ch04_能量整定.mp4` | 2:12 | 怎么把摆甩上去 |
| 5 | `course/out/ch05_滑模.mp4` | 2:00 | 怎么顶住扰动 |
| 6 | `course/out/ch06_切换.mp4` | 1:33 | 怎么把两者缝起来 |
| 7 | `course/out/ch07_数据与模仿.mp4` | 2:33 | 为什么 BC 会失败 |
| 8 | `course/out/ch08_Transformer.mp4` | 2:27 | 这个网络长什么样 |
| 9 | `course/out/ch09_DAgger与评测.mp4` | 2:23 | 怎么修分布漂移、怎么验收 |
| 10 | `course/out/ch10_强化学习.mp4` | 3:03 | RL 能超专家，但会崩在分布外 |
| A | `course/out/chA_Transformer内部机制.mp4` | 3:16 | 逐层前向/反向的真实数值 |
| 11 | `tutorial/out/t1_为什么长这样.mp4` | 2:07 | 注意力的动机与 √d |
| 12 | `tutorial/out/t2_前向与反向.mp4` | 2:23 | 两个雅可比 |
| 13 | `tutorial/out/t3_两种实现.mp4` | 2:35 | numpy 与 torch 对齐 |
| 14 | `tutorial/out/t4_前向逐算子.mp4` | 3:39 | 9 个算子的公式 ↔ 代码 |
| 15 | `tutorial/out/t5_反向逐算子.mp4` | 3:51 | 每个雅可比的推导 ↔ 代码 |
| 16 | `tutorial/out/t6_串起来.mp4` | 3:21 | 一次迭代 → 训练 → 部署 |
| 17 | `tutorial/out/t7_训练方法.mp4` | 3:37 | SGD→AdamW 为什么这么选 |

每集都配同名 `.srt` 字幕。**只要 5 分钟**的话：看 1、2、7、11（约 12 分钟）。

### ③ 想动手复现

按这个顺序跑，每步都能独立验证：

```bash
# 0) 依赖
pip install numpy scipy torch pillow av edge-tts

# 1) 30 秒：验算手写反向传播（这是全书的地基）
cd tutorial/code && python3 parity_test.py
#    → 期望：全部参数梯度最大偏差 3.0e-14

# 2) 2 分钟：五种控制器统一评测（复现结论表）
cd ../../build && python3 eval_all.py

# 3) 1 分钟：LQR 的吸引域扫描（第 2 讲的核心图）
python3 ch02_experiments.py

# 4) 3 分钟：BC + DAgger 训练（第 7、9 讲）
python3 train.py

# 5) 10 分钟：PPO 微调 + 域随机化（第 10 讲）
python3 rl.py

# 6) 40 秒：优化器对比（第 17 集）
cd ../tutorial/code && python3 optim_demo.py

# 7) 5 分钟：重渲任意一集视频
cd ../../course/video && python3 make_ch03.py
```

**只想看结果、不动手**：双击 `index.html`，浏览器里实时跑五种控制器。

## 三、全书结构（三篇 + 附录）

### 第一篇　建模与经典控制（第 1–6 章 / 视频 1–6）

| 章 | 内容 | 核心结论 | 视频 |
|---|---|---|---|
| 1 | 建模：系统标注、9 个参数、拉格朗日、能量、线性化 | 正特征根 +4.00，不稳定时间常数 0.25 s | 4:49 |
| 2 | 线性基线：LQR 与能力边界 | 吸引域 0.64 rad；全局成功率仅 **25%** | 3:12 |
| 3 | 部分反馈线性化 | 反解误差 9.2e-4；能量泵 0/3 → 3/3 | 2:06 |
| 4 | 能量整定 | 摆起 1.98 s；k_E 太小抓不住、太大撞墙 | 2:12 |
| 5 | 滑模控制 | 抗扰 0.0121 → **0.0053** rad（2.3×） | 2:00 |
| 6 | 切换与监督 | 捕获窗 0.40 rad；切换瞬态 Δu ≈ 11 N | 1:33 |

### 第二篇　从数据到 Transformer（第 7–10 章 / 视频 7–10）

| 章 | 内容 | 核心结论 | 视频 |
|---|---|---|---|
| 7 | 数据与模仿 | 漂移界 ε(e^{Lt}−1)/L；纯 BC 在 **0.10 s** 就跑偏 | 2:33 |
| 8 | 因果 Transformer | 18,177 参数；int8 后 16.6 KB | 2:27 |
| 9 | DAgger 与评测 | O(εT²) → O(εT)；30/30 的 95% 区间 [0.886, 1] | 2:23 |
| 10 | 策略梯度与 PPO | 摆起 −22%、力 −15%；**+4N 侧风只剩 3%** | 3:03 |

### 第三篇　Transformer 从零实现（第 11–17 章 / 视频 11–17）

| 章 | 内容 | 核心结论 | 视频 |
|---|---|---|---|
| 11 | 为什么长这样 | RNN 两个硬伤；√d 来自点积方差 | 2:07 |
| 12 | 前向与反向 | softmax / LN 雅可比；残差是梯度高速路 | 2:25 |
| 13 | 纯 numpy 实现 | 手写反向 vs autograd：**3.0e-14** | 2:35 |
| 14 | 前向逐算子 | 9 个算子，公式 ↔ 源码 ↔ 真实数值 | 3:39 |
| 15 | 反向逐算子 | 每个雅可比的推导 ↔ 对应代码 | 3:51 |
| 16 | 串起来 | 一次迭代 → 训练 → int8 → 闭环应用 | 3:22 |
| 17 | 训练方法 | SGD 发散、Adam 收敛；偏差修正为何必要 | 3:37 |

### 附录

- **附录 A**　Transformer 控制器：前向、反向与注意力的完整解剖（视频 3:16）
- **附录 B**　资源索引：代码、视频、交互式报告

---

## 四、五个可验证的数字

这本教程的每个关键结论都能用仓库里的代码复现：

| 结论 | 数字 | 复现命令 |
|---|---|---|
| 手写反向 == autograd | 梯度差 **3.0e-14** | `cd tutorial/code && python3 parity_test.py` |
| numpy 版可直接上岗 | 闭环 **30/30**，max\|Δu\|=5.4e-6 N | `python3 load_pretrained.py` |
| 两版训练完全一致 | 逐 epoch 损失差 **1.7e-7** | `python3 train_demo.py` |
| LQR 的能力边界 | 吸引域 **51.8%**、成功率 **25%** | `cd build && python3 ch02_experiments.py` |
| 优化器的差别 | Adam 0.047 vs SGD 4.47（同模型同 3 轮） | `cd tutorial/code && python3 optim_demo.py` |

---

## 五、四类产物

| 产物 | 位置 | 说明 |
|---|---|---|
| **合订本讲义** | `book/main.tex` | 三篇 + 附录，`xelatex main.tex` 两遍出 PDF |
| **视频 18 集** | `course/out/`、`tutorial/out/` | 约 40 分钟，均配 `.srt` 字幕 |
| **交互式报告** | `index.html` | 单文件零依赖，浏览器实时仿真 |
| **代码** | `build/`、`tutorial/code/` | 可复现全部实验 |

视频规格：1280×720@30、H.264 + AAC、LaTeX 公式（KaTeX 离线渲染）、
中文旁白（edge-tts 晓晓）、程序合成配乐、烧录字幕（半透明、居中、数字用阿拉伯数字）。

---

## 六、快速开始

```bash
# 1) 看交互式报告
open index.html

# 2) 验证手写反向（3 秒出结果）
cd tutorial/code && python3 parity_test.py

# 3) 跑一遍完整评测（约 2 分钟）
cd build && python3 eval_all.py

# 4) 编讲义（需要 TeX Live + ctex）
cd book && xelatex main.tex && xelatex main.tex

# 5) 重渲任意一集视频（约 5 分钟）
cd course/video && python3 make_ch03.py
```

依赖：Python 3.9+、numpy、scipy、torch、pillow、av、edge-tts。

---

## 七、许可

代码与文档 MIT；视频与讲义可自由引用，注明出处即可。
