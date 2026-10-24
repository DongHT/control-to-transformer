# Transformer 从零实现

一份**可以逐行验证**的 Transformer 教程：从动机、推导，到前向/反向的每一个雅可比，
再到**纯 numpy 手写反向**与 **torch autograd** 两份实现，以及三张结果可视化。

> 核心承诺：手写反向与 autograd 的前向输出差 **3×10⁻¹⁵**、全部 31 个参数张量的梯度差 **3×10⁻¹⁴**。
> 这不是"跑通了"，而是"证明它是对的"。

## 内容

| 部分 | 文件 | 说明 |
|---|---|---|
| **讲义** | `latex/tutorial.tex` | 约 30 页：背景 → 结构 → 逐层前向/反向 → 两份实现 → 可视化 → 习题 |
| **视频①** | `out/t1_为什么长这样.mp4` | 2:07　RNN 的两个硬伤、检索视角、√d 的方差推导、因果掩码 |
| **视频②** | `out/t2_前向与反向.mp4` | 2:25　结构框图、六行前向公式、softmax/LN 雅可比、注意力反传 |
| **视频③** | `out/t3_两种实现.mp4` | 2:37　代码骨架、五个易错点、训练曲线重合、加载生产权重、注意力×倒立摆 |
| **numpy 实现** | `code/tinytf_numpy.py` | 前向 40 行 + 手写反向 80 行 + AdamW，无任何自动微分 |
| **torch 实现** | `code/tinytf_torch.py` | 等价结构，含权重互转（`state_from_numpy` / `numpy_from_state`）|
| **对齐检验** | `code/parity_test.py` | 与 autograd、中心差分双重验证 |
| **训练对比** | `code/train_demo.py` | 同初始化同数据，双实现同步训练 |
| **加载预训练** | `code/load_pretrained.py` | 生产权重装进 numpy 版：一致性、闭环、注意力 |
| **插图** | `code/make_figs.py` | 损失曲线、梯度量级 |

## 快速验证（三分钟）

```bash
cd code
python3 parity_test.py     # 手写反向 vs autograd：看到 3e-14 就说明对了
```

输出节选：

```
① 前向输出        最大偏差 = 3.109e-15
② 反向梯度        全部参数的最大梯度偏差 = 3.020e-14
③ 中心差分        l0.W1(3,7): 相对误差 = 8.04e-09
结论: ✅ 手写反向与 autograd 一致（<1e-6）
```

## 训练与部署

```bash
python3 train_demo.py       # 双实现各训 8 个 epoch（numpy 142s / torch 35s）
python3 load_pretrained.py  # 加载生产权重：512 窗口 max|Δu|=5.4e-6 N，闭环 30/30
python3 make_figs.py        # 生成 out/fig_*.png
```

训练对比（同初始化、同数据、同批次顺序）：

| | 训练损失 | 验证损失 | 耗时 |
|---|---|---|---|
| numpy（手写反向） | 3.60×10⁻³ | 3.694×10⁻³ | 142 s |
| torch（autograd） | 3.60×10⁻³ | 3.695×10⁻³ | 35 s |
| 差异 | 1.7×10⁻⁷ | 1.3×10⁻⁷ | 4.1× 加速 |

## 结构（18,177 参数）

```
x (B,24,6) → 嵌入 (B,24,32) + 位置编码
  → ×2 层: [LN → 4 头因果注意力 → ⊕] → [LN → FFN 32→64→32 + GELU → ⊕]
  → 取最后 token → LN → 线性 → u = 12·tanh(·)
```

每个算子的张量形状与参数量见讲义第 2 节。

## 五个易错点（讲义第 6.2 节）

1. **多头 reshape 顺序**：`(B,T,d) → (B,T,H,dh) → transpose → (B,H,T,dh)`
2. **softmax 减基线项**：`dS = a * (da - sum(da*a))`，漏了会拖慢收敛
3. **W2 / b2 的梯度**来自残差主干 `dz`，不是 `dact`（形状会从 (d,ff) 变 (ff,ff)）
4. **LayerNorm 用紧凑式**：`dx = γ/σ (g − ḡ − ẑ·mean(gẑ))`
5. **掩码位置梯度自动为零**：`a=0 ⇒ dS=0`，无需再清零

## 与其他材料的关系

本教程的模型就是主项目里部署的控制器（`build/model.py`）。
把它变成控制器只差三处：输入特征（6 维归一化）、窗口（24 步 @ 50 Hz）、输出限幅（12 N）。
量化到 int8 后 16.6 KB，浏览器内单步 0.32 ms。

## 编讲义

```bash
cd latex && xelatex tutorial.tex && xelatex tutorial.tex   # 需要 ctex 与 listings
```

## 许可

MIT（与主项目一致）。
