"""生成教程插图：损失曲线、梯度范数、速度对比。"""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
F = "/System/Library/Fonts/Songti.ttc"
CY, AM, MG, INK, DIM = (24,160,148), (200,130,20), (190,60,110), (28,34,44), (90,104,120)
def f(sz, i=6): return ImageFont.truetype(F, sz, index=i)

D = json.load(open(os.path.join(OUT, "train_demo.json")))
W, H = 1100, 640
img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
L, R, T = 110, 60, 70
# ---------- 上：损失曲线
d.text((L, 22), "训练损失：纯 numpy 手写反向 vs torch autograd（同初始化、同数据、同批次顺序）",
       font=f(21, 1), fill=INK)
top_b, top_h = 400, 300
y0, y1 = 1e-3, 1.0
sy = lambda v: top_b - (np.log10(max(v,y0))-np.log10(y0))/(np.log10(y1)-np.log10(y0))*top_h
sx = lambda ep: L + (ep-1)/max(1,(D["epochs"]-1))*(W-L-R)
for v in (1e-3, 1e-2, 1e-1, 1.0):
    y = sy(v); d.line([L, y, W-R, y], fill=(230,234,240), width=1)
    d.text((L-10, y), f"{v:g}", font=f(15), fill=DIM, anchor="rm")
for ep in range(1, D["epochs"]+1):
    d.text((sx(ep), top_b+8), str(ep), font=f(14), fill=DIM, anchor="ma")
np_tr = [h["train"] for h in D["numpy"]]; to_tr = [h["train"] for h in D["torch"]]
np_va = [h["val"] for h in D["numpy"]];   to_va = [h["val"] for h in D["torch"]]
# numpy：粗青线；torch：细虚线叠在上面 → 看得见"重合"
for i in range(len(np_tr)-1):
    d.line([(sx(i+1), sy(np_tr[i])), (sx(i+2), sy(np_tr[i+1]))], fill=CY, width=5)
for i in range(len(to_tr)-1):
    x1,y1p = sx(i+1), sy(to_tr[i]); x2,y2p = sx(i+2), sy(to_tr[i+1])
    for k in range(3):
        a, b = k/3, (k+0.55)/3
        d.line([(x1+(x2-x1)*a, y1p+(y2p-y1p)*a), (x1+(x2-x1)*b, y1p+(y2p-y1p)*b)],
               fill=(255,255,255), width=2)
d.line([(sx(i+1), sy(np_va[i])) for i in range(len(np_va))], fill=AM, width=2)
for i in range(len(np_va)):
    x,y = sx(i+1), sy(np_va[i]); d.ellipse([x-3.5,y-3.5,x+3.5,y+3.5], outline=AM, width=2)
d.text((L, top_b+34), "横轴 epoch；粗青= numpy 训练损失，其上的白色虚线= torch（与青线完全重合）；橙圈= 验证损失",
       font=f(15), fill=DIM)

# ---------- 下：|Δloss|
bot_t, bot_b = 470, 590
d.text((L, bot_t-34), "两版逐 epoch 的差：|numpy − torch|（对数纵轴，证明二者数学上完全等价）",
       font=f(17,1), fill=INK)
dif = [abs(a["train"]-b["train"]) for a,b in zip(D["numpy"], D["torch"])]
difv = [abs(a["val"]-b["val"]) for a,b in zip(D["numpy"], D["torch"])]
lo, hi = 1e-9, 1e-3
sy2 = lambda v: bot_b - (np.log10(max(v,lo))-np.log10(lo))/(np.log10(hi)-np.log10(lo))*(bot_b-bot_t)
for v in (1e-9, 1e-7, 1e-5, 1e-3):
    y = sy2(v); d.line([L, y, W-R, y], fill=(230,234,240), width=1)
    d.text((L-10, y), f"{v:g}", font=f(14), fill=DIM, anchor="rm")
d.line([(sx(i+1), sy2(dif[i])) for i in range(len(dif))], fill=CY, width=3)
for i in range(len(dif)):
    x,y = sx(i+1), sy2(dif[i]); d.ellipse([x-3.2,y-3.2,x+3.2,y+3.2], fill=CY)
d.line([(sx(i+1), sy2(max(difv[i],1e-12))) for i in range(len(difv))], fill=AM, width=2)
for i in range(len(difv)):
    x,y = sx(i+1), sy2(max(difv[i],1e-12)); d.ellipse([x-3, y-3, x+3, y+3], outline=AM, width=1)
d.text((sx(len(dif)), sy2(dif[-1])-16), f"训练 {dif[-1]:.1e}", font=f(14), fill=CY, anchor="ra")
d.text((sx(len(difv)), sy2(max(difv[-1],1e-12))+16), f"验证 {difv[-1]:.1e}", font=f(14), fill=AM, anchor="ra")
d.text((L, H-24), f"耗时 numpy {D['t_numpy']:.0f}s / torch {D['t_torch']:.0f}s = {D['speedup']:.1f}×；"
                  f"训练损失最大差 {max(dif):.1e}（float32 舍入量级）", font=f(16), fill=DIM)
img.save(os.path.join(OUT, "fig_loss.png"))

# ---- 梯度范数条形图
PR = json.load(open(os.path.join(OUT, "pretrained_report.json")))
bp = json.load(open("/Users/liwenhao/my_doc/transformer/build/out/chA.json"))["backward"]["grads"]
items = sorted(bp.items(), key=lambda kv: -kv[1])[:8]
W2, H2 = 900, 420
img = Image.new("RGB", (W2, H2), "white"); d = ImageDraw.Draw(img)
d.text((40, 24), "反向传播：各张量的梯度范数（单样本，真实权重）", font=f(20,1), fill=INK)
mx = max(v for _, v in items)
for i, (k, v) in enumerate(items):
    y = 80 + i*38
    d.text((40, y+10), k.replace("blocks.","L").replace(".weight",""), font=f(15), fill=INK)
    d.rectangle([330, y, 330 + v/mx*440, y+22], fill=CY)
    d.text((330 + v/mx*440 + 10, y+10), f"{v:.1f}", font=f(15), fill=DIM)
d.text((40, H2-30), "输出头最大，越往里衰减越慢——残差提供梯度直通路", font=f(15), fill=DIM)
img.save(os.path.join(OUT, "fig_grad.png"))
print("figs ok")
