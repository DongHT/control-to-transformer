"""全量审查：公式方框、字形缺字、字幕数字、多音字、断句长度。

用法：python3 build/audit.py [--render]
  --render  同时把公式重渲染一遍，扫描 KaTeX 报错色（慢，几分钟）
"""
import importlib
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/Users/liwenhao/my_doc/transformer"
VID = os.path.join(ROOT, "course", "video")
sys.path.insert(0, VID)
sys.path.insert(0, os.path.join(VID, "scenes"))

MODULES = ["ch01", "ch02", "ch03", "ch04", "ch05", "ch06", "ch07", "ch08", "ch09",
           "ch10", "chA", "t1", "t2", "t3", "t4", "t5", "t6", "t7"]

SONGTI = "/System/Library/Fonts/Songti.ttc"
MENLO = "/System/Library/Fonts/Menlo.ttc"
HEITI = "/System/Library/Fonts/STHeiti Medium.ttc"

# 多音字黑名单：(词, 说明)。只列在技术语境里容易读错的。
POLY = [
    ("行", "行列/行列式 → háng；行走/执行 → xíng"),
    ("重", "重力 zhòng；重新 chóng"),
    ("差", "误差 chā；差别 chā；差劲 chà"),
    ("率", "效率 lǜ；概率 lǜ"),
    ("为", "为 wéi；为了 wèi"),
    ("长", "长度 cháng；增长 zhǎng"),
    ("数", "数量 shù；数数 shǔ"),
    ("调", "调节 tiáo；调度 diào"),
    ("圈", "整圈 quān"),
    ("卷", "卷积 juǎn"),
    ("略", "策略 lüè"),
    ("累", "累积 lěi；劳累 lèi"),
    ("冲", "冲激 chōng"),
    ("倒", "倒立 dào；倒塌 dǎo"),
    ("称", "对称 chèn；称呼 chēng"),
    ("朝", "朝向 cháo"),
    ("落", "落地 luò"),
    ("应", "应用 yìng；应该 yīng"),
    ("分", "分量 fèn；分别 fēn"),
    ("便", "方便 biàn"),
]

CN_NUM = "零一二三四五六七八九十百千万亿"


def font_sets():
    fs = [ImageFont.truetype(SONGTI, 40, index=i) for i in (1, 3, 6)]
    return fs, ImageFont.truetype(MENLO, 40, index=0), ImageFont.truetype(HEITI, 40, index=0)


def bmp(f, c):
    im = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(im).text((10, 10), c, font=f, fill=255)
    return np.asarray(im)


def main():
    fs, menlo, heiti = font_sets()
    nd = [bmp(f, "\uFFFF") for f in fs]
    ndm = bmp(menlo, "\uFFFF")
    ndh = bmp(heiti, "\uFFFF")

    def has(f, b):
        return b.max() > 0 and not np.array_equal(b, nd_ if False else b)  # placeholder

    def ok_char(c):
        if c in " \u3000\x00":
            return True
        if ord(c) < 128:
            b = bmp(menlo, c)
            return b.max() > 0 and not np.array_equal(b, ndm)
        for k, f in enumerate(fs):
            b = bmp(f, c)
            if b.max() > 0 and not np.array_equal(b, nd[k]):
                return True
        return False

    def ok_sub(c):
        if c in " \u3000":
            return True
        b = bmp(heiti, c)
        return b.max() > 0 and not np.array_equal(b, ndh)

    from render import Canvas, INK
    import mathrender as mr

    frame_strs, texs, subs, ttss = set(), set(), [], []
    orig_text, orig_math = Canvas.text, None
    import kit
    orig_math = kit.math_

    cur = {"mod": None}

    def patched_text(self, x, y, s, size=32, color=INK, anchor="la", weight="regular",
                     alpha=1.0, check=True, tag=""):
        frame_strs.add(s)
        return orig_text(self, x, y, s, size, color, anchor, weight, alpha, check, tag)

    def patched_math(cv, x, y, tex, size=26, color=kit.INK, anchor="ls", check=True, tag=None):
        texs.add(tex)
        return orig_math(cv, x, y, tex, size, color, anchor, check, tag)

    Canvas.text = patched_text
    kit.math_ = patched_math

    for m in MODULES:
        mod = importlib.import_module(m)
        if hasattr(mod, "math_"):
            mod.math_ = patched_math
        for sh in mod.SHOTS:
            for s in sh["say"]:
                disp = s["say"] if isinstance(s, dict) else s
                tts = s.get("tts") if isinstance(s, dict) else None
                subs.append((m, sh["name"], disp))
                if tts:
                    ttss.append((m, sh["name"], tts))
                else:
                    # 没有 tts 时，朗读文本 = 去 LaTeX 的显示文本
                    ttss.append((m, sh["name"], re.sub(r"\$[^$]*\$", "", disp)))
            for tt in (0.8, 0.5 * sh.get("min_dur", 10), sh.get("min_dur", 10) - 0.5):
                try:
                    sh["draw"](Canvas(), tt, sh.get("min_dur", 10))
                except Exception as e:
                    print(f"[draw-error] {m}/{sh['name']}: {e}")

    print("=" * 78)
    print(f"① 画面文字缺字（Songti/Menlo）")
    bad = sorted({c for s in frame_strs for c in s if not ok_char(c)})
    print("   ", bad or "none")

    print(f"② 字幕缺字（STHeiti）")
    bads = sorted({c for _, _, s in subs for c in re.sub(r'\$[^$]*\$', '', s) if not ok_sub(c)})
    print("   ", bads or "none")

    print(f"③ 字幕里的中文数字（应改阿拉伯数字）")
    hits = []
    for m, sh, s in subs:
        plain = re.sub(r"\$[^$]*\$", "", s)
        for mt in re.finditer(f"[{CN_NUM}]+", plain):
            seg = mt.group(0)
            # 放过常见非数量词
            if any(k in plain[max(0, mt.start() - 1):mt.end() + 1] for k in
                   ("一样", "一些", "一定", "一直", "一般", "一个", "一步", "一次", "一条",
                    "第一", "第二", "第三", "第四", "第五", "第六", "第七", "第八", "第九",
                    "十分", "几乎", "统一", "唯一", "之一", "一半", "一起", "一块", "一律",
                    "一切", "一致", "一方面", "一系列", "一点点")):
                continue
            hits.append((m, sh, seg, s[:56]))
    if hits:
        for h in hits:
            print(f"    {h[0]:6s} {h[1]:10s} 「{h[2]}」 {h[3]}")
    else:
        print("    none")

    print(f"④ 朗读文本里的多音字候选")
    poly_hits = []
    for m, sh, t in ttss:
        for w, note in POLY:
            if w in t:
                poly_hits.append((m, sh, w, note, t[:44]))
    seen = set()
    for m, sh, w, note, ctx in poly_hits:
        key = (m, sh, w)
        if key in seen:
            continue
        seen.add(key)
        print(f"    {m:6s} {sh:10s} 「{w}」 {note}   …{ctx}…")

    print(f"⑤ 单句过长（朗读断句风险，>42 字）")
    long_ = [(m, sh, len(t), t[:50]) for m, sh, t in ttss if len(t) > 42]
    for l in long_:
        print(f"    {l[0]:6s} {l[1]:10s} {l[2]} 字  {l[3]}…")
    if not long_:
        print("    none")

    if "--render" in sys.argv:
        print(f"⑥ KaTeX 公式报错扫描（{len(texs)} 个）")
        bad = []
        for tex in sorted(texs):
            try:
                img = mr.render(tex, 22, "#E8F0F8")
            except Exception as e:
                bad.append((tex, "exception " + str(e)[:60]))
                continue
            a = np.asarray(img.convert("RGB")).astype(int)
            r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
            red = ((r > 150) & (g < 70) & (b < 70)).sum()
            if red > 20:
                bad.append((tex, f"red px={red}"))
        for tex, why in bad:
            print(f"    [{why}] {tex[:100]}")
        print("    总计问题公式:", len(bad))


if __name__ == "__main__":
    main()
