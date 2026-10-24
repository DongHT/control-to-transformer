"""把字幕（显示文本）里的中文数词转成阿拉伯数字；朗读文本(tss)保持中文。

规则：
  · 只处理「数词 + 量词」与「第 + 数词」
  · 白名单保护固定词：一样、一些、第一、一下、一点、几乎、十分…
  · tts 字段一律不动（配音需要中文读音）
"""
import re
import sys

MEASURE = "个条行维步层次倍组秒度米块段篇页张项种节层"
CN = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
      "六": 6, "七": 7, "八": 8, "九": 9}
PROTECT = ["一样", "一些", "一定", "一直", "一般", "一起", "唯一", "一下", "一点",
           "一步", "一部分", "一边", "一方面", "一系列", "一律", "一切", "一致",
           "一半", "十分", "几乎", "统一", "之一", "不一", "千万", "万一", "一贯",
           "一再", "一同", "一并", "一览", "一览表", "一步到位"]


def cn2int(s):
    """把 '十二' '二十五' '三' '百' 转成整数；失败返回 None"""
    if s == "百":
        return 100
    if "十" in s:
        a, _, b = s.partition("十")
        tens = 1 if a == "" else CN.get(a)
        ones = 0 if b == "" else CN.get(b)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if len(s) == 1:
        return CN.get(s)
    # 两位以上非十结构（如 二五）不处理
    return None


def convert_text(t, changes):
    out = t
    # 1) 第 + 数词（讲/步/层/章/节…）
    for m in list(re.finditer(r"第([零一二三四五六七八九十百两]+)([讲步层章节部分条行维组次])", out)):
        v = cn2int(m.group(1))
        if v is None:
            continue
        new = f"第 {v} {m.group(2)}"
        old = m.group(0)
        if old in PROTECT:
            continue
        out = out.replace(old, new, 1)
        changes.append((old, new))
    # 2) 数词 + 量词
    for m in list(re.finditer(rf"([零一二三四五六七八九十百两]+)([{MEASURE}])", out)):
        old, num, meas = m.group(0), m.group(1), m.group(2)
        if any(old == p or old.startswith(p) for p in PROTECT):
            continue
        # 前一个字是“第”的已处理
        if m.start() > 0 and out[m.start() - 1] == "第":
            continue
        v = cn2int(num)
        if v is None:
            continue
        new = f"{v} {meas}"
        out = out.replace(old, new, 1)
        changes.append((old, new))
    # 3) 百分之X / 万分之X → 数字%
    for m in list(re.finditer(r"百分之([零一二三四五六七八九十百两]+)", out)):
        v = cn2int(m.group(1))
        if v is None:
            continue
        old = m.group(0); new = f"{v}%"
        out = out.replace(old, new, 1); changes.append((old, new))
    for m in list(re.finditer(r"万分之([零一二三四五六七八九十百两]+)", out)):
        v = cn2int(m.group(1))
        if v is None:
            continue
        old = m.group(0); new = f"1e-4×{v}"
        out = out.replace(old, new, 1); changes.append((old, new))
    return out


def process_module(path, dry=True):
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:
        print(f"[skip] {path}: {e}")
        return []
    changes = []
    for sh in getattr(m, "SHOTS", []):
        for s in sh["say"]:
            if isinstance(s, dict):
                changes.append(("", ""))  # placeholder to keep type
                new = convert_text(s["say"], changes)
                s["say"] = new
            else:
                pass
    return changes


if __name__ == "__main__":
    print("本脚本只用于审阅转换效果；实际改动由 apply_cn.py 完成")
