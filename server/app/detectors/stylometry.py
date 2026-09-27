"""不依赖模型的统计特征（仅作参考展示，默认不参与 AI 概率计算）。"""
import math
import re

_SENT = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
AI_PHRASES = [
    "首先", "其次", "再次", "此外", "另外", "值得注意的是", "值得强调的是", "综上所述", "总的来说", "总而言之",
    "由此可见", "不仅如此", "不可否认", "毋庸置疑", "需要指出的是", "与此同时", "事实上", "换言之", "众所周知",
    "在这种背景下", "从某种意义上说", "不难发现", "可以看出", "综合来看", "一方面", "另一方面", "显而易见",
    "总体而言", "在很大程度上", "具有重要意义", "发挥着重要作用", "提供了有力支撑", "深远影响",
]


def features(text: str) -> dict:
    sents = [s.strip() for s in _SENT.findall(text) if len(s.strip()) > 1]
    lens = [len(s) for s in sents] or [len(text)]
    mean = sum(lens) / len(lens)
    std = math.sqrt(sum((x - mean) ** 2 for x in lens) / len(lens))
    cv = std / mean if mean else 0.0
    bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
    uniq = len(set(bigrams)) / len(bigrams) if bigrams else 1.0
    hits = [p for p in AI_PHRASES if p in text]
    per_k = len(hits) / max(len(text), 1) * 1000
    mid = sum(1 for x in lens if 18 <= x <= 42) / len(lens)
    # 与上一版网页工具保持一致的启发式打分（0–100）
    burst = max(0.0, min(100.0, 100 - cv * 130))
    rep = max(0.0, min(100.0, (0.72 - uniq) * 260))
    phr = max(0.0, min(100.0, per_k * 22))
    rng = max(0.0, min(100.0, (mid - 0.4) * 140))
    score = burst * 0.35 + rep * 0.25 + phr * 0.25 + rng * 0.15
    return {
        "sentence_len_cv": round(cv, 3),
        "bigram_unique_ratio": round(uniq, 3),
        "template_phrases": hits,
        "mid_length_ratio": round(mid, 3),
        "heuristic_score": round(score, 1),
    }
