"""不依赖模型的统计特征（仅作参考展示，默认不参与 AI 概率计算）。"""
import math
import re

_SENT = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_EN_SENT = re.compile(r"[^.!?\n]+[.!?]*[\"”’)]?")
_LATIN = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[㐀-䶿一-鿿]")
_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
# 英文 AI 文本高频词语（参考 GPTZero、Originality.ai 等公开的"AI 腔"词表与相关研究的统计）
AI_PHRASES_EN = [
    "delve", "tapestry", "testament to", "it is worth noting", "it's worth noting", "in conclusion", "furthermore",
    "moreover", "additionally", "in summary", "overall,", "notably", "crucial", "pivotal", "multifaceted",
    "intricate", "underscore", "underscores", "showcase", "showcasing", "navigate", "navigating", "realm",
    "landscape", "foster", "fostering", "embark", "vibrant", "seamless", "robust", "leverage", "holistic",
    "nuanced", "profound", "enduring", "serves as a", "plays a crucial role", "plays a vital role",
    "a rich tapestry", "in today's", "ever-evolving", "it is important to note", "not only", "shed light",
    "stands as", "a myriad of", "paramount", "resonate", "resonates", "meticulous", "meticulously",
]
AI_PHRASES = [
    "首先", "其次", "再次", "此外", "另外", "值得注意的是", "值得强调的是", "综上所述", "总的来说", "总而言之",
    "由此可见", "不仅如此", "不可否认", "毋庸置疑", "需要指出的是", "与此同时", "事实上", "换言之", "众所周知",
    "在这种背景下", "从某种意义上说", "不难发现", "可以看出", "综合来看", "一方面", "另一方面", "显而易见",
    "总体而言", "在很大程度上", "具有重要意义", "发挥着重要作用", "提供了有力支撑", "深远影响",
]


def features(text: str) -> dict:
    english = len(_LATIN.findall(text)) >= max(30, 2 * len(_CJK.findall(text)))
    if english:
        sents = [x.strip() for x in _EN_SENT.findall(text) if len(_WORD.findall(x)) >= 2]
        lens = [len(_WORD.findall(x)) for x in sents] or [len(_WORD.findall(text))]
        low = text.lower()
        hits = [p for p in AI_PHRASES_EN if re.search(r"(?<![a-z])" + re.escape(p), low)]
    else:
        sents = [x.strip() for x in _SENT.findall(text) if len(x.strip()) > 1]
        lens = [len(x) for x in sents] or [len(text)]
        hits = [p for p in AI_PHRASES if p in text]
    mean = sum(lens) / len(lens)
    std = math.sqrt(sum((x - mean) ** 2 for x in lens) / len(lens))
    cv = std / mean if mean else 0.0
    bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
    uniq = len(set(bigrams)) / len(bigrams) if bigrams else 1.0
    per_k = len(hits) / max(len(text), 1) * 1000
    mid = sum(1 for x in lens if (12 <= x <= 28 if english else 18 <= x <= 42)) / len(lens)
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
