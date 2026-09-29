"""检测引擎：加载模型、逐段打分、汇总成报告；长文档放进后台任务队列。

每段先识别文体（现代汉语 / 文言 / 英文），再用该文体对应的分类器和校准参数打分——
与知网、维普、Turnitin 等平台"先分语种、再用专门模型判断"的做法一致。"""
from __future__ import annotations

import logging
import math
import queue
import threading
import time
import uuid

from . import config, scoring
from .detectors import stylometry
from .detectors.classifier import Classifier, make_english_classifier
from .detectors.lm_scorer import LMScorer
from .segmenter import REGISTER_NAMES, detect_register, min_chars, segment_text

log = logging.getLogger("engine")


def style_scores(text: str, feats: dict | None = None) -> dict:
    """供校准使用的统计特征：句长变异系数、每千字套话数。"""
    f = feats or stylometry.features(text)
    return {"style_cv": f["sentence_len_cv"],
            "style_phrases": len(f["template_phrases"]) / max(len(text), 1) * 1000}


LM_KEYS = ("fastdetect", "fastdetect_norm", "binoculars", "ppl", "x_ppl", "log_rank", "lrr", "entropy",
           "top1", "top10", "lp_burstiness")


def memorized(raw: dict) -> bool:
    """语言模型几乎能逐字复现（困惑度极低），而分类器明确判为人写：多半是模型训练时背熟的公开名篇
    （莎士比亚、唐诗宋词、经典演讲等）。这时语言模型信号会误把名篇当成 AI，应当不予采信。"""
    ppl, cls = raw.get("ppl"), raw.get("classifier")
    return ppl is not None and cls is not None and math.exp(ppl) < config.MEMORIZED_PPL and cls < 0.3


def is_short(seg, text: str) -> bool:
    if seg.register == "zh_poetry":
        return False
    if seg.register == "en":
        return len(text.split()) < config.SHORT_WORDS_EN
    return len(text) < config.SHORT_CHARS_ZH


def score_text(seg) -> str:
    """送进模型打分的文字：去掉开头的标题行（标题不是作者的正文，短诗里标题占比又很大）。"""
    if seg.title and seg.text.startswith(seg.title):
        rest = seg.text[len(seg.title):].strip()
        if len(rest) >= 10:
            return rest
    return seg.text


def works_summary(seg_out: list) -> list:
    """按作品（标题分开的部分）汇总，类似知网报告里的“章节 / 片段 AI 率”：每篇给出字数、AI 率和结论。"""
    works: dict = {}
    for seg in seg_out:
        w = works.setdefault(seg["block"], {"block": seg["block"], "title": "", "chars": 0, "counted": 0,
                                            "flagged": 0, "psum": 0.0, "refsum": 0.0, "refchars": 0,
                                            "registers": [], "segments": 0})
        if seg["title"] and not w["title"]:
            w["title"] = seg["title"]
        w["chars"] += seg["chars"]
        w["segments"] += 1
        if seg["register"] not in w["registers"]:
            w["registers"].append(seg["register"])
        if seg["prob"] is not None:
            w["counted"] += seg["chars"]
            w["psum"] += seg["prob"] * seg["chars"]
            if seg["level"] in ("high", "mid", "light"):
                w["flagged"] += seg["chars"]
        elif seg["kind"] == "reference_only" and seg["ref_prob"] is not None:
            w["refchars"] += seg["chars"]
            w["refsum"] += seg["ref_prob"] * seg["chars"]
    out = []
    for w in works.values():
        if not w["title"]:
            first = next((x["text"] for x in seg_out if x["block"] == w["block"]), "")
            w["title"] = first.strip().splitlines()[0][:20] + "……" if first.strip() else "（无标题）"
        if w["counted"]:
            rate = w["flagged"] / w["counted"]
            verdict = "疑似 AI 生成" if rate >= 0.5 else "部分段落疑似 AI" if rate > 0 else "未见明显 AI 特征"
            out.append({**{k: w[k] for k in ("block", "title", "chars", "registers", "segments")},
                        "ai_rate": round(rate, 4), "mean_prob": round(w["psum"] / w["counted"], 4),
                        "counted": True, "verdict": verdict})
        elif w["refchars"]:
            out.append({**{k: w[k] for k in ("block", "title", "chars", "registers", "segments")},
                        "ai_rate": None, "mean_prob": round(w["refsum"] / w["refchars"], 4),
                        "counted": False, "verdict": "仅供参考（诗词不计入）"})
        else:
            out.append({**{k: w[k] for k in ("block", "title", "chars", "registers", "segments")},
                        "ai_rate": None, "mean_prob": None, "counted": False, "verdict": "未计入（引文 / 参考文献）"})
    return out


class Engine:
    def __init__(self):
        self.lm = LMScorer() if config.ENABLE_LM else None
        self.cls = Classifier() if (config.ENABLE_CLASSIFIER and config.CLASSIFIER_MODEL) else None
        self.cls_en = (make_english_classifier()
                       if (config.ENABLE_EN_CLASSIFIER and config.EN_CLASSIFIER_MODEL) else None)
        self.cls_poetry = (Classifier(config.POETRY_CLASSIFIER_MODEL, 128) if config.POETRY_CLASSIFIER_MODEL else None)
        self.cls_classical = (Classifier(config.CLASSICAL_CLASSIFIER_MODEL, 256) if config.CLASSICAL_CLASSIFIER_MODEL else None)
        self.loading = True
        self.loaded_event = threading.Event()
        self.load_started = time.time()
        self.cal, self.cal_source = self._load_cal()
        self.tokens_per_sec = None

    # ---------- 加载 ----------
    def _load_cal(self):
        cal, source = self._load_base_cal()
        user = config.load_user_profiles()
        for prof, c in user.items():
            cal = scoring.merge_profile(cal, c, prof)
        if user:
            source += "；已叠加你的标注校准（" + "、".join(scoring.PROFILE_NAMES.get(p, p) for p in user) + "）"
        return cal, source

    def reload_calibration(self):
        self.cal, self.cal_source = self._load_cal()

    def _load_base_cal(self):
        default = config.load_default_calibration()
        override, source = config.load_calibration_override()
        if override:
            cal = dict(scoring.DEFAULTS)
            cal.update(override)
            # 管理员只校准了部分文体时，其余文体沿用内置默认校准
            profs = dict((default or {}).get("profiles") or {})
            profs.update(override.get("profiles") or {})
            if profs:
                cal["profiles"] = profs
            return cal, source
        if default:
            cal = dict(scoring.DEFAULTS)
            cal.update(default)
            names = "、".join(["现代汉语"] * bool(default.get("calibrated")) +
                             [scoring.PROFILE_NAMES.get(k, k) for k in (default.get("profiles") or {})])
            return cal, f"内置默认校准（公开数据集；{names}）"
        return dict(scoring.DEFAULTS), "内置经验值（未校准）"

    def set_calibration(self, cal: dict, source: str):
        self.cal, self.cal_source = cal, source

    def load_all(self):
        try:
            for name, det in (("语言模型", self.lm), ("中文分类器", self.cls), ("英文分类器", self.cls_en),
                              ("诗词分类器", self.cls_poetry), ("文言分类器", self.cls_classical)):
                if det is None:
                    continue
                try:
                    det.load()
                except Exception as e:  # noqa: BLE001 —— 单个检测器失败不影响其他
                    log.exception("failed to load %s", name)
                    det.error = f"{type(e).__name__}: {e}"
        finally:
            self.loading = False
            self.loaded_event.set()

    def wait_loaded(self, timeout: float = 900) -> bool:
        """等所有模型加载完再打分：避免"语言模型好了、分类器还没好"时算出缺项的结果。"""
        return self.loaded_event.wait(timeout)

    def status(self) -> dict:
        def st(det, model):
            if det is None:
                return {"enabled": False}
            return {"enabled": True, "ready": det.ready, "error": det.error, "model": model}
        profiles = {"zh": bool(self.cal.get("calibrated"))}
        for k in ("zh_classical", "zh_poetry", "en"):
            profiles[k] = bool(((self.cal.get("profiles") or {}).get(k) or {}).get("calibrated"))
        return {
            "loading": self.loading,
            "lm": st(self.lm, [config.OBSERVER_MODEL, config.PERFORMER_MODEL]),
            "classifier": st(self.cls, config.CLASSIFIER_MODEL),
            "classifier_en": st(self.cls_en, config.EN_CLASSIFIER_MODEL),
            "classifier_poetry": st(self.cls_poetry, config.POETRY_CLASSIFIER_ID),
            "classifier_classical": st(self.cls_classical, config.CLASSICAL_CLASSIFIER_ID),
            "calibration": {"calibrated": bool(self.cal.get("calibrated")), "source": self.cal_source,
                            "threshold": self.cal.get("threshold"), "note": self.cal.get("note"),
                            "profiles": profiles},
            "tokens_per_sec": self.tokens_per_sec,
        }

    def any_ready(self) -> bool:
        return bool((self.lm and self.lm.ready) or (self.cls and self.cls.ready) or (self.cls_en and self.cls_en.ready))

    def special_classifier(self, register: str):
        """诗词、文言各有专用分类器（配置了才有）。"""
        return {"zh_poetry": self.cls_poetry, "zh_classical": self.cls_classical}.get(register)

    def classifier_for(self, register: str):
        special = self.special_classifier(register)
        if special is not None:
            # 配置了专用分类器却没加载成功时，不退回通用分类器（校准参数是按专用分类器拟合的）
            return special if special.ready else None
        det = self.cls_en if register == "en" else self.cls
        return det if (det and det.ready) else None

    # ---------- 打分 ----------
    def classify(self, texts: list[str], registers: list[str]) -> list:
        """按文体分组送进对应的分类器；没有可用分类器的返回 None。"""
        out = [None] * len(texts)
        for reg in set(registers):
            det = self.classifier_for(reg)
            if not det:
                continue
            idx = [i for i, r in enumerate(registers) if r == reg]
            for i, p in zip(idx, det.predict([texts[i] for i in idx])):
                out[i] = p
        return out

    def classify_second(self, texts: list[str], registers: list[str]) -> list:
        """诗词、文言段落再用通用中文分类器（MPU）打一次分，作为第二意见（与专用分类器互相制衡）。"""
        out = [None] * len(texts)
        if not (self.cls and self.cls.ready):
            return out
        idx = [i for i, r in enumerate(registers) if self.special_classifier(r) is not None]
        if idx:
            for i, p in zip(idx, self.cls.predict([texts[i] for i in idx])):
                out[i] = p
        return out

    def raw_scores(self, texts: list[str], progress=None, registers: list[str] | None = None) -> list[dict]:
        """对若干段文字算原始分数（校准、评估时也用这个）。"""
        registers = registers or [detect_register(t) for t in texts]
        out = [style_scores(t) for t in texts]
        for i, p in enumerate(self.classify(texts, registers)):
            if p is not None:
                out[i]["classifier"] = p
        for i, p in enumerate(self.classify_second(texts, registers)):
            if p is not None:
                out[i]["classifier_mpu"] = p
        if self.lm and self.lm.ready:
            for i, t in enumerate(texts):
                t0 = time.time()
                r = self.lm.score(t)
                if r:
                    out[i].update(r)
                    dt = time.time() - t0
                    if dt > 0:
                        tps = r["tokens"] * 2 / dt  # 两个模型
                        self.tokens_per_sec = tps if self.tokens_per_sec is None else 0.8 * self.tokens_per_sec + 0.2 * tps
                if progress:
                    progress(i + 1, len(texts))
        elif progress:
            progress(len(texts), len(texts))
        return out

    def analyze(self, text: str, mode: str = "full", exclude_references: bool = True,
                flag_quotations: bool = True, progress=None) -> dict:
        t_start = time.time()
        self.wait_loaded()
        segs = segment_text(text, exclude_references, flag_quotations)
        # 全文都被判为引文/参考文献时，退回为全部计入，否则报告里没有任何数值
        fallback_all = bool(segs) and not any(s.counted for s in segs)
        if fallback_all:
            for s in segs:
                if s.kind != "body":
                    s.notes = list(s.notes) + ["全文均被判为引文/参考文献，已改为计入"]
                    s.kind = "body"
        # 某种文体的检测在评估中不够稳定时（目前可能是诗词），该文体只给参考值、不计入 AI 率
        ref_only_regs = {r for r in ("zh_poetry",)
                         if (self.cal.get("profiles") or {}).get(r, {}).get("reference_only")}
        for s in segs:
            if s.kind == "body" and s.register in ref_only_regs:
                s.kind = "reference_only"
                s.notes = list(s.notes) + [f"{REGISTER_NAMES.get(s.register, s.register)}检测不够稳定，结果仅供参考、不计入 AI 率"]
        counted = [s for s in segs if s.counted]
        # 不计入的引文段落也打分，作为"参考值"显示（不影响 AI 率）；参考文献条目没有检测意义，不打分
        ref_scored = [s for s in segs if s.kind in ("quotation", "reference_only")]
        lm_targets = counted
        sampled = False
        if mode == "fast" and len(counted) > config.FAST_MODE_MAX_SEGMENTS:
            step = len(counted) / config.FAST_MODE_MAX_SEGMENTS
            lm_targets = [counted[int(i * step)] for i in range(config.FAST_MODE_MAX_SEGMENTS)]
            sampled = True
        else:
            lm_targets = counted + ref_scored
        lm_ids = {s.index for s in lm_targets}

        results = {s.index: {} for s in segs}
        scored = counted + ref_scored
        # 分类器很快：对所有正文和引文段落都算（按文体选分类器）
        sc_texts, sc_regs = [score_text(s) for s in scored], [s.register for s in scored]
        for s, p in zip(scored, self.classify(sc_texts, sc_regs)):
            if p is not None:
                results[s.index]["classifier"] = p
        for s, p in zip(scored, self.classify_second(sc_texts, sc_regs)):
            if p is not None:
                results[s.index]["classifier_mpu"] = p
        # 语言模型较慢：快速模式下抽样
        if self.lm and self.lm.ready:
            done = 0
            for s in segs:
                if s.index not in lm_ids:
                    continue
                t0 = time.time()
                r = self.lm.score(score_text(s))
                if r:
                    results[s.index].update(r)
                    dt = time.time() - t0
                    if dt > 0:
                        tps = r["tokens"] * 2 / dt
                        self.tokens_per_sec = tps if self.tokens_per_sec is None else 0.8 * self.tokens_per_sec + 0.2 * tps
                done += 1
                if progress:
                    progress(done, len(lm_ids))

        cal = self.cal
        styles = {s.index: stylometry.features(score_text(s)) for s in scored}
        for s in scored:
            results[s.index].update(style_scores(score_text(s), styles[s.index]))

        # 1) 每段按自己的文体用对应的校准参数打分（引文段落的分数只作参考值）
        scored_ids = {s.index for s in scored}
        prof = {}
        has_short = bool((cal.get("profiles") or {}).get("zh_short"))
        for s in segs:
            reg = s.register
            if reg == "zh" and has_short and len(score_text(s)) < config.SHORT_SEGMENT_CHARS:
                reg = "zh_short"
            prof[s.index] = scoring.profile_for(cal, reg)
        memo = {s.index for s in scored if memorized(results[s.index])}

        def for_combine(i):
            if i in memo:
                return {k: v for k, v in results[i].items() if k not in LM_KEYS}
            return results[i]
        combos = {s.index: (scoring.combine(for_combine(s.index), prof[s.index][0]) if s.index in scored_ids
                            else {"prob": None, "signals": {}})
                  for s in segs}
        # 2) 与相邻正文段落平滑：只在同一篇作品、同一文体的正文段落之间进行（标题行分开的作品互不影响）
        smoothed = {}
        segs_by_idx = {s.index: s for s in segs}
        groups: dict = {}
        for s in counted:
            groups.setdefault((s.block, s.register), []).append(s.index)
        for idxs in groups.values():
            smoothed.update(zip(idxs, scoring.smooth([combos[i]["prob"] for i in idxs], config.SMOOTHING)))

        # 3) 整篇一致性（参考 Turnitin / 知网按“整篇作品”给结论的做法）：AI 文章通常整篇一次生成。
        #    同一篇作品（同一标题下、同一文体）里，已判为疑似 AI 的文字占多数时，
        #    本篇中“接近阈值”的段落也按轻度疑似计入，并注明原因。只会把接近阈值的段落往上拉，
        #    不会影响整体判为人写的作品。
        work_ai = set()
        for (blk, reg), idxs in groups.items():
            if len(idxs) < 3:
                continue
            thr_g = {i: float(prof[i][0].get("threshold", 0.5)) for i in idxs}
            tot = sum(len(segs_by_idx[i].text) for i in idxs if smoothed.get(i) is not None)
            hit = sum(len(segs_by_idx[i].text) for i in idxs
                      if smoothed.get(i) is not None and smoothed[i] >= thr_g[i])
            if tot and hit >= config.WORK_MAJORITY * tot:
                work_ai.update(i for i in idxs if smoothed.get(i) is not None
                               and thr_g[i] - scoring.NEAR_MARGIN <= smoothed[i] < thr_g[i])

        seg_out, counted_chars, prob_weighted = [], 0, 0.0
        chars_by_level = {"high": 0, "mid": 0, "light": 0, "low": 0}
        level_counts = {"high": 0, "mid": 0, "light": 0, "low": 0}
        chars_by_register: dict[str, int] = {}
        uncalibrated_regs = set()
        near_chars = 0
        for s in segs:
            sc = results[s.index]
            comb = combos[s.index]
            pcal, has_cal = prof[s.index]
            thr = float(pcal.get("threshold", 0.5))
            prob = smoothed.get(s.index) if s.counted else None
            level, label = scoring.level_of(prob, thr) if s.counted else ("none", "")
            near = bool(s.counted and prob is not None and thr - scoring.NEAR_MARGIN <= prob < thr)
            by_work = s.index in work_ai
            if by_work:
                level, label, near = "light", "轻度疑似（整篇判断）", False
            if s.counted and prob is not None:
                counted_chars += len(s.text)
                prob_weighted += prob * len(s.text)
                chars_by_level[level] += len(s.text)
                level_counts[level] += 1
                chars_by_register[s.register] = chars_by_register.get(s.register, 0) + len(s.text)
                if near:
                    near_chars += len(s.text)
                if not has_cal:
                    uncalibrated_regs.add(s.register)
            seg_out.append({
                "index": s.index, "start": s.start, "chars": len(s.text), "text": s.text,
                "kind": s.kind, "notes": s.notes, "register": s.register, "block": s.block, "title": s.title,
                "register_name": REGISTER_NAMES.get(s.register, s.register),
                "threshold": thr,
                "prob": None if prob is None else round(prob, 4),
                "ref_prob": (None if s.counted or comb["prob"] is None else round(comb["prob"], 4)),
                "prob_unsmoothed": None if comb["prob"] is None else round(comb["prob"], 4),
                "level": level, "label": label, "near_threshold": near, "by_work": by_work,
                "memorized": s.index in memo,
                "short": is_short(s, score_text(s)),
                "signals": {k: (None if v is None else round(v, 4)) for k, v in comb["signals"].items()},
                "raw": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sc.items()},
                "style": styles.get(s.index),
                "scored_by_lm": s.index in lm_ids and "fastdetect" in sc,
            })

        excluded = [s for s in segs if not s.counted]
        flagged_chars = chars_by_level["high"] + chars_by_level["mid"] + chars_by_level["light"]
        rate = (lambda n: round(n / counted_chars, 4) if counted_chars else None)
        main_reg = max(chars_by_register, key=chars_by_register.get) if chars_by_register else "zh"
        main_cal = scoring.profile_for(cal, main_reg)[0]

        # 可信度提示（参考 Turnitin、GPTZero 等平台对短文本、未校准、非常规文体的处理）
        notes = []
        en_chars = chars_by_register.get("en", 0)
        if counted_chars < 300 or (main_reg == "en" and en_chars < 1500):
            notes.append("参与计算的正文较短（中文不足 300 字 / 英文不足约 300 词），结果波动较大，仅供参考。")
        if len(chars_by_register) > 1:
            notes.append("文中含多种文体（" + "、".join(f"{REGISTER_NAMES.get(k, k)} {v} 字"
                                                     for k, v in chars_by_register.items()) + "），各自用对应的模型和阈值判断。")
        for reg in sorted(uncalibrated_regs):
            notes.append(f"{REGISTER_NAMES.get(reg, reg)}部分尚无专门校准，结果只宜作相对参考。")
        if chars_by_register.get("zh_poetry"):
            notes.append("诗词对联篇幅短、格律限制多，是公认最难检测的文体（ACL 2026 ChangAn 基准中多数检测器接近随机），"
                         "诗词部分的结果请只作参考。")
        if chars_by_register.get("zh_classical"):
            notes.append("文言检测难度远高于白话：名篇原文常被模型\"背过\"而显得像 AI，AI 仿写的文言又较少见，"
                         "请把文言部分的结果当作线索而非结论。")
        if main_reg == "zh" and str(cal.get("source", "")).startswith("NLPCC") and cal.get("calibrated"):
            notes.append("使用的是内置默认校准（公开数据集：学术摘要、新闻、作文）；用你自己的文字在管理页校准后会更贴合你的文风。")
        n_refonly = sum(1 for s in segs if s.kind == "reference_only")
        if n_refonly:
            notes.append(f"{n_refonly} 段诗词只给参考值、不计入 AI 率：评估发现诗词检测在不同来源之间很不稳定——"
                         "唐诗宋词名篇会被误判（约 19%），复述故事情节的 AI 诗又几乎认不出。"
                         "如需让诗词计入，可在管理页用你自己标注的诗词校准。")
        if memo:
            notes.append(f"有 {len(memo)} 段文字语言模型几乎能逐字复现、而分类器判为人写，疑为公开名篇原文"
                         "（如经典诗文、名人演讲）；这些段落不采信语言模型信号，只按分类器判断。")
        n_short = sum(1 for s in counted if is_short(s, score_text(s)))
        if n_short:
            notes.append(f"有 {n_short} 段篇幅较短（中文不足 {config.SHORT_CHARS_ZH} 字 / 英文不足 {config.SHORT_WORDS_EN} 词），"
                         "已标“篇幅短”，这些段落的结果波动较大。")
        if work_ai:
            notes.append(f"有 {len(work_ai)} 段 AI 概率接近阈值，但所在作品的大部分段落已判为疑似 AI，"
                         "按整篇判断计为“轻度疑似（整篇判断）”（AI 文章通常整篇生成）。")
        if near_chars and counted_chars:
            notes.append(f"另有 {near_chars / counted_chars:.0%} 的文字 AI 概率接近阈值（已标“接近阈值”，未计入 AI 率），"
                         "可重点复核；AI 翻译、经过改写或人工润色的 AI 文字常落在这一区间。")
        if sampled:
            notes.append(f"快速模式：语言模型只检测了 {len(lm_ids)} 段，其余段落仅用分类器。")
        if fallback_all:
            notes.append("全文都被识别为引文（引号对话多或文言虚词多）或参考文献，已改为全部计入计算。")
        elif excluded and sum(len(x.text) for x in excluded) > 0.3 * max(1, len(text)):
            notes.append("超过 30% 的文字因参考文献或引文被排除，AI 率只反映其余正文；"
                         "被排除的引文段落仍给出“参考值”。如需计入，请取消勾选“引文为主的段落不计入”。")
        if not (self.lm and self.lm.ready):
            notes.append("语言模型未就绪，本次只使用了分类器。")
        missing_cls = {REGISTER_NAMES.get(r, r) for r in chars_by_register if not self.classifier_for(r)}
        if missing_cls:
            notes.append("、".join(sorted(missing_cls)) + "分类器未就绪，这部分只用了语言模型特征。")

        return {
            "summary": {
                "ai_rate": rate(flagged_chars),
                "high_rate": rate(chars_by_level["high"]),
                "mid_rate": rate(chars_by_level["mid"]),
                "light_rate": rate(chars_by_level["light"]),
                "mean_prob": round(prob_weighted / counted_chars, 4) if counted_chars else None,
                "threshold": float(main_cal.get("threshold", 0.5)),
                "total_chars": len(text),
                "counted_chars": counted_chars,
                "flagged_chars": flagged_chars,
                "near_threshold_rate": rate(near_chars),
                "chars_by_level": chars_by_level,
                "chars_by_register": chars_by_register,
                "main_register": main_reg,
                "excluded_chars": sum(len(s.text) for s in excluded),
                "fallback_all_counted": fallback_all,
                "excluded_reference_segments": sum(1 for s in excluded if s.kind == "reference"),
                "excluded_quotation_segments": sum(1 for s in excluded if s.kind == "quotation"),
                "segments": len(segs),
                "segments_by_level": level_counts,
                "smoothing": config.SMOOTHING,
                "mode": mode,
                "lm_sampled": sampled,
                "lm_scored_segments": len(lm_ids) if (self.lm and self.lm.ready) else 0,
                "calibrated": bool(main_cal.get("calibrated")),
                "calibration_note": main_cal.get("note"),
                "calibration_features": (main_cal.get("lr") or {}).get("features"),
                "reliability_notes": notes,
                "methods": {
                    "fastdetect": bool(self.lm and self.lm.ready),
                    "binoculars": bool(self.lm and self.lm.ready),
                    "classifier": bool(self.cls and self.cls.ready),
                    "classifier_en": bool(self.cls_en and self.cls_en.ready),
                    "classifier_poetry": bool(self.cls_poetry and self.cls_poetry.ready),
                    "classifier_classical": bool(self.cls_classical and self.cls_classical.ready),
                },
                "elapsed_sec": round(time.time() - t_start, 1),
            },
            "segments": seg_out,
            "works": works_summary(seg_out),
        }

    def estimate_seconds(self, text: str, mode: str) -> float | None:
        if not self.tokens_per_sec or not (self.lm and self.lm.ready):
            return None
        chars = len(text)
        if mode == "fast":
            chars = min(chars, config.FAST_MODE_MAX_SEGMENTS * config.SEGMENT_TARGET_CHARS)
        tokens = chars / 1.4   # 中文约 1.4 字 / token（Qwen 分词器，经验值）
        return tokens * 2 / self.tokens_per_sec


# ---------------- 后台任务 ----------------

class JobQueue:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.q: queue.Queue = queue.Queue()
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def pending(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j["status"] in ("queued", "running"))

    def submit(self, kind: str, owner: str, payload: dict) -> dict:
        self._gc()
        jid = uuid.uuid4().hex[:16]
        job = {"id": jid, "kind": kind, "owner": owner, "status": "queued", "progress": 0.0,
               "done": 0, "total": 0, "created": time.time(), "started": None, "finished": None,
               "result": None, "error": None, "payload": payload,
               "eta_sec": self.engine.estimate_seconds(payload.get("text", ""), payload.get("mode", "full"))
               if kind == "detect" else None}
        with self.lock:
            self.jobs[jid] = job
        self.q.put(jid)
        return self.public(job)

    def get(self, jid: str) -> dict | None:
        with self.lock:
            return self.jobs.get(jid)

    def position(self, jid: str) -> int:
        with self.lock:
            queued = sorted((j for j in self.jobs.values() if j["status"] == "queued"), key=lambda j: j["created"])
        for i, j in enumerate(queued):
            if j["id"] == jid:
                return i + 1
        return 0

    def public(self, job: dict) -> dict:
        out = {k: v for k, v in job.items() if k not in ("payload", "owner")}
        if job["status"] == "queued":
            out["queue_position"] = self.position(job["id"])
        return out

    def _gc(self):
        now = time.time()
        with self.lock:
            for jid in [j for j, v in self.jobs.items()
                        if v["finished"] and now - v["finished"] > config.JOB_TTL_SECONDS]:
                del self.jobs[jid]

    def _worker(self):
        while True:
            jid = self.q.get()
            job = self.get(jid)
            if not job:
                continue
            job.update(status="running", started=time.time())

            def progress(done, total, job=job):
                job.update(done=done, total=total, progress=round(done / total, 4) if total else 1.0)
                if done and job["started"]:
                    elapsed = time.time() - job["started"]
                    job["eta_sec"] = round(elapsed / done * (total - done), 1)

            try:
                p = job["payload"]
                if job["kind"] == "detect":
                    job["result"] = self.engine.analyze(p["text"], p.get("mode", "full"),
                                                        p.get("exclude_references", True),
                                                        p.get("flag_quotations", True), progress)
                elif job["kind"] == "calibrate":
                    job["result"] = run_calibration(self.engine, p, progress)
                job["status"] = "done"
            except Exception as e:  # noqa: BLE001
                log.exception("job %s failed", jid)
                job.update(status="error", error=f"{type(e).__name__}: {e}")
            finally:
                job["finished"] = time.time()
                job["payload"] = {}
                job["progress"] = 1.0 if job["status"] == "done" else job["progress"]


def load_builtin_calib(profile: str) -> list[dict]:
    """随代码发布的内置校准数据（公开数据集的原始分数），见 tools/evaluate.py 的 write_calib_data。"""
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parent / "calib_data" / f"{profile}.json"
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text("utf-8"))
    except (OSError, ValueError):
        return []


def run_calibration(engine: Engine, p: dict, progress=None) -> dict:
    """用管理员提供的样本校准某一文体。profile=auto 时按样本中字数最多的文体确定。
    include_builtin=True（默认）时与内置公开数据合并：用户样本合计占约 30% 的权重，
    这样即使只标了几段（甚至只标了 AI 一类），也能在不破坏整体效果的前提下向你的文字偏移。"""
    engine.wait_loaded()

    forced = p.get("profile") if p.get("trust_register") and p.get("profile") not in (None, "", "auto") else None

    def to_segments(texts):
        out = []
        for t in texts:
            for s in segment_text(t, True, False):
                if forced:
                    # 来自报告页的逐段标注：文体已在检测时判定，直接按该文体使用（诗词、单独一段文言重新切分时可能被判成别的文体）
                    if s.kind != "reference" and len(s.text) >= min_chars(forced):
                        s.register = forced
                        out.append(s)
                elif s.counted and len(s.text) >= min_chars(s.register):
                    out.append(s)
        return out

    human = to_segments(p.get("human", []))
    ai = to_segments(p.get("ai", []))
    profile = p.get("profile") or "auto"
    if profile == "auto":
        by: dict[str, int] = {}
        for s in human + ai:
            by[s.register] = by.get(s.register, 0) + len(s.text)
        profile = max(by, key=by.get) if by else "zh"
    skipped = sum(1 for s in human + ai if s.register != profile)
    human = [score_text(s) for s in human if s.register == profile]
    ai = [score_text(s) for s in ai if s.register == profile]
    total = len(human) + len(ai)
    done = [0]

    def prog(_d, _t):
        done[0] += 1
        if progress:
            progress(done[0], total)

    hs = engine.raw_scores(human, prog, [profile] * len(human)) if human else []
    as_ = engine.raw_scores(ai, prog, [profile] * len(ai)) if ai else []
    include_builtin = p.get("include_builtin", True)
    builtin = load_builtin_calib(profile) if include_builtin else []
    n_user = len(hs) + len(as_)
    if builtin and n_user:
        w_user = max(1.0, config.USER_SAMPLE_SHARE * len(builtin) / ((1 - config.USER_SAMPLE_SHARE) * n_user))
        bh = [r["s"] for r in builtin if r["y"] == 0]
        ba = [r["s"] for r in builtin if r["y"] == 1]
        H, A = bh + hs, ba + as_
        hw, aw = [1.0] * len(bh) + [w_user] * len(hs), [1.0] * len(ba) + [w_user] * len(as_)
        hu, au = [False] * len(bh) + [True] * len(hs), [False] * len(ba) + [True] * len(as_)
    else:
        H, A, hw, aw, hu, au = hs, as_, None, None, None, None
    # 沿用当前这一文体使用的特征组合（诗词、英文等都是评估后选定的）
    cur = scoring.profile_for(engine.cal, profile)[0]
    feats = (cur.get("lr") or {}).get("features") or scoring.PROFILE_FEATURES.get(profile)
    res = scoring.calibrate(H, A, float(p.get("target_fpr", 0.05)), features=feats,
                            human_w=hw, ai_w=aw, human_user=hu, ai_user=au)
    res["calibration"]["models"] = {"observer": config.OBSERVER_MODEL, "performer": config.PERFORMER_MODEL,
                                    "classifier": config.classifier_for(profile)}
    res["calibration"]["profile"] = profile
    # 内置评估认定"不够稳定、只作参考"的文体（目前是诗词），除非管理员明确要求，否则保持只作参考
    if cur.get("reference_only") and not p.get("count_in_rate"):
        res["calibration"]["reference_only"] = True
    note = f"{REGISTER_NAMES.get(profile, profile)}：用你的 {len(hs)} 段人写、{len(as_)} 段 AI 样本校准"
    if builtin and n_user:
        note += f"（并合并内置公开数据 {len(builtin)} 段，你的样本约占 {config.USER_SAMPLE_SHARE:.0%} 权重）"
    res["calibration"]["note"] = note + "。"
    res["report"]["profile"] = profile
    res["report"]["profile_name"] = REGISTER_NAMES.get(profile, profile)
    res["report"]["skipped_other_register_segments"] = skipped
    res["report"]["builtin_samples"] = len(builtin) if (builtin and n_user) else 0
    res["report"]["user_human"] = len(hs)
    res["report"]["user_ai"] = len(as_)
    return res
