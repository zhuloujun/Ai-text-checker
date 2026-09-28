"""检测引擎：加载模型、逐段打分、汇总成报告；长文档放进后台任务队列。

每段先识别文体（现代汉语 / 文言 / 英文），再用该文体对应的分类器和校准参数打分——
与知网、维普、Turnitin 等平台"先分语种、再用专门模型判断"的做法一致。"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid

from . import config, scoring
from .detectors import stylometry
from .detectors.classifier import Classifier, make_english_classifier
from .detectors.lm_scorer import LMScorer
from .segmenter import REGISTER_NAMES, detect_register, segment_text

log = logging.getLogger("engine")


def style_scores(text: str, feats: dict | None = None) -> dict:
    """供校准使用的统计特征：句长变异系数、每千字套话数。"""
    f = feats or stylometry.features(text)
    return {"style_cv": f["sentence_len_cv"],
            "style_phrases": len(f["template_phrases"]) / max(len(text), 1) * 1000}


class Engine:
    def __init__(self):
        self.lm = LMScorer() if config.ENABLE_LM else None
        self.cls = Classifier() if (config.ENABLE_CLASSIFIER and config.CLASSIFIER_MODEL) else None
        self.cls_en = (make_english_classifier()
                       if (config.ENABLE_EN_CLASSIFIER and config.EN_CLASSIFIER_MODEL) else None)
        self.loading = True
        self.loaded_event = threading.Event()
        self.load_started = time.time()
        self.cal, self.cal_source = self._load_cal()
        self.tokens_per_sec = None

    # ---------- 加载 ----------
    def _load_cal(self):
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
            for name, det in (("语言模型", self.lm), ("中文分类器", self.cls), ("英文分类器", self.cls_en)):
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
        for k in ("zh_classical", "en"):
            profiles[k] = bool(((self.cal.get("profiles") or {}).get(k) or {}).get("calibrated"))
        return {
            "loading": self.loading,
            "lm": st(self.lm, [config.OBSERVER_MODEL, config.PERFORMER_MODEL]),
            "classifier": st(self.cls, config.CLASSIFIER_MODEL),
            "classifier_en": st(self.cls_en, config.EN_CLASSIFIER_MODEL),
            "calibration": {"calibrated": bool(self.cal.get("calibrated")), "source": self.cal_source,
                            "threshold": self.cal.get("threshold"), "note": self.cal.get("note"),
                            "profiles": profiles},
            "tokens_per_sec": self.tokens_per_sec,
        }

    def any_ready(self) -> bool:
        return bool((self.lm and self.lm.ready) or (self.cls and self.cls.ready) or (self.cls_en and self.cls_en.ready))

    def classifier_for(self, register: str):
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

    def raw_scores(self, texts: list[str], progress=None, registers: list[str] | None = None) -> list[dict]:
        """对若干段文字算原始分数（校准、评估时也用这个）。"""
        registers = registers or [detect_register(t) for t in texts]
        out = [style_scores(t) for t in texts]
        for i, p in enumerate(self.classify(texts, registers)):
            if p is not None:
                out[i]["classifier"] = p
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
        counted = [s for s in segs if s.counted]
        # 不计入的引文段落也打分，作为"参考值"显示（不影响 AI 率）；参考文献条目没有检测意义，不打分
        ref_scored = [s for s in segs if s.kind == "quotation"]
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
        for s, p in zip(scored, self.classify([s.text for s in scored], [s.register for s in scored])):
            if p is not None:
                results[s.index]["classifier"] = p
        # 语言模型较慢：快速模式下抽样
        if self.lm and self.lm.ready:
            done = 0
            for s in segs:
                if s.index not in lm_ids:
                    continue
                t0 = time.time()
                r = self.lm.score(s.text)
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
        styles = {s.index: stylometry.features(s.text) for s in scored}
        for s in scored:
            results[s.index].update(style_scores(s.text, styles[s.index]))

        # 1) 每段按自己的文体用对应的校准参数打分（引文段落的分数只作参考值）
        scored_ids = {s.index for s in scored}
        prof = {}
        for s in segs:
            prof[s.index] = scoring.profile_for(cal, s.register)
        combos = {s.index: (scoring.combine(results[s.index], prof[s.index][0]) if s.index in scored_ids
                            else {"prob": None, "signals": {}})
                  for s in segs}
        # 2) 与相邻正文段落平滑（只在正文段落之间进行）
        body_idx = [s.index for s in counted]
        raw_probs = [combos[i]["prob"] for i in body_idx]
        smoothed = dict(zip(body_idx, scoring.smooth(raw_probs, config.SMOOTHING)))

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
                "kind": s.kind, "notes": s.notes, "register": s.register,
                "register_name": REGISTER_NAMES.get(s.register, s.register),
                "threshold": thr,
                "prob": None if prob is None else round(prob, 4),
                "ref_prob": (None if s.counted or comb["prob"] is None else round(comb["prob"], 4)),
                "prob_unsmoothed": None if comb["prob"] is None else round(comb["prob"], 4),
                "level": level, "label": label, "near_threshold": near,
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
        if chars_by_register.get("zh_classical"):
            notes.append("文言检测难度远高于白话：名篇原文常被模型\"背过\"而显得像 AI，AI 仿写的文言又较少见，"
                         "请把文言部分的结果当作线索而非结论。")
        if main_reg == "zh" and str(cal.get("source", "")).startswith("NLPCC") and cal.get("calibrated"):
            notes.append("使用的是内置默认校准（公开数据集：学术摘要、新闻、作文）；用你自己的文字在管理页校准后会更贴合你的文风。")
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
                },
                "elapsed_sec": round(time.time() - t_start, 1),
            },
            "segments": seg_out,
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


def run_calibration(engine: Engine, p: dict, progress=None) -> dict:
    """用管理员提供的样本校准某一文体。profile=auto 时按样本中字数最多的文体确定。"""
    engine.wait_loaded()

    def to_segments(texts):
        out = []
        for t in texts:
            for s in segment_text(t, True, False):
                if s.counted and len(s.text) >= config.SEGMENT_MIN_CHARS:
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
    human = [s.text for s in human if s.register == profile]
    ai = [s.text for s in ai if s.register == profile]
    total = len(human) + len(ai)
    done = [0]

    def prog(_d, _t):
        done[0] += 1
        if progress:
            progress(done[0], total)

    hs = engine.raw_scores(human, prog, [profile] * len(human))
    as_ = engine.raw_scores(ai, prog, [profile] * len(ai))
    res = scoring.calibrate(hs, as_, float(p.get("target_fpr", 0.05)), features=scoring.PROFILE_FEATURES.get(profile))
    res["calibration"]["models"] = {"observer": config.OBSERVER_MODEL, "performer": config.PERFORMER_MODEL,
                                    "classifier": config.classifier_for(profile)}
    res["calibration"]["profile"] = profile
    res["calibration"]["note"] = (f"{REGISTER_NAMES.get(profile, profile)}：" + res["calibration"].get("note", ""))
    res["report"]["profile"] = profile
    res["report"]["profile_name"] = REGISTER_NAMES.get(profile, profile)
    res["report"]["skipped_other_register_segments"] = skipped
    return res
