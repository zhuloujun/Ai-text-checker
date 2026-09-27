"""检测引擎：加载模型、逐段打分、汇总成报告；长文档放进后台任务队列。"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid

from . import config, scoring
from .detectors import stylometry
from .detectors.classifier import Classifier
from .detectors.lm_scorer import LMScorer
from .segmenter import segment_text

log = logging.getLogger("engine")

LEVELS = [(0.8, "high", "高度疑似"), (0.6, "mid", "中度疑似"), (0.0, "low", "")]


class Engine:
    def __init__(self):
        self.lm = LMScorer() if config.ENABLE_LM else None
        self.cls = Classifier() if (config.ENABLE_CLASSIFIER and config.CLASSIFIER_MODEL) else None
        self.loading = True
        self.load_started = time.time()
        self.cal, self.cal_source = self._load_cal()
        self.tokens_per_sec = None

    # ---------- 加载 ----------
    def _load_cal(self):
        override, source = config.load_calibration_override()
        if override:
            cal = dict(scoring.DEFAULTS)
            cal.update(override)
            return cal, source
        return dict(scoring.DEFAULTS), source

    def set_calibration(self, cal: dict, source: str):
        self.cal, self.cal_source = cal, source

    def load_all(self):
        for name, det in (("语言模型", self.lm), ("分类器", self.cls)):
            if det is None:
                continue
            try:
                det.load()
            except Exception as e:  # noqa: BLE001 —— 单个检测器失败不影响其他
                log.exception("failed to load %s", name)
                det.error = f"{type(e).__name__}: {e}"
        self.loading = False

    def status(self) -> dict:
        def st(det, model):
            if det is None:
                return {"enabled": False}
            return {"enabled": True, "ready": det.ready, "error": det.error, "model": model}
        return {
            "loading": self.loading,
            "lm": st(self.lm, [config.OBSERVER_MODEL, config.PERFORMER_MODEL]),
            "classifier": st(self.cls, config.CLASSIFIER_MODEL),
            "calibration": {"calibrated": bool(self.cal.get("calibrated")), "source": self.cal_source,
                            "threshold": self.cal.get("threshold"), "note": self.cal.get("note")},
            "tokens_per_sec": self.tokens_per_sec,
        }

    def any_ready(self) -> bool:
        return bool((self.lm and self.lm.ready) or (self.cls and self.cls.ready))

    # ---------- 打分 ----------
    def raw_scores(self, texts: list[str], progress=None) -> list[dict]:
        """对若干段文字算原始分数（校准时也用这个）。"""
        out = [{} for _ in texts]
        if self.cls and self.cls.ready:
            for i, p in enumerate(self.cls.predict(texts)):
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
        segs = segment_text(text, exclude_references, flag_quotations)
        counted = [s for s in segs if s.counted]
        lm_targets = counted
        sampled = False
        if mode == "fast" and len(counted) > config.FAST_MODE_MAX_SEGMENTS:
            step = len(counted) / config.FAST_MODE_MAX_SEGMENTS
            lm_targets = [counted[int(i * step)] for i in range(config.FAST_MODE_MAX_SEGMENTS)]
            sampled = True
        lm_ids = {s.index for s in lm_targets}

        results = {s.index: {} for s in segs}
        # 分类器很快：对所有正文段落都算
        if self.cls and self.cls.ready and counted:
            for s, p in zip(counted, self.cls.predict([s.text for s in counted])):
                results[s.index]["classifier"] = p
        # 语言模型较慢：只算正文（快速模式下抽样）
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
        thr = float(cal.get("threshold", 0.5))
        seg_out, flagged_chars, counted_chars, prob_weighted = [], 0, 0, 0.0
        level_counts = {"high": 0, "mid": 0, "low": 0}
        for s in segs:
            sc = results[s.index]
            comb = scoring.combine(sc, cal) if s.counted else {"prob": None, "signals": {}}
            prob = comb["prob"]
            level, label = "none", ""
            if s.counted and prob is not None:
                counted_chars += len(s.text)
                prob_weighted += prob * len(s.text)
                if prob >= thr:
                    flagged_chars += len(s.text)
                    level, label = ("high", "高度疑似") if prob >= max(0.8, thr) else ("mid", "疑似")
                else:
                    level = "low"
                level_counts[level] += 1
            seg_out.append({
                "index": s.index, "start": s.start, "chars": len(s.text), "text": s.text,
                "kind": s.kind, "notes": s.notes,
                "prob": None if prob is None else round(prob, 4),
                "level": level, "label": label,
                "signals": {k: (None if v is None else round(v, 4)) for k, v in comb["signals"].items()},
                "raw": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sc.items()},
                "style": stylometry.features(s.text) if s.counted else None,
                "scored_by_lm": s.index in lm_ids and "fastdetect" in sc,
            })

        excluded = [s for s in segs if not s.counted]
        return {
            "summary": {
                "ai_rate": round(flagged_chars / counted_chars, 4) if counted_chars else None,
                "mean_prob": round(prob_weighted / counted_chars, 4) if counted_chars else None,
                "threshold": thr,
                "total_chars": len(text),
                "counted_chars": counted_chars,
                "flagged_chars": flagged_chars,
                "excluded_chars": sum(len(s.text) for s in excluded),
                "excluded_reference_segments": sum(1 for s in excluded if s.kind == "reference"),
                "excluded_quotation_segments": sum(1 for s in excluded if s.kind == "quotation"),
                "segments": len(segs),
                "segments_by_level": level_counts,
                "mode": mode,
                "lm_sampled": sampled,
                "lm_scored_segments": len(lm_ids) if (self.lm and self.lm.ready) else 0,
                "calibrated": bool(cal.get("calibrated")),
                "calibration_note": cal.get("note"),
                "methods": {
                    "fastdetect": bool(self.lm and self.lm.ready),
                    "binoculars": bool(self.lm and self.lm.ready),
                    "classifier": bool(self.cls and self.cls.ready),
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
    def to_segments(texts):
        out = []
        for t in texts:
            out.extend(s.text for s in segment_text(t, True, True) if s.counted and len(s.text) >= config.SEGMENT_MIN_CHARS)
        return out

    human = to_segments(p.get("human", []))
    ai = to_segments(p.get("ai", []))
    total = len(human) + len(ai)
    done = [0]

    def prog(_d, _t):
        done[0] += 1
        if progress:
            progress(done[0], total)

    hs = engine.raw_scores(human, prog)
    as_ = engine.raw_scores(ai, prog)
    res = scoring.calibrate(hs, as_, float(p.get("target_fpr", 0.05)))
    res["calibration"]["models"] = {"observer": config.OBSERVER_MODEL, "performer": config.PERFORMER_MODEL,
                                    "classifier": config.CLASSIFIER_MODEL}
    return res
