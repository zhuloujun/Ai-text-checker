"""把各检测信号换算成"AI 概率"，并支持用你自己的样本校准。

未校准时：每个信号用一条逻辑斯蒂曲线换成 0–1 概率，再按权重平均（默认值见 DEFAULTS，是经验值）。
校准后：用你提供的"确定是人写的"和"确定是 AI 写的"样本，拟合一个逻辑回归，
并按"人写文本的误判率不超过目标值"来选阈值。
"""
from __future__ import annotations

import math

SIGNALS = ("fastdetect", "binoculars", "classifier")

DEFAULTS = {
    "version": 1,
    "calibrated": False,
    # direction=+1：分数越高越像 AI；-1：越低越像 AI
    "signals": {
        "fastdetect": {"center": 1.5, "scale": 0.6, "direction": 1},
        "binoculars": {"center": 0.90, "scale": 0.04, "direction": -1},
        "classifier": {"center": 0.5, "scale": 0.12, "direction": 1},
    },
    "weights": {"fastdetect": 0.35, "binoculars": 0.25, "classifier": 0.40},
    "threshold": 0.5,
    "note": "内置经验参数，未经你的样本校准，结果仅作相对参考。",
}


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def signal_prob(name: str, value, cal: dict):
    if value is None:
        return None
    s = cal["signals"][name]
    return _sigmoid(s["direction"] * (value - s["center"]) / s["scale"])


def features_of(scores: dict):
    """逻辑回归使用的特征：Fast-DetectGPT 原值、Binoculars 原值、分类器概率的 logit。"""
    fd, bi, cl = scores.get("fastdetect"), scores.get("binoculars"), scores.get("classifier")
    return [fd, bi, None if cl is None else _logit(cl)]


def combine(scores: dict, cal: dict) -> dict:
    """scores: {'fastdetect':x,'binoculars':y,'classifier':p}，任一可为 None。"""
    per = {k: signal_prob(k, scores.get(k), cal) for k in SIGNALS}
    lr = cal.get("lr")
    if lr:
        feats = features_of(scores)
        z, used = lr["b"], 0
        for i, f in enumerate(feats):
            if f is None:
                # 缺失特征用校准集的均值代替
                f = lr["mean"][i]
            else:
                used += 1
            z += lr["w"][i] * (f - lr["mean"][i]) / lr["std"][i]
        prob = _sigmoid(z) if used else None
    else:
        num = den = 0.0
        for k, p in per.items():
            if p is not None:
                w = cal["weights"].get(k, 0)
                num += w * p
                den += w
        prob = num / den if den else None
    return {"prob": prob, "signals": per}


# ---------------- 校准 ----------------

def _auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


def _fit_logreg(X: list[list[float]], y: list[int], l2: float = 1.0, iters: int = 200):
    """小规模逻辑回归（牛顿法 + L2），不依赖 sklearn。"""
    import numpy as np

    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    n, d = Xa.shape
    Xb = np.hstack([Xa, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - ya) + reg @ w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg
        step = np.linalg.solve(H + np.eye(d + 1) * 1e-9, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w[:-1].tolist(), float(w[-1])


def calibrate(human: list[dict], ai: list[dict], target_fpr: float = 0.05) -> dict:
    """human / ai：每个元素是一段文字的原始分数字典。返回新的校准 JSON 和评估指标。"""
    if len(human) < 5 or len(ai) < 5:
        raise ValueError(f"有效样本不足：人写 {len(human)} 段、AI {len(ai)} 段，每类至少需要 5 段（建议各 30 段以上）。"
                         "注意：参考文献和以引文为主的段落不参与校准；每段至少约 80 字。")

    report = {"n_human": len(human), "n_ai": len(ai), "auroc": {}}
    cal = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    cal["signals"] = {k: dict(v) for k, v in DEFAULTS["signals"].items()}

    # 1) 单信号：中心取两类中位数的中点，尺度取两类标准差的均值，方向由均值比较决定
    for k in SIGNALS:
        hs = [s[k] for s in human if s.get(k) is not None]
        as_ = [s[k] for s in ai if s.get(k) is not None]
        if len(hs) < 3 or len(as_) < 3:
            continue
        hs.sort(); as_.sort()
        mh, ma = hs[len(hs) // 2], as_[len(as_) // 2]
        sd = lambda v: (sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** 0.5
        direction = 1 if ma >= mh else -1
        cal["signals"][k] = {"center": (mh + ma) / 2, "scale": max((sd(hs) + sd(as_)) / 2, 1e-3), "direction": direction}
        au = _auroc(as_, hs)
        report["auroc"][k] = round(au if direction == 1 else 1 - au, 4)

    # 2) 组合：逻辑回归（特征标准化后拟合）
    import numpy as np
    rows, ys = [], []
    for y, group in ((0, human), (1, ai)):
        for s in group:
            f = features_of(s)
            if all(v is not None for v in f):
                rows.append(f); ys.append(y)
    if len(rows) >= 10 and len(set(ys)) == 2:
        Xa = np.asarray(rows, dtype=float)
        mean, std = Xa.mean(0), Xa.std(0) + 1e-9
        w, b = _fit_logreg(((Xa - mean) / std).tolist(), ys)
        cal["lr"] = {"w": w, "b": b, "mean": mean.tolist(), "std": std.tolist(),
                     "features": ["fastdetect", "binoculars", "logit(classifier)"]}

    # 3) 阈值：让人写文本的误判率不超过 target_fpr
    hp = sorted(p for p in (combine(s, cal)["prob"] for s in human) if p is not None)
    ap = [p for p in (combine(s, cal)["prob"] for s in ai) if p is not None]
    if hp:
        idx = min(len(hp) - 1, max(0, math.ceil(len(hp) * (1 - target_fpr)) - 1))
        thr = max(0.5, hp[idx] + 1e-6)
    else:
        thr = 0.5
    cal["threshold"] = round(min(thr, 0.99), 4)
    fpr = sum(p >= cal["threshold"] for p in hp) / len(hp) if hp else None
    tpr = sum(p >= cal["threshold"] for p in ap) / len(ap) if ap else None
    report["combined_auroc"] = round(_auroc(ap, hp), 4) if hp and ap else None
    report["threshold"] = cal["threshold"]
    report["human_flagged_rate"] = None if fpr is None else round(fpr, 4)
    report["ai_caught_rate"] = None if tpr is None else round(tpr, 4)
    report["note"] = "以上指标是在你提供的校准样本上算出的，样本越多、越接近你要检测的文本，越可信。"
    cal["calibrated"] = True
    cal["note"] = f"已用 {len(human)} 段人写文本、{len(ai)} 段 AI 文本校准。"
    return {"calibration": cal, "report": report}
