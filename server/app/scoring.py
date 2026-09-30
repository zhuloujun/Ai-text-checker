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


# 逻辑回归可用的特征。BASE 在样本少时使用；样本足够多（每类 ≥ 30 段）时使用 EXTENDED。
BASE_FEATURES = ["fastdetect", "binoculars", "logit_classifier"]
EXTENDED_FEATURES = BASE_FEATURES + [
    "fastdetect_norm", "lrr", "log_rank", "entropy", "top10", "lp_burstiness",
    "style_cv", "style_phrases",
]
_LEGACY_NAMES = {"logit(classifier)": "logit_classifier"}


def feature_value(scores: dict, name: str):
    name = _LEGACY_NAMES.get(name, name)
    if name == "logit_classifier":
        cl = scores.get("classifier")
        return None if cl is None else _logit(cl)
    if name == "logit_classifier_mpu":
        cl = scores.get("classifier_mpu")
        return None if cl is None else _logit(cl)
    v = scores.get(name)
    return None if v is None else float(v)


def features_of(scores: dict, names=None):
    return [feature_value(scores, n) for n in (names or BASE_FEATURES)]


def combine(scores: dict, cal: dict) -> dict:
    """scores: {'fastdetect':x,'binoculars':y,'classifier':p}，任一可为 None。"""
    per = {k: signal_prob(k, scores.get(k), cal) for k in SIGNALS}
    lr = cal.get("lr")
    if lr:
        feats = features_of(scores, lr.get("features") or BASE_FEATURES)
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


def _fit_logreg(X: list[list[float]], y: list[int], l2: float = 1.0, iters: int = 200, weights=None):
    """小规模逻辑回归（牛顿法 + L2），不依赖 sklearn。两类样本数不同时按类别加权（各占一半权重），
    避免样本多的一类把概率整体拉偏。weights：每个样本的额外权重（用户标注的样本权重更高）。"""
    import numpy as np

    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    n, d = Xa.shape
    wa = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    n1 = max((wa * ya).sum(), 1e-9)
    n0 = max((wa * (1 - ya)).sum(), 1e-9)
    tot = wa.sum()
    sw = wa * np.where(ya == 1, tot / (2 * n1), tot / (2 * n0))
    Xb = np.hstack([Xa, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (sw * (p - ya)) + reg @ w
        H = (Xb * (sw * p * (1 - p))[:, None]).T @ Xb + reg
        step = np.linalg.solve(H + np.eye(d + 1) * 1e-9, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w[:-1].tolist(), float(w[-1])


def calibrate(human: list[dict], ai: list[dict], target_fpr: float = 0.05, features: list | None = None,
              human_w: list | None = None, ai_w: list | None = None,
              human_user: list | None = None, ai_user: list | None = None) -> dict:
    """human / ai：每个元素是一段文字的原始分数字典。返回新的校准 JSON 和评估指标。
    human_w / ai_w：各样本权重；human_user / ai_user：是否为用户自己标注的样本（单独报告这部分的效果）。"""
    human_w = human_w or [1.0] * len(human)
    ai_w = ai_w or [1.0] * len(ai)
    human_user = human_user or [False] * len(human)
    ai_user = ai_user or [False] * len(ai)
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

    # 2) 组合：逻辑回归。每类 ≥ 30 段时使用扩展特征，否则只用三个主信号。
    import numpy as np
    n_each = min(len(human), len(ai))
    candidates = list(features) if features else (EXTENDED_FEATURES if n_each >= 30 else BASE_FEATURES)
    all_rows = [(features_of(s, candidates), 0) for s in human] + [(features_of(s, candidates), 1) for s in ai]
    all_w = list(human_w) + list(ai_w)
    all_u = list(human_user) + list(ai_user)
    # 丢掉缺失太多的特征（例如没有语言模型时的那些）
    names = [n for j, n in enumerate(candidates)
             if sum(r[j] is not None for r, _ in all_rows) >= 0.9 * len(all_rows)]
    for j, n in enumerate(candidates):
        if n in EXTENDED_FEATURES and n not in BASE_FEATURES:
            hv = [r[j] for r, y in all_rows if y == 0 and r[j] is not None]
            av = [r[j] for r, y in all_rows if y == 1 and r[j] is not None]
            au = _auroc(av, hv)
            if au is not None:
                report["auroc"][n] = round(max(au, 1 - au), 4)
    idx = [candidates.index(n) for n in names]
    X, ys, ws, us = [], [], [], []
    for (r, y), w_, u_ in zip(all_rows, all_w, all_u):
        vals = [r[j] for j in idx]
        if sum(v is None for v in vals) <= 1:
            X.append(vals); ys.append(y); ws.append(w_); us.append(u_)
    hw, uw = None, None
    if names and len(X) >= 10 and len(set(ys)) == 2:
        Xa = np.array([[np.nan if v is None else v for v in row] for row in X], dtype=float)
        mean = np.nanmean(Xa, 0)
        Xa = np.where(np.isnan(Xa), mean, Xa)
        std = Xa.std(0) + 1e-9
        Z = (Xa - mean) / std
        ya = np.asarray(ys)
        l2 = 1.0 if len(names) <= 3 else 3.0
        wa_ = np.asarray(ws, dtype=float)
        ua_ = np.asarray(us, dtype=bool)
        w, b = _fit_logreg(Z.tolist(), ys, l2=l2, weights=wa_)
        cal["lr"] = {"w": w, "b": b, "mean": mean.tolist(), "std": std.tolist(), "features": names}
        report["features_used"] = names
        # 5 折交叉验证：用"没见过的样本"上的预测来评估和选阈值，避免过于乐观
        oof = np.full(len(ya), np.nan)
        if len(ya) >= 20:
            rng = np.random.default_rng(0)
            order = rng.permutation(len(ya))
            folds = np.array_split(order, 5)
            for f in folds:
                tr = np.setdiff1d(order, f)
                if len(set(ya[tr])) < 2:
                    continue
                wf, bf = _fit_logreg(Z[tr].tolist(), ya[tr].tolist(), l2=l2, weights=wa_[tr])
                oof[f] = 1 / (1 + np.exp(-(Z[f] @ np.asarray(wf) + bf)))
        cv_ok = not np.isnan(oof).any()
        report["cross_validated"] = bool(cv_ok)
        if cv_ok:
            hp = sorted(oof[ya == 0].tolist())
            ap = oof[ya == 1].tolist()
            order_h = np.argsort(oof[ya == 0])
            hw = wa_[ya == 0][order_h].tolist()
            if ua_.any():
                uw = {"human": oof[(ya == 0) & ua_].tolist(), "ai": oof[(ya == 1) & ua_].tolist()}
        else:
            hp = sorted(combine(s, cal)["prob"] for s in human if combine(s, cal)["prob"] is not None)
            ap = [combine(s, cal)["prob"] for s in ai if combine(s, cal)["prob"] is not None]
    else:
        hp = sorted(p for p in (combine(s, cal)["prob"] for s in human) if p is not None)
        ap = [p for p in (combine(s, cal)["prob"] for s in ai) if p is not None]

    # 3) 阈值：让人写文本的误判率不超过 target_fpr
    if hp and hw and len(set(hw)) > 1:
        # 加权：从高往低累加人写样本的权重，超过目标误判率之前的最低概率即阈值
        total_w = sum(hw)
        acc, thr = 0.0, 0.5
        for p_, w_ in sorted(zip(hp, hw), reverse=True):
            if (acc + w_) / total_w > target_fpr:
                thr = max(0.5, p_ + 1e-6)
                break
            acc += w_
    elif hp:
        k = min(len(hp) - 1, max(0, math.ceil(len(hp) * (1 - target_fpr)) - 1))
        thr = max(0.5, hp[k] + 1e-6)
    else:
        thr = 0.5
    cal["threshold"] = round(min(thr, 0.99), 4)
    fpr = sum(p >= cal["threshold"] for p in hp) / len(hp) if hp else None
    tpr = sum(p >= cal["threshold"] for p in ap) / len(ap) if ap else None
    report["combined_auroc"] = round(_auroc(ap, hp), 4) if hp and ap else None
    report["threshold"] = cal["threshold"]
    report["human_flagged_rate"] = None if fpr is None else round(fpr, 4)
    report["ai_caught_rate"] = None if tpr is None else round(tpr, 4)
    if uw:
        t_ = cal["threshold"]
        report["user_samples"] = {
            "n_human": len(uw["human"]), "n_ai": len(uw["ai"]),
            "ai_caught_rate": round(sum(p >= t_ for p in uw["ai"]) / len(uw["ai"]), 4) if uw["ai"] else None,
            "human_flagged_rate": round(sum(p >= t_ for p in uw["human"]) / len(uw["human"]), 4) if uw["human"] else None,
        }
    report["note"] = ("以上指标" + ("用 5 折交叉验证（每次用没参与拟合的样本）算出" if report.get("cross_validated") else "是在校准样本上直接算出的，可能偏乐观")
                      + "；样本越多、越接近你要检测的文本，越可信。")
    cal["calibrated"] = True
    cal["note"] = f"已用 {len(human)} 段人写文本、{len(ai)} 段 AI 文本校准。"
    return {"calibration": cal, "report": report}


# ---------------- 按文体分别校准 ----------------
# 校准 JSON 的顶层是"现代汉语"参数；profiles 里放其他文体（en 英文、zh_classical 文言、zh_poetry 诗词）各自的参数。
# 各文体的文字特征差别很大（英文用英文分类器；文言的困惑度分布与白话完全不同），不能共用一套阈值。

PROFILE_NAMES = {"zh": "现代汉语", "zh_short": "现代汉语短段", "zh_classical": "文言", "zh_poetry": "诗词", "en": "英文"}
# 各文体用哪些特征做组合（用训练时没见过的评估集比较后选定，见 tools/EVAL_REPORT.md）：
# 英文：分类器 + Fast-DetectGPT + Binoculars 三个主信号，在 GPT-4 新领域和改写文本上都优于全部特征；
# 现代汉语：全部扩展特征更好。
# 文言：有了文言专用分类器后，三个主信号与全部特征的交叉验证几乎相同（0.9960 vs 0.9956），但在没见过的
#   DeepSeek / Kimi / 文心一言仿古文上，全部特征只认出 81%，三个主信号认出 94%（古籍保留集误判 2.8%，与接入分类器前持平）：
#   困惑度、预测熵等扩展特征会被"刻意仿古"的文风带偏，把 AI 仿写拉回"像人写"。故文言只用三个主信号。
PROFILE_FEATURES = {"en": BASE_FEATURES, "zh_classical": BASE_FEATURES}
NEAR_MARGIN = 0.15   # 低于阈值不到这么多的段落标为"接近阈值"（不计入 AI 率）


def profile_for(cal: dict, register: str):
    """返回 (该文体使用的校准参数, 是否有专门校准)。"""
    if register in (None, "", "zh"):
        return cal, bool(cal.get("calibrated"))
    prof = (cal.get("profiles") or {}).get(register)
    if prof:
        merged = dict(DEFAULTS)
        merged.update(prof)
        return merged, bool(prof.get("calibrated"))
    if register == "en":
        # 英文没有专门校准时，现代汉语的参数没有意义（分类器都不一样），退回经验值
        return dict(DEFAULTS), False
    return cal, False


def merge_profile(base: dict, new: dict, profile: str) -> dict:
    """把某一文体的新校准并入整套校准，其他文体保持不变。"""
    base = dict(base or DEFAULTS)
    profiles = dict(base.get("profiles") or {})
    if profile in (None, "", "zh"):
        out = {k: v for k, v in new.items() if k not in ("profile", "profiles")}
        if profiles:
            out["profiles"] = profiles
        return out
    profiles[profile] = {k: v for k, v in new.items() if k != "profiles"}
    base["profiles"] = profiles
    return base


# ---------------- 相邻段落平滑与分级 ----------------

def smooth(probs: list, strength: float) -> list:
    """在 logit 空间里把每段的概率与前后段落做加权平均（参考 Turnitin 的滑动窗口做法），
    减少孤立段落的偶然误判。strength=0 表示不平滑。None 保持不变。"""
    if strength <= 0 or len(probs) < 2:
        return list(probs)
    z = [None if p is None else _logit(p) for p in probs]
    out = []
    for i, zi in enumerate(z):
        if zi is None:
            out.append(None)
            continue
        nb = [v for v in (z[i - 1] if i > 0 else None, z[i + 1] if i + 1 < len(z) else None) if v is not None]
        if not nb:
            out.append(probs[i])
            continue
        zn = sum(nb) / len(nb)
        out.append(_sigmoid((1 - strength) * zi + strength * zn))
    return out


LEVELS = (("high", "高度疑似", 0.80), ("mid", "中度疑似", 0.65), ("light", "轻度疑似", 0.0))


def level_of(prob, threshold: float):
    """返回 (level, 中文标签)。低于阈值为 low。"""
    if prob is None:
        return "none", ""
    if prob < threshold:
        return "low", ""
    for key, label, lower in LEVELS:
        if prob >= max(threshold, lower):
            return key, label
    return "light", "轻度疑似"
