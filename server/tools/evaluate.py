"""用公开的中文数据集评估检测效果，并生成默认校准参数。

数据：NLPCC 2025 Shared Task 1（中文 AI 生成文本检测）
      https://github.com/NLP2CT/NLPCC-2025-Task1
      训练集含 3 个领域（CSL 中文科技论文摘要、CNewSum 新闻、ASAP 作文），
      AI 文本由 GPT-4o、GLM-4-flash、Qwen-turbo 生成；带标签测试集另含 DeepSeek-V3 生成的文本。

流程：
  1. 从训练集按"领域 × 模型"均衡抽样 → 用于拟合校准参数（逻辑回归 + 5 折交叉验证选阈值）
  2. 从带标签测试集（训练时没见过）另抽样 → 独立评估：AUROC、检出率、误判率
  3. 写出 default_calibration.json 和 EVAL_REPORT.md

用法（GitHub Actions 里自动运行，见 .github/workflows/evaluate.yml）：
  python tools/evaluate.py --data <数据目录> --n-cal 900 --n-test 600 --target-fpr 0.05
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, scoring  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.segmenter import segment_text  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def balanced_sample(rows, n, keyfn, seed):
    groups = {}
    for r in rows:
        groups.setdefault(keyfn(r), []).append(r)
    rnd = random.Random(seed)
    per = max(1, n // len(groups))
    out = []
    for k in sorted(groups):
        g = groups[k][:]
        rnd.shuffle(g)
        out.extend(g[:per])
    return out


def to_segment(text):
    """与正式检测一致：按同样规则分段，取最长的一段正文（样本本身多为单段，约 200–1800 字）。"""
    segs = [s.text for s in segment_text(text, True, True) if s.counted]
    if not segs:
        return None
    return max(segs, key=len)


def auroc(pos, neg):
    return scoring._auroc(pos, neg)


def build_samples(args):
    train = load(Path(args.data) / "train.json")
    test = load(Path(args.data) / "test_with_label.json")
    # 校准集：人写与 AI 各一半；AI 在 3 个模型间均衡；领域偏重 CSL（学术摘要）
    human = [r for r in train if r["label"] == 0]
    ai = [r for r in train if r["label"] == 1]
    weights = {"csl": 0.5, "cnewsum": 0.25, "asap": 0.25}
    cal_h, cal_a = [], []
    for dom, w in weights.items():
        k = int(args.n_cal / 2 * w)
        cal_h += balanced_sample([r for r in human if r["source"] == dom], k, lambda r: r["source"], 1)
        cal_a += balanced_sample([r for r in ai if r["source"] == dom], k, lambda r: r["model"], 2)
    cal = cal_h + cal_a
    random.Random(7).shuffle(cal)
    test_s = balanced_sample([r for r in test if len(r["text"]) >= 150], args.n_test, lambda r: r["label"], 3)
    used = {id(r) for r in cal}
    csl_hold = balanced_sample([r for r in train if r["source"] == "csl" and id(r) not in used],
                               args.n_csl, lambda r: (r["label"], r["model"]), 4)
    half = len(cal) // 2
    return {"cal1": cal[:half], "cal2": cal[half:], "test": test_s, "csl": csl_hold}


def stage_score(args):
    """打分阶段：只对其中一份样本打分，结果写到 --scores-dir/<part>.json（可在多台机器上并行）。"""
    parts = build_samples(args)
    rows = parts[args.part]
    engine = Engine()
    engine.load_all()
    if not engine.any_ready():
        raise SystemExit(f"模型加载失败：{engine.status()}")
    segs, labels, meta = [], [], []
    for r in rows:
        sg = to_segment(r["text"])
        if sg and len(sg) >= config.SEGMENT_MIN_CHARS:
            segs.append(sg); labels.append(r["label"]); meta.append(r.get("model") or "")
    print(f"[{args.part}] 打分 {len(segs)} 段 …", flush=True)
    t = time.time(); last = [0.0]

    def prog(d, n):
        if time.time() - last[0] > 60:
            last[0] = time.time()
            print(f"  {d}/{n}  {time.time()-t:.0f}s", flush=True)
    sc = engine.raw_scores(segs, prog)
    out = [{"y": l, "model": m, "s": {k: v for k, v in x.items() if isinstance(v, (int, float))}}
           for x, l, m in zip(sc, labels, meta)]
    d = Path(args.scores_dir); d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.part}.json").write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    print(f"::notice title=打分完成 {args.part}::{len(out)} 段，用时 {time.time()-t:.0f} 秒", flush=True)


def stage_fit(args):
    """拟合与评估阶段：读取各份打分结果，拟合校准参数，在没见过的测试集上评估。"""
    d = Path(args.scores_dir)
    parts = {}
    for name in ("cal1", "cal2", "test", "csl"):
        f = d / f"{name}.json"
        if f.exists():
            parts[name] = json.loads(f.read_text("utf-8"))
    cal_rows = parts.get("cal1", []) + parts.get("cal2", [])
    if not cal_rows:
        raise SystemExit("没有校准集打分结果")
    hs = [r["s"] for r in cal_rows if r["y"] == 0]
    as_ = [r["s"] for r in cal_rows if r["y"] == 1]
    res = scoring.calibrate(hs, as_, args.target_fpr)
    cal = res["calibration"]
    cal["models"] = {"observer": config.OBSERVER_MODEL, "performer": config.PERFORMER_MODEL,
                     "classifier": config.CLASSIFIER_MODEL}
    cal["source"] = "NLPCC 2025 Task 1（CSL 学术摘要 / 新闻 / 作文；GPT-4o、GLM-4、Qwen）"
    cal["note"] = (f"内置默认校准：用公开数据集 NLPCC 2025 的 {len(hs)} 段人写、{len(as_)} 段 AI 文本拟合；"
                   "建议再用你自己的文字校准。")
    out = Path(args.out)
    (out / "app" / "default_calibration.json").write_text(json.dumps(cal, ensure_ascii=False, indent=1), "utf-8")

    def evaluate(rows, name):
        probs = [scoring.combine(r["s"], cal)["prob"] for r in rows]
        thr = cal["threshold"]
        y = [r["y"] for r in rows]
        hp = [p for p, l in zip(probs, y) if l == 0 and p is not None]
        ap_ = [p for p, l in zip(probs, y) if l == 1 and p is not None]
        res_ = {"name": name, "n_human": len(hp), "n_ai": len(ap_),
                "auroc": round(auroc(ap_, hp), 4) if hp and ap_ else None,
                "ai_caught": round(sum(p >= thr for p in ap_) / len(ap_), 4) if ap_ else None,
                "human_flagged": round(sum(p >= thr for p in hp) / len(hp), 4) if hp else None,
                "per_signal_auroc": {}, "ai_caught_by_model": {}}
        for k in scoring.EXTENDED_FEATURES:
            pv = [scoring.feature_value(r["s"], k) for r in rows if r["y"] == 1]
            nv = [scoring.feature_value(r["s"], k) for r in rows if r["y"] == 0]
            pv = [v for v in pv if v is not None]; nv = [v for v in nv if v is not None]
            a = auroc(pv, nv)
            if a is not None:
                res_["per_signal_auroc"][k] = round(a, 4)
        by = {}
        for p, r in zip(probs, rows):
            if r["y"] == 1 and p is not None and r.get("model"):
                by.setdefault(r["model"], []).append(p >= thr)
        res_["ai_caught_by_model"] = {m: round(sum(v) / len(v), 4) for m, v in by.items()}
        return res_

    ev = []
    if parts.get("test"):
        ev.append(evaluate(parts["test"], "NLPCC 测试集（训练时未见，含 DeepSeek-V3）"))
    if parts.get("csl"):
        ev.append(evaluate(parts["csl"], "CSL 学术摘要保留集"))
    before = None
    if parts.get("test"):
        op = [scoring.combine(r["s"], scoring.DEFAULTS)["prob"] for r in parts["test"]]
        yy = [r["y"] for r in parts["test"]]
        oa = [p for p, l in zip(op, yy) if l == 1 and p is not None]
        oh = [p for p, l in zip(op, yy) if l == 0 and p is not None]
        before = {"ai_caught": round(sum(p >= 0.5 for p in oa) / max(1, len(oa)), 4),
                  "human_flagged": round(sum(p >= 0.5 for p in oh) / max(1, len(oh)), 4),
                  "auroc": round(auroc(oa, oh), 4) if oa and oh else None}

    report = {"calibration_report": res["report"], "evaluation": ev, "before_calibration_on_test": before,
              "models": cal["models"]}
    (out / "tools" / "eval_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    lines = ["# 检测效果评估报告", "",
             f"模型：{cal['models']['observer']} + {cal['models']['performer']} + {cal['models']['classifier']}",
             f"数据：NLPCC 2025 Task 1。校准集 {res['report']['n_human']} 人写 / {res['report']['n_ai']} AI。", ""]
    if before:
        lines += [f"**校准前**（旧经验参数，阈值 0.5）在测试集上：AI 检出率 {before['ai_caught']:.1%}，人写误判率 {before['human_flagged']:.1%}，AUROC {before['auroc']}", ""]
    lines += [f"**校准后**：阈值 {cal['threshold']}，特征 {', '.join(res['report'].get('features_used') or [])}；"
              f"交叉验证 AUROC {res['report'].get('combined_auroc')}", ""]
    for e in ev:
        lines.append(f"## {e['name']}")
        lines.append(f"- AUROC {e['auroc']}；AI 检出率 {e['ai_caught']:.1%}；人写误判率 {e['human_flagged']:.1%}（{e['n_ai']} AI / {e['n_human']} 人写）")
        if e["ai_caught_by_model"]:
            lines.append("- 各模型检出率：" + "，".join(f"{m} {v:.1%}" for m, v in e["ai_caught_by_model"].items()))
        lines.append("- 各特征 AUROC（>0.5 越大越像 AI，<0.5 越小越像 AI）：" +
                     "，".join(f"{k} {v}" for k, v in e["per_signal_auroc"].items()))
        lines.append("")
    (out / "tools" / "EVAL_REPORT.md").write_text("\n".join(lines), "utf-8")
    print("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    for e in ev:
        print(f"::notice title={e['name']}::AUROC {e['auroc']} · AI 检出率 {e['ai_caught']:.1%} · 人写误判率 {e['human_flagged']:.1%} · "
              + " · ".join(f"{m} {v:.0%}" for m, v in e["ai_caught_by_model"].items()), flush=True)
    if before:
        print(f"::notice title=校准前对照::AI 检出率 {before['ai_caught']:.1%} · 人写误判率 {before['human_flagged']:.1%} · AUROC {before['auroc']}", flush=True)
    print("::notice title=校准报告::" + json.dumps(res["report"], ensure_ascii=False)[:3000], flush=True)
    if ev:
        print("::notice title=各特征 AUROC::" + json.dumps(ev[0]["per_signal_auroc"], ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "fit"], required=True)
    ap.add_argument("--part", choices=["cal1", "cal2", "test", "csl"])
    ap.add_argument("--data")
    ap.add_argument("--scores-dir", default="/tmp/eval_scores")
    ap.add_argument("--n-cal", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-csl", type=int, default=240)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()
    if args.stage == "score":
        if not args.part or not args.data:
            raise SystemExit("--stage score 需要 --part 和 --data")
        stage_score(args)
    else:
        stage_fit(args)


if __name__ == "__main__":
    main()
