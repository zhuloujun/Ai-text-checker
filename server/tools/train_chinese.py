"""训练"中文第二分类器"：专门识别新一代国产大模型（DeepSeek / Kimi / 文心 / 千问 / 豆包）写的中文，
尤其是 MPU 中文分类器容易漏掉的文学类文字（散文、游记、回忆、读后感）。

数据（全部按"题目"划分训练 / 评估，评估题目的真人版和 AI 版都不参与训练）：
  · AI：tools/data/gen_zh/ai_*.jsonl（各家按一句话指令写的长文，每篇按网站同一套分段规则取多段）、
        NLPCC 2025 Task 1 的 GPT-4o / GLM / Qwen 文本、仓库自带的 AI 读后感 / 散文；
  · 真人：NLPCC 的论文摘要 / 新闻 / 点评（2022 年前）、tools/data/gen_zh/human_*.jsonl
        （知乎长回答、网络小说、中文网页、高考现代文阅读材料）——真人叙事和散文必须足量，
        否则模型会学成"文笔好 = AI"。
评估除了逐段的 1% / 5% 误判率，还报告"整篇"指标：每篇文档各段得分按字数加权的中位数，
真人文档最高多少、AI 文档有多少能超过它——线上用它做整篇判断。

用法：python tools/train_chinese.py --nlpcc <NLPCC data 目录> --out <输出目录>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate as ev  # noqa: E402

GEN = ev.DATA_DIR / "gen_zh"


def split_of(key: str) -> str:
    return "test" if int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 5 == 0 else "train"


def _wmedian(vals):
    vals = sorted(vals)
    half, acc = sum(n for _, n in vals) / 2, 0
    for v, n in vals:
        acc += n
        if acc >= half:
            return v
    return vals[-1][0]


def load_jsonl(f: Path) -> list[dict]:
    return [json.loads(l) for l in f.read_text("utf-8").splitlines() if l.strip()] if f.exists() else []


def segments(text: str) -> list[tuple[str, int]]:
    """与线上一致的分段：只取现代汉语正文段（≥ 100 字）。"""
    from app.segmenter import segment_text
    return [(s.text, len(s.text)) for s in segment_text(text) if s.counted and s.register == "zh"
            and len(re.findall(r"[一-鿿]", s.text)) >= 100]


def build(args, rnd):
    docs = []          # {"doc", "y", "src", "split", "segs": [(text, n)]}
    for f in sorted(GEN.glob("ai_*.jsonl")):
        for r in load_jsonl(f):
            docs.append({"doc": f"{f.stem}:{r['title']}:{r.get('genre')}", "y": 1, "src": f"gen-{f.stem[3:]}-{r.get('genre')}",
                         "split": split_of(r["title"]), "segs": segments(r["text"])})
    for f in sorted(GEN.glob("human_*.jsonl")):
        for r in load_jsonl(f):
            docs.append({"doc": r["title"], "y": 0, "src": f"human-{f.stem[6:]}", "split": split_of(r["title"]),
                         "segs": segments(r["text"])})
    nl = json.loads((Path(args.nlpcc) / "train.json").read_text("utf-8"))
    by = {}
    for r in nl:
        by.setdefault((r["source"], int(r["label"]), r.get("model")), []).append(r)
    for (src, y, model), rows in by.items():
        rnd.shuffle(rows)
        for r in rows[: args.n_nlpcc_human if y == 0 else args.n_nlpcc_ai]:
            t = r["text"].strip()
            docs.append({"doc": t[:30], "y": y, "src": f"nlpcc-{src}-{'human' if y == 0 else model}",
                         "split": split_of(t[:60]), "segs": [(t, len(t))]})
    for t in ev.read_blocks(ev.DATA_DIR / "ai_zh_essays.txt"):
        docs.append({"doc": t[:30], "y": 1, "src": "repo-ai-zh-essay", "split": "train", "segs": [(t, len(t))]})
    return [d for d in docs if d["segs"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlpcc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--n-nlpcc-human", type=int, default=1500, help="NLPCC 每个领域取多少篇真人")
    ap.add_argument("--n-nlpcc-ai", type=int, default=500, help="NLPCC 每个领域、每个模型取多少篇 AI")
    ap.add_argument("--max-segs", type=int, default=6, help="每篇长文最多取多少段训练")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--time-budget-min", type=float, default=150)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{dev}", flush=True)
    torch.manual_seed(0)
    rnd = random.Random(0)
    docs = build(args, rnd)
    rows = []
    for d in docs:
        segs = d["segs"] if d["split"] == "test" else rnd.sample(d["segs"], min(len(d["segs"]), args.max_segs))
        for t, n in segs:
            rows.append({"text": t, "y": d["y"], "src": d["src"], "split": d["split"], "doc": d["doc"], "n": n})
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    rnd.shuffle(train)
    dev_set = train[: max(400, len(train) // 15)]
    train = train[len(dev_set):]
    h_tr, a_tr = [r for r in train if r["y"] == 0], [r for r in train if r["y"] == 1]
    if h_tr and a_tr:
        if len(h_tr) < len(a_tr):
            h_tr = (h_tr * math.ceil(len(a_tr) / len(h_tr)))[:len(a_tr)]
        else:
            a_tr = (a_tr * math.ceil(len(h_tr) / len(a_tr)))[:len(h_tr)]
    train = h_tr + a_tr
    rnd.shuffle(train)
    cnt = lambda rs: {s: sum(r["src"] == s for r in rs) for s in sorted({r["src"] for r in rs})}
    print(f"训练 {len(train)}：{json.dumps(cnt(train), ensure_ascii=False)}", flush=True)
    print(f"评估 {len(test)}：{json.dumps(cnt(test), ensure_ascii=False)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, id2label={0: "human", 1: "ai"}, label2id={"human": 0, "ai": 1}).to(dev)

    def batches(data, bs, shuffle):
        idx = list(range(len(data)))
        if shuffle:
            rnd.shuffle(idx)
        for i in range(0, len(idx), bs):
            chunk = [data[j] for j in idx[i:i + bs]]
            enc = tok([r["text"] for r in chunk], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            yield {k: v.to(dev) for k, v in enc.items()}, torch.tensor([r["y"] for r in chunk], device=dev)

    def predict(data):
        model.eval()
        out = []
        with torch.inference_mode(), torch.autocast(dev, enabled=dev == "cuda"):
            for enc, _ in batches(data, 64, False):
                out.extend(torch.softmax(model(**enc).logits.float(), -1)[:, 1].tolist())
        model.train()
        return out

    total = max(1, int(math.ceil(len(train) / args.batch) * args.epochs))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.06), total)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    model.train()
    t0, step, best, best_state = time.time(), 0, -1.0, None
    evals = sorted({int(total * f) for f in (0.25, 0.5, 0.75, 1.0)})
    done = False
    while not done:
        for enc, y in batches(train, args.batch, True):
            with torch.autocast(dev, enabled=dev == "cuda"):
                logits = model(**enc).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y, label_smoothing=args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                el = time.time() - t0
                print(f"  step {step}/{total} loss {loss.item():.4f}  {el/60:.1f} 分钟", flush=True)
            over = (time.time() - t0) / 60 > args.time_budget_min
            if step in evals or step >= total or over:
                p = predict(dev_set)
                a = ev.auroc([q for q, r in zip(p, dev_set) if r["y"] == 1], [q for q, r in zip(p, dev_set) if r["y"] == 0])
                print(f"  开发集 AUROC {a:.4f}（step {step}）", flush=True)
                if a > best:
                    best, best_state = a, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step >= total or over:
                done = True
                break
    model.load_state_dict(best_state)

    pt = predict(test)
    hum = sorted(p for p, r in zip(pt, test) if r["y"] == 0)
    thr = {f: (hum[min(len(hum) - 1, int(len(hum) * (1 - f)))] if hum else 0.5) for f in (0.01, 0.05)}
    groups = {}
    for p, r in zip(pt, test):
        groups.setdefault(r["src"], []).append(p)
    res = {"dev_auroc": round(best, 4),
           "test_auroc": round(ev.auroc([p for p, r in zip(pt, test) if r["y"] == 1], hum), 4),
           "thresholds": {f"fpr_{int(f*100)}pct": round(t, 4) for f, t in thr.items()}}
    for f, t in thr.items():
        k = f"{int(f*100)}pct"
        res[f"segments_at_{k}"] = {s: round(sum(x >= t for x in v) / len(v), 3) for s, v in groups.items()}
    # 整篇：每篇文档（≥ 3 段）各段得分按字数加权的中位数
    per_doc = {}
    for p, r in zip(pt, test):
        per_doc.setdefault((r["src"], r["doc"], r["y"]), []).append((p, r["n"]))
    doc_meds = {}
    for (src, doc, y), v in per_doc.items():
        if len(v) >= 3:
            doc_meds.setdefault(src, []).append(_wmedian(v))
    hmax = max([m for s, v in doc_meds.items() if "human" in s for m in v] or [1.0])
    res["doc_level"] = {"human_doc_max_median": round(hmax, 4),
                        "by_source": {s: {"n": len(v), "min": round(min(v), 3), "median": round(sorted(v)[len(v) // 2], 3),
                                          "max": round(max(v), 3), "above_human_max": round(sum(m > hmax for m in v) / len(v), 3)}
                                      for s, v in doc_meds.items()}}
    # 用户确认过来源的文档（只评估）
    ud = load_jsonl(ev.DATA_DIR / "eval_zh_user_docs.jsonl")
    if ud:
        flat = []
        for r in ud:
            for t, n in segments(r["text"]):
                flat.append({"text": t, "y": r["y"], "doc": r["doc"], "n": n})
        pu = predict(flat) if flat else []
        ur = {}
        for p, r in zip(pu, flat):
            ur.setdefault(f"{r['doc']}（{'AI' if r['y'] else '真人'}）", []).append((p, r["n"]))
        res["user_docs"] = {d: {"segs": [round(p, 2) for p, _ in v], "median": round(_wmedian(v), 3)} for d, v in ur.items()}
    res.update({"n_train": len(train), "n_test": len(test), "base": args.base, "steps": step,
                "minutes": round((time.time() - t0) / 60, 1), "device": dev})
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    print("::notice title=中文第二分类器：逐段（误判率 1%）::" + json.dumps(res["segments_at_1pct"], ensure_ascii=False), flush=True)
    print("::notice title=中文第二分类器：逐段（误判率 5%）::" + json.dumps(res["segments_at_5pct"], ensure_ascii=False), flush=True)
    print("::notice title=中文第二分类器：整篇::" + json.dumps(res["doc_level"], ensure_ascii=False), flush=True)
    if "user_docs" in res:
        print("::notice title=中文第二分类器：用户文档::" + json.dumps(res["user_docs"], ensure_ascii=False)[:4000], flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "training_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    (out / "README.md").write_text("# 中文第二分类器（新一代国产大模型中文）\n\n由 tools/train_chinese.py 微调 "
                                   f"{args.base} 得到，标签 0 = 人写、1 = AI。\n\n```json\n"
                                   + json.dumps(res, ensure_ascii=False, indent=1) + "\n```\n", "utf-8")


if __name__ == "__main__":
    main()
