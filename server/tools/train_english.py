"""训练"英文第二分类器"：专门识别国产大模型（DeepSeek / Kimi / 文心一言 / 通义千问）写的英文，
与现有 desklib 英文分类器（主要见过 GPT、LLaMA 等国外模型）互补。在 GitHub Actions 的 CPU 上运行，见 .github/workflows/train-english.yml。

数据（按"论文标题"划分训练 / 评估，评估题目的真人版和 AI 版都从不参与训练）：
  人写：arXiv 2015–2021 年摘要（tools/data/gen_en/arxiv_human.jsonl）+ MAGE valid.csv 的人写文本（新闻、评论、故事等多领域）
  AI：  tools/data/gen_en/{deepseek,kimi,wenxin,qwen}.jsonl（同一批标题让各家模型写摘要 / 引言 / 结果 / 结论）
        + MAGE valid.csv 里较强模型的文本 + 用户提供的国产模型英文短篇（2/3，另外 1/3 只评估）
  MAGE 的 test / ood 文件留给 evaluate.py 做校准与评估，这里一概不用。

用法：python tools/train_english.py --mage-dir <MAGE 目录> --out <输出目录>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

GEN_DIR = ev.DATA_DIR / "gen_en"


def split_of(key: str) -> str:
    return "test" if int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 5 == 0 else "train"


def window(text: str, rnd: random.Random) -> str:
    """与线上一致：英文按 1000–1500 字符的窗口检测；长文随机取一个窗口（在句末截断）。"""
    text = " ".join(text.split())
    if len(text) <= 1500:
        return text
    start = rnd.randint(0, len(text) - 1200)
    dot = text.find(". ", start)
    start = dot + 2 if 0 <= dot < start + 200 else start
    chunk = text[start:start + 1500]
    end = chunk.rfind(". ")
    return chunk[:end + 1] if end > 900 else chunk


def load_jsonl(f: Path) -> list[dict]:
    return [json.loads(l) for l in f.read_text("utf-8").splitlines() if l.strip()] if f.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mage-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="FacebookAI/roberta-base")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--n-mage", type=int, default=1200, help="MAGE 人写 / AI 各取多少")
    ap.add_argument("--time-budget-min", type=float, default=270)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from app.segmenter import normalize_english

    torch.set_num_threads(int(os.getenv("TORCH_THREADS", os.cpu_count() or 2)))
    torch.manual_seed(0)
    rnd = random.Random(0)
    rows = []   # {text, y, src, split}
    for h in load_jsonl(GEN_DIR / "arxiv_human.jsonl"):
        rows.append({"text": h["text"], "y": 0, "src": "arxiv-human", "split": split_of(h["title"])})
    gen_models = []
    for f in sorted(GEN_DIR.glob("*.jsonl")):
        if f.stem == "arxiv_human":
            continue
        gen_models.append(f.stem)
        for g in load_jsonl(f):
            rows.append({"text": g["text"], "y": 1, "src": f"gen-{f.stem}", "split": split_of(g["title"])})
    if not gen_models:
        raise SystemExit("tools/data/gen_en 里还没有生成数据，请先运行“生成英文 AI 训练数据”工作流")
    mage = ev.read_mage(Path(args.mage_dir) / "valid.csv")
    mh = [r for r in mage if r["y"] == 0]
    ma = [r for r in mage if r["y"] == 1 and ev.EN_STRONG.search(r["model"])] or [r for r in mage if r["y"] == 1]
    for r in ev.balanced_sample(mh, args.n_mage, lambda r: r["domain"], 41):
        rows.append({"text": r["text"], "y": 0, "src": f"mage-human-{r['domain']}", "split": split_of(r["text"][:200])})
    for r in ev.balanced_sample(ma, args.n_mage, lambda r: r["domain"], 42):
        rows.append({"text": r["text"], "y": 1, "src": "mage-ai", "split": split_of(r["text"][:200])})
    for r in ev.user_ai_rows("english"):
        rows.append({"text": r["text"], "y": 1, "src": r["model"], "split": "test" if r["split"] == "test" else "train"})
    for t in ev.read_blocks(ev.DATA_DIR / "ai_english_extra.txt"):
        rows.append({"text": t, "y": 1, "src": "repo-ai-english", "split": "train"})
    for r in rows:
        r["text"] = normalize_english(window(r["text"], rnd))
    rows = [r for r in rows if len(r["text"]) >= 300]

    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    rnd.shuffle(train)
    dev = train[: max(150, len(train) // 12)]
    train = train[len(dev):]
    # 两类平衡：少的一类过采样
    h_tr = [r for r in train if r["y"] == 0]
    a_tr = [r for r in train if r["y"] == 1]
    if len(h_tr) < len(a_tr):
        h_tr = h_tr * math.ceil(len(a_tr) / len(h_tr))
        h_tr = h_tr[:len(a_tr)]
    elif len(a_tr) < len(h_tr):
        a_tr = (a_tr * math.ceil(len(h_tr) / len(a_tr)))[:len(h_tr)]
    train = h_tr + a_tr
    rnd.shuffle(train)
    by = lambda rs: {s: sum(r["src"] == s for r in rs) for s in sorted({r["src"] for r in rs})}
    print(f"训练 {len(train)}：{json.dumps(by(train), ensure_ascii=False)}", flush=True)
    print(f"评估 {len(test)}：{json.dumps(by(test), ensure_ascii=False)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, id2label={0: "human", 1: "ai"}, label2id={"human": 0, "ai": 1})

    def batches(data, bs, shuffle):
        idx = list(range(len(data)))
        if shuffle:
            rnd.shuffle(idx)
        for i in range(0, len(idx), bs):
            chunk = [data[j] for j in idx[i:i + bs]]
            enc = tok([r["text"] for r in chunk], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            yield enc, torch.tensor([r["y"] for r in chunk])

    def predict(data):
        model.eval()
        out = []
        with torch.inference_mode():
            for enc, _ in batches(data, 32, False):
                out.extend(torch.softmax(model(**enc).logits.float(), -1)[:, 1].tolist())
        model.train()
        return out

    total = max(1, int(math.ceil(len(train) / args.batch) * args.epochs))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.06), total)
    model.train()
    t0 = time.time()
    step, best, best_state = 0, -1.0, None
    evals = sorted({int(total * f) for f in (0.34, 0.67, 1.0)})
    done = False
    while not done:
        for enc, y in batches(train, args.batch, True):
            loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"  step {step}/{total} loss {loss.item():.4f}  {el/60:.1f} 分钟（剩余约 {el/step*(total-step)/60:.0f} 分钟）", flush=True)
            over = (time.time() - t0) / 60 > args.time_budget_min
            if step in evals or step >= total or over:
                p = predict(dev)
                a = ev.auroc([q for q, r in zip(p, dev) if r["y"] == 1], [q for q, r in zip(p, dev) if r["y"] == 0])
                print(f"  开发集 AUROC {a:.4f}（step {step}）", flush=True)
                if a > best:
                    best, best_state = a, {k: v.detach().clone() for k, v in model.state_dict().items()}
            if step >= total or over:
                done = True
                break

    model.load_state_dict(best_state)
    pt = predict(test)
    hum = [p for p, r in zip(pt, test) if r["y"] == 0]
    thr = sorted(hum)[int(len(hum) * 0.95)] if hum else 0.5
    groups = {}
    for p, r in zip(pt, test):
        groups.setdefault(r["src"], []).append(p)
    res = {"dev_auroc": round(best, 4),
           "test_auroc_all": round(ev.auroc([p for p, r in zip(pt, test) if r["y"] == 1], hum), 4),
           "reference_threshold_at_5pct_fpr": round(thr, 4),
           "ai_caught_at_ref_threshold": {s: round(sum(x >= thr for x in v) / len(v), 3)
                                          for s, v in groups.items() if not s.startswith(("arxiv-human", "mage-human"))},
           "human_flagged_at_ref_threshold": {s: round(sum(x >= thr for x in v) / len(v), 3)
                                              for s, v in groups.items() if s.startswith(("arxiv-human", "mage-human"))},
           "auroc_vs_arxiv_human": {s: round(ev.auroc(v, groups.get("arxiv-human", [])), 4)
                                    for s, v in groups.items() if s.startswith("gen-") and groups.get("arxiv-human")},
           "n_train": len(train), "n_test": len(test), "gen_models": gen_models,
           "base": args.base, "epochs": args.epochs, "steps": step, "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    print("::notice title=英文第二分类器评估（没参与训练的题目与样本）::" + json.dumps(
        {k: res[k] for k in ("test_auroc_all", "ai_caught_at_ref_threshold", "human_flagged_at_ref_threshold")},
        ensure_ascii=False), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "training_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    (out / "README.md").write_text("# 英文第二分类器（国产大模型英文）\n\n由 tools/train_english.py 微调 "
                                   f"{args.base} 得到，标签 0 = 人写、1 = AI。\n\n```json\n"
                                   + json.dumps(res, ensure_ascii=False, indent=1) + "\n```\n", "utf-8")


if __name__ == "__main__":
    main()
