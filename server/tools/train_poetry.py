"""训练"诗词专用"AI 检测分类器（在 GitHub Actions 的 CPU 上运行，见 .github/workflows/train-poetry.yml）。

依据：ChangAn 论文（ACL 2026，arXiv:2604.10101）的评测中，通用中文检测器在诗词上的 AUROC 只有约 72%，
零样本方法（Fast-DetectGPT 等）接近随机；在 ChangAn 上微调的 RoBERTa 可达约 95%，对没见过的生成模型也有 93%–98%。

做法：以 hfl/chinese-roberta-wwm-ext 为底座，在 ChangAn 的"训练"划分上微调二分类（人写 / AI）。
人写与 AI 各取相同数量；AI 样本在 DeepSeek、豆包（Seed）、GPT-4.1 三个模型和两种生成策略间均衡。
Kimi-K2 与"校准 / 评估"划分完全不参与训练（划分规则见 tools/changan.py）。

用法：python tools/train_poetry.py --changan <ChangAn 目录> --out <输出目录>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import changan  # noqa: E402


def auroc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # 秩和法（处理并列）
    ranks, i = {}, 0
    rank_sum = 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        rank_sum += sum(r for k in range(i, j + 1) if allv[k][1] == 1)
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--changan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-train-per-class", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--time-budget-min", type=float, default=240)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.set_num_threads(int(os.getenv("TORCH_THREADS", os.cpu_count() or 2)))
    torch.manual_seed(0)
    rnd = random.Random(0)

    rows = changan.load(args.changan)
    train_h = [r for r in rows if r["split"] == "train" and r["y"] == 0]
    train_a = [r for r in rows if r["split"] == "train" and r["y"] == 1]
    rnd.shuffle(train_h); rnd.shuffle(train_a)
    n = min(len(train_h), len(train_a), args.max_train_per_class)
    by_model = {}
    for r in train_a:
        by_model.setdefault(r["model"], []).append(r)
    per = n // len(by_model)
    train_a = [r for m in sorted(by_model) for r in by_model[m][:per]]
    train = train_h[:n] + train_a
    rnd.shuffle(train)
    dev = train[: min(max(200, len(train) // 20), len(train) // 5)]
    train = train[len(dev):]
    test = [r for r in rows if r["split"] == "test"]
    rnd.shuffle(test)
    test_h = [r for r in test if r["y"] == 0][: args.n_test // 2]
    test_a = [r for r in test if r["y"] == 1][: args.n_test // 2]
    print(f"训练 {len(train)}（人写 {sum(r['y'] == 0 for r in train)} / AI {sum(r['y'] == 1 for r in train)}），"
          f"开发 {len(dev)}，评估 {len(test_h)} 人写 / {len(test_a)} AI", flush=True)

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
            for enc, _ in batches(data, 64, False):
                out.extend(torch.softmax(model(**enc).logits.float(), -1)[:, 1].tolist())
        model.train()
        return out

    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = max(1, int(steps_per_epoch * args.epochs))
    if not train:
        raise SystemExit("没有训练样本")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.06), total)
    model.train()
    t0 = time.time()
    step, best, best_state = 0, -1.0, None
    evals = sorted({int(total * f) for f in (0.5, 0.75, 1.0)})
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
            over_budget = (time.time() - t0) / 60 > args.time_budget_min
            if step in evals or step >= total or over_budget:
                p = predict(dev)
                a = auroc([q for q, r in zip(p, dev) if r["y"] == 1], [q for q, r in zip(p, dev) if r["y"] == 0])
                print(f"  开发集 AUROC {a:.4f}（step {step}）", flush=True)
                if a > best:
                    best = a
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if step >= total or over_budget:
                if over_budget:
                    print(f"::warning title=诗词分类器::超出时间预算，在第 {step} 步停止", flush=True)
                done = True
                break

    model.load_state_dict(best_state)
    ph = predict(test_h)
    pa = predict(test_a)
    by = {}
    for p, r in zip(pa, test_a):
        by.setdefault(r["model"], []).append(p)
    res = {"dev_auroc": round(best, 4), "test_auroc": round(auroc(pa, ph), 4),
           "test_auroc_by_model": {m: round(auroc(v, ph), 4) for m, v in by.items()},
           "n_train": len(train), "n_test_human": len(ph), "n_test_ai": len(pa),
           "base": args.base, "epochs": args.epochs, "lr": args.lr, "steps": step,
           "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    print("::notice title=诗词分类器评估（没见过的作者与模型）::" + json.dumps(res, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "training_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    (out / "README.md").write_text(
        "# 诗词 AI 检测分类器\n\n由 tools/train_poetry.py 在 ChangAn（ACL 2026，MIT）上微调 "
        f"{args.base} 得到，标签 0 = 人写、1 = AI。\n\n评估结果：\n\n```json\n"
        + json.dumps(res, ensure_ascii=False, indent=1) + "\n```\n", "utf-8")


if __name__ == "__main__":
    main()
