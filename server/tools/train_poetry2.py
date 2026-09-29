"""训练"诗词专用"AI 检测分类器 v2（在 GitHub Actions 的 CPU 上运行，见 .github/workflows/train-poetry-v2.yml）。

v2 相对 v1（train_poetry.py）：人写训练样本加入唐宋古人作品，AI 训练样本加入 DeepSeek / Kimi / 文心一言写的诗词；
评估额外报告《唐诗三百首》《宋词三百首》的误判、国产新模型诗词与"复述故事"诗词的检出（这些都不参与训练）。

以下为 v1 说明：

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
import evaluate as ev  # noqa: E402


def load_classic(cpoetry_dir, n, rnd):
    """全唐诗 / 全宋词里随机抽 n 首作人写训练样本；《唐诗三百首》《宋词三百首》只留作评估，不参与训练。"""
    import glob
    from opencc import OpenCC
    cc = OpenCC("t2s")
    d = Path(cpoetry_dir)
    norm = lambda t: "".join(ch for ch in t if "\u4e00" <= ch <= "\u9fff")
    tang300 = [cc.convert("\n".join(p["paragraphs"])) for p in json.loads((d / "全唐诗" / "唐诗三百首.json").read_text("utf-8"))]
    song300 = ["\n".join(p["paragraphs"]) for p in json.loads((d / "宋词" / "宋词三百首.json").read_text("utf-8"))]
    evals = [{"text": t, "y": 0, "model": "唐宋三百首"} for t in tang300 + song300 if len(t) >= 16]
    held = {norm(r["text"])[:24] for r in evals}
    pool = []
    for f in sorted(glob.glob(str(d / "全唐诗" / "poet.tang.*.json"))):
        pool += [cc.convert("\n".join(p.get("paragraphs") or [])) for p in json.loads(Path(f).read_text("utf-8"))]
    for f in sorted(glob.glob(str(d / "宋词" / "ci.song.*.json"))):
        pool += ["\n".join(p.get("paragraphs") or []) for p in json.loads(Path(f).read_text("utf-8"))]
    pool = [t for t in pool if 16 <= len(t) <= 240 and norm(t)[:24] not in held]
    rnd.shuffle(pool)
    return [{"text": t, "y": 0, "model": "tang-song"} for t in pool[:n]], evals


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
    ap.add_argument("--cpoetry", required=True, help="chinese-poetry 仓库目录（全唐诗、宋词）")
    ap.add_argument("--n-classic", type=int, default=2500)
    ap.add_argument("--user-share", type=float, default=0.15, help="国产新模型样本（过采样后）占 AI 训练样本的比例")
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
    train_h = train_h[:n]
    # v2 新增 1：唐宋古人作品作为人写训练样本（不含评估用的《唐诗三百首》《宋词三百首》），
    #          纠正 v1 把古典名篇当成 AI 的问题（v1 在唐宋名篇上误判约 19%）
    classic_train, classic_eval = load_classic(args.cpoetry, args.n_classic, rnd)
    # v2 新增 2：国产新模型（DeepSeek / Kimi / 文心一言）写的诗词，每个模型 2/3 参与训练（过采样），1/3 留作评估
    user = [dict(r, y=1) for r in ev.user_ai_rows("poem")]
    user_train = [r for r in user if r["split"] == "fit"]
    user_test = [r for r in user if r["split"] == "test"]
    base = train_h + classic_train + train_a + user_train
    rnd.shuffle(base)
    dev = base[: min(max(200, len(base) // 20), len(base) // 5)]
    train = base[len(dev):]
    # 国产新模型样本少：过采样到约占 AI 训练样本的 user_share（开发集里的不再复制，避免泄漏）
    ut = [r for r in train if r.get("split") == "fit" and r["y"] == 1 and r.get("model", "").startswith("repo-ai-")]
    n_ai = sum(r["y"] == 1 for r in train)
    reps = max(1, round(args.user_share * n_ai / max(1, len(ut))))
    train += ut * (reps - 1)
    n_h = sum(r["y"] == 0 for r in train)
    n_a = sum(r["y"] == 1 for r in train)
    if n_h > n_a:   # 人写多了唐宋古人作品，AI 这边补齐，保持两类平衡
        pool_a = [r for r in train if r["y"] == 1 and not r.get("model", "").startswith("repo-ai-")]
        train += rnd.sample(pool_a, min(len(pool_a), n_h - n_a))
    rnd.shuffle(train)
    test = [r for r in rows if r["split"] == "test"]
    rnd.shuffle(test)
    test_h = [r for r in test if r["y"] == 0][: args.n_test // 2]
    test_a = [r for r in test if r["y"] == 1][: args.n_test // 2]
    story = [{"text": t, "y": 1, "model": "story"} for t in ev.read_blocks(ev.DATA_DIR / "ai_poems_story.txt")]
    print(f"训练 {len(train)}（ChangAn 人写 {len(train_h)} + 唐宋古人 {len(classic_train)} / ChangAn AI {len(train_a)} + "
          f"国产新模型 {len(ut)}×{reps}），开发 {len(dev)}；评估：ChangAn {len(test_h)} 人写 / {len(test_a)} AI，"
          f"唐宋三百首 {len(classic_eval)}，国产新模型 {len(user_test)}，故事诗 {len(story)}", flush=True)

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
    pc = predict(classic_eval)
    pu = predict(user_test)
    ps = predict(story)
    by = {}
    for p, r in zip(pa, test_a):
        by.setdefault(r["model"], []).append(p)
    thr = sorted(ph)[int(len(ph) * 0.95)]          # ChangAn 人写误判约 5% 的参考阈值
    rate = lambda v: round(sum(x >= thr for x in v) / len(v), 3) if v else None
    res = {"dev_auroc": round(best, 4), "test_auroc": round(auroc(pa, ph), 4),
           "test_auroc_by_model": {m: round(auroc(v, ph), 4) for m, v in by.items()},
           "changan_ai_vs_tang_song_300_auroc": round(auroc(pa, pc), 4),
           "user_models_vs_changan_human_auroc": round(auroc(pu, ph), 4) if pu else None,
           "user_models_vs_tang_song_300_auroc": round(auroc(pu, pc), 4) if pu else None,
           "story_poems_vs_tang_song_300_auroc": round(auroc(ps, pc), 4),
           "reference_threshold": round(thr, 4),
           "at_threshold": {"changan_ai_caught": rate(pa), "user_models_caught": rate(pu), "story_caught": rate(ps),
                            "tang_song_300_flagged": rate(pc)},
           "n_train": len(train), "n_test_human": len(ph), "n_test_ai": len(pa),
           "base": args.base, "epochs": args.epochs, "lr": args.lr, "steps": step,
           "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    print("::notice title=诗词分类器 v2 评估（没见过的作者、名篇与模型）::" + json.dumps(res, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "training_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    (out / "README.md").write_text(
        "# 诗词 AI 检测分类器 v2\n\n由 tools/train_poetry2.py 在 ChangAn（ACL 2026，MIT）上微调 "
        f"{args.base} 得到，标签 0 = 人写、1 = AI。\n\n评估结果：\n\n```json\n"
        + json.dumps(res, ensure_ascii=False, indent=1) + "\n```\n", "utf-8")


if __name__ == "__main__":
    main()
