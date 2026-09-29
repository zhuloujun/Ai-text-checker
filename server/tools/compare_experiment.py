"""把两次评估（现用模型 vs 实验模型）的结果并排列成表。
用法：python tools/compare_experiment.py <基准 eval_result.json> <实验 eval_result.json> <实验名>
"""
import json
import sys

NAMES = {"zh": "现代汉语", "zh_short": "现代汉语短段", "en": "英文", "zh_classical": "文言", "zh_poetry": "诗词"}


def fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def main():
    base = json.load(open(sys.argv[1], encoding="utf-8"))
    exp = json.load(open(sys.argv[2], encoding="utf-8"))
    tag = sys.argv[3] if len(sys.argv) > 3 else "实验"
    lines = [f"# 打分模型对比：现用（Qwen2.5-0.5B）vs {tag}", "",
             "AUROC 越高越好（1 = 完美区分，0.5 = 随机）；检出率越高越好；误判率越低越好。所有数字都在训练 / 校准时没见过的数据上算出。", "",
             "| 文体 | 评估集 | AUROC（现用 → 实验） | 检出率（现用 → 实验） | 误判率（现用 → 实验） |",
             "|---|---|---|---|---|"]
    for prof, name in NAMES.items():
        b = {e["name"]: e for e in (base.get(prof) or {}).get("evaluation", [])}
        x = {e["name"]: e for e in (exp.get(prof) or {}).get("evaluation", [])}
        for ev in list(dict.fromkeys(list(b) + list(x))):
            eb, ex = b.get(ev, {}), x.get(ev, {})
            lines.append(f"| {name} | {ev} | {fmt(eb.get('auroc'))} → **{fmt(ex.get('auroc'))}** | "
                         f"{fmt(eb.get('ai_caught'), True)} → **{fmt(ex.get('ai_caught'), True)}** | "
                         f"{fmt(eb.get('human_flagged'), True)} → **{fmt(ex.get('human_flagged'), True)}** |")
    for label, data in (("现用", base), (tag, exp)):
        vt = ((data.get("zh_poetry") or {}).get("calibration_report") or {}).get("variant_table")
        if vt:
            lines += ["", f"诗词各特征组合（{label}）：", ""]
            for k, v in vt.items():
                lines.append(f"- {k}：" + "，".join(f"{c} {a}" for c, a in v.items()))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
