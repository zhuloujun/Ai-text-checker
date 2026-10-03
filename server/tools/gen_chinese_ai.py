"""生成"中文第二分类器"的训练数据：
  · AI：DeepSeek / Kimi / 文心 / 千问 / 豆包（有哪家的密钥就用哪家）按普通用户的一句话指令写中文散文、游记、
    回忆、读后感、议论文、小说、演讲稿、学术小论文等 → tools/data/gen_zh/ai_<家>.jsonl
  · 真人：ChatGPT 出现之前的中文长文（知乎长回答、网络小说、中文网页、高考现代文阅读材料）→ tools/data/gen_zh/human_<来源>.jsonl
    （NLPCC 的论文摘要、新闻、点评在训练时直接从 NLPCC 仓库读取）

为什么要真人"文学类"文字：只拿 AI 散文和真人新闻 / 点评训练，模型会学成"文笔好 = AI"，把老舍、余秋雨这样的
真人散文判成 AI。真人叙事、散文类文字必须足量。

用法（工作流 gen-chinese.yml 调用）：python tools/gen_chinese_ai.py --n-titles 120 --time-budget-min 260
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_english_ai as g  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "gen_zh"

PLACES = ["桂林山水", "敦煌", "泰山", "天坛", "西湖", "黄山", "丽江古城", "鼓浪屿", "长城", "故宫", "苏州园林", "洱海",
          "布达拉宫", "张家界", "九寨沟", "平遥古城", "乌镇", "三峡", "呼伦贝尔草原", "大理", "岳阳楼", "滕王阁", "江南小镇",
          "黄河壶口", "西安城墙", "凤凰古城", "婺源", "青海湖", "香格里拉", "武夷山"]
LIFE = ["老农的回忆", "外婆的厨房", "父亲的自行车", "故乡的老井", "母亲的手", "童年的夏天", "雨中的小巷", "老屋", "下棋",
        "海边旧事", "门前的老槐树", "冬夜", "一碗面", "旧书摊", "爷爷的烟斗", "第一次离家", "邻居王奶奶", "菜市场",
        "秋天的稻田", "老街的理发店", "春节回家", "毕业那年", "小城的雨", "村口的小学", "深夜的火车站"]
BOOKS = ["小火狐与星石", "活着", "平凡的世界", "边城", "围城", "骆驼祥子", "朝花夕拾", "红楼梦", "西游记", "老人与海",
         "小王子", "三体", "百年孤独", "城南旧事", "呼兰河传", "简爱", "钢铁是怎样炼成的", "傅雷家书"]
ESSAYS = ["坚持的意义", "读书的价值", "科技改变生活", "传统文化的传承", "诚信", "青春与梦想", "家风", "劳动最光荣",
          "人工智能与就业", "短视频与阅读", "长征精神", "乡村振兴", "环境保护", "团队合作", "失败与成长", "感恩"]
PAPERS = ["中国古典诗歌的审美特质", "人类记忆与逻辑推理的极限", "宋代科举与社会流动", "乡村振兴中的农村电商",
          "大棚蔬菜增产技术", "短视频对青少年的影响", "城市垃圾分类", "红色文化与思政教育", "家庭教育与自主学习",
          "新能源汽车产业发展", "中医药现代化", "老龄化与养老服务", "碳达峰与企业转型", "茶文化与生活美学"]

GENRES = [
    ("prose", PLACES, ["帮我写一篇关于{t}的散文。", "写一篇{t}的游记，要有历史文化底蕴，语言优美。",
                       "以《{t}》为题写一篇文化散文，1500字左右。", "写一篇关于{t}的写景抒情散文。"]),
    ("memoir", LIFE, ["以《{t}》为题写一篇回忆性散文。", "用第一人称写一篇散文《{t}》，要真挚感人。",
                      "帮我写一篇题为“{t}”的记叙文，1200字左右。", "写一篇关于{t}的短篇小说。"]),
    ("review", BOOKS, ["帮我写一篇《{t}》的读后感。", "写一篇《{t}》的读书笔记，谈谈你的感悟。"]),
    ("essay", ESSAYS, ["帮我写一篇关于{t}的议论文。", "以“{t}”为题写一篇演讲稿。", "写一篇关于{t}的800字作文。"]),
    ("paper", PAPERS, ["帮我写一篇关于{t}的中文短篇论文，格式要严格按照论文格式。",
                       "写一篇题为《{t}研究》的学术论文，包括摘要、关键词、引言、正文、结论和参考文献。"]),
]


def clean(t: str) -> str:
    t = re.sub(r"^\s*#+\s*", "", t, flags=re.M).replace("**", "")
    lines = [l for l in t.splitlines() if not re.fullmatch(r"\s*[-=*_|]{3,}\s*", l)]
    return "\n".join(lines).strip()


def load(f: Path):
    return [json.loads(l) for l in f.read_text("utf-8").split("\n") if l.strip()] if f.exists() else []


def run(name, key, endpoints, n, t0, budget, lock):
    eps = g.working_endpoints(name, key, endpoints, limit=3)
    if not eps:
        return
    out_f = OUT / f"ai_{name}.jsonl"
    done = {r["title"] + "|" + r["ver"] + "|" + r["genre"] for r in load(out_f)}
    rnd = random.Random(f"zh-{name}")
    jobs = [(gen, t, tpl) for gen, topics, tpls in GENRES for t in topics for tpl in tpls]
    rnd.shuffle(jobs)
    n_new = 0
    for i, (gen, t, tpl) in enumerate(jobs[: n * 2]):
        if n_new >= n or (time.time() - t0) / 60 > budget:
            break
        url, model = eps[i % len(eps)]
        if f"{t}|{model}|{gen}" in done:
            continue
        text = g.chat(url, key, model, tpl.format(t=t), rnd.choice([0.7, 0.8, 0.95]), 3000)
        if text and len(re.findall(r"[一-鿿]", text)) > 500:
            with lock, out_f.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"title": t, "genre": gen, "model": name, "ver": model, "prompt": tpl,
                                     "text": clean(text)}, ensure_ascii=False) + "\n")
            n_new += 1
    print(f"::notice title={name}（中文）::新生成 {n_new} 篇（模型：{', '.join(m for _, m in eps)}）", flush=True)


# ---------------- 真人长文（ChatGPT 之前）----------------
HUMAN_SOURCES = [  # (数据集, 配置, 文本字段（None = 自动取最长的文字字段）, 名称, 篇数, 最少汉字数)
    ("wangrui6/Zhihu-KOL", None, "RESPONSE", "zhihu", 1200, 600),
    ("wdndev/webnovel-chinese", None, "text", "webnovel", 600, 600),
    # 文学类真人文字（散文、记叙文、小说片段）——缺了它，模型会把名家散文判成 AI
    ("AsakusaRinne/gaokao_bench", "2010-2022_Chinese_Modern_Lit", None, "gaokao", 300, 400),
    ("clue/clue", "c3", "context", "c3", 1500, 250),
    ("Hello-SimpleAI/HC3-Chinese", "all", "human_answers", "hc3", 800, 300),
]


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def _text_of(v):
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join(_text_of(x) for x in v if x)
    if isinstance(v, dict):
        return "\n".join(_text_of(x) for x in v.values() if isinstance(x, (str, list)))
    return ""


def hf_rows(dataset, config, field, want, rnd, min_cjk=600):
    q = urllib.parse.quote(dataset, safe="")
    sp = _get(f"https://datasets-server.huggingface.co/splits?dataset={q}")["splits"]
    if config:
        sp = [s for s in sp if s["config"] == config] or sp
    cfg, split = sp[0]["config"], sp[0]["split"]
    try:
        size = _get(f"https://datasets-server.huggingface.co/size?dataset={q}")["size"]["splits"]
        n_rows = next((s["num_rows"] for s in size if s["config"] == cfg and s["split"] == split), 20000)
    except Exception:  # noqa: BLE001
        n_rows = 20000
    out, seen, probed = [], set(), False
    offsets = list(range(0, max(1, n_rows), 100))
    rnd.shuffle(offsets)
    for off in offsets[:200]:
        try:
            rows = _get(f"https://datasets-server.huggingface.co/rows?dataset={q}&config={urllib.parse.quote(cfg)}"
                        f"&split={urllib.parse.quote(split)}&offset={off}&length=100")["rows"]
        except Exception as e:  # noqa: BLE001
            print(f"  {dataset} offset {off}: {e}", flush=True)
            time.sleep(2)
            continue
        for row in rows:
            r = row["row"]
            if not probed:
                probed = True
                print(f"::notice title=真人中文数据字段 {dataset}::{cfg}/{split} 共 {n_rows} 行；字段："
                      + ", ".join(f"{k}({len(_text_of(v))})" for k, v in r.items()), flush=True)
            if field:
                t = _text_of(r.get(field))
            else:
                t = max((_text_of(v) for v in r.values()), key=len, default="")
            t = t.strip()
            if dataset.endswith("gaokao_bench"):     # 高考阅读题：只取文章，去掉后面的题目
                t = re.split(r"\n\s*\d+\s*[.．、]\s*\S|（\s*\d+\s*分\s*）|[（(]\s*1\s*[）)]", t)[0]
            cjk = len(re.findall(r"[\u4e00-\u9fff]", t))
            if cjk >= min_cjk and cjk >= 0.6 * len(t) and t[:80] not in seen:
                seen.add(t[:80])
                if len(t) > 4000:
                    cut = t.rfind("\n", 0, 4000)
                    t = t[:cut if cut > 1500 else 4000]
                out.append(t)
        if len(out) >= want:
            break
    return out[:want]


def collect_human():
    rnd = random.Random(11)
    for ds, cfg, field, name, want, min_cjk in HUMAN_SOURCES:
        f = OUT / f"human_{name}.jsonl"
        if len(load(f)) >= want * 0.8:
            continue
        try:
            rows = hf_rows(ds, cfg, field, want, rnd, min_cjk)
        except Exception as e:  # noqa: BLE001
            print(f"::warning title=真人中文 {name} 下载失败::{ds} {e}", flush=True)
            continue
        with f.open("w", encoding="utf-8") as fh:
            for i, t in enumerate(rows):
                fh.write(json.dumps({"title": f"{name}-{i}", "source": name, "text": t}, ensure_ascii=False) + "\n")
        print(f"::notice title=真人中文 {name}::{len(rows)} 篇（{ds}）", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-titles", type=int, default=120)
    ap.add_argument("--time-budget-min", type=float, default=260)
    ap.add_argument("--skip-human", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.skip_human or os.getenv("ONLY_PROVIDERS") == "human":
        collect_human()
    if os.getenv("ONLY_PROVIDERS") == "human":
        return
    t0, lock, ths = time.time(), threading.Lock(), []
    for name, (env, eps) in g.PROVIDERS.items():
        if os.getenv('ONLY_PROVIDERS') and name not in os.getenv('ONLY_PROVIDERS').split(','):
            continue
        key = os.getenv(env, "").strip()
        if not key:
            print(f"::notice title=未设置 {env}::跳过 {name}", flush=True)
            continue
        th = threading.Thread(target=run, args=(name, key, eps, a.n_titles, t0, a.time_budget_min, lock), daemon=True)
        th.start()
        ths.append(th)
    for th in ths:
        th.join()


if __name__ == "__main__":
    main()
