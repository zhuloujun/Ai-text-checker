"""中文"整篇判断"的验证数据：真人长文（维基、知乎长回答、BBC 中文新闻）+ 国产大模型写的中文论文，
逐篇送到线上服务，记录每段的 MPU 分类器分数、校准后概率等，写到 tools/data/zh_doc_check.json（由工作流提交回仓库，
再离线决定整篇判断的阈值：真人长文几乎不能触发，AI 论文大多能触发）。

用法（工作流 zh-doc-check.yml 调用）：python tools/zh_doc_check.py --n-human 80 --n-ai 12
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

DATA = Path(__file__).resolve().parent / "data"
BASE = "https://zhuloujun--ai-text-checker-web.modal.run"
HUMAN_SOURCES = [  # (数据集, 配置, 文本字段, 名称)——均为 ChatGPT 之前的真人中文
    ("pleisto/wikipedia-cn-20230720-filtered", None, "completion", "维基百科"),
    ("wangrui6/Zhihu-KOL", None, "RESPONSE", "知乎长回答"),
    ("csebuetnlp/xlsum", "chinese_simplified", "text", "BBC中文新闻"),
]
ZH_TOPICS = ["中国古典诗歌的意境与审美", "宋词中的离愁书写", "唐代科举制度对社会流动的影响", "儒家修身思想的当代价值",
             "乡村振兴背景下的农村电商发展", "大棚蔬菜种植增产技术", "人工智能对就业结构的影响", "短视频对青少年阅读习惯的影响",
             "城市垃圾分类的推进困境与对策", "红色文化在思想政治教育中的作用", "家庭教育与儿童自主学习能力培养",
             "新能源汽车产业的发展现状", "中医药现代化面临的挑战", "鲁迅小说中的国民性批判", "《红楼梦》的人物塑造艺术",
             "老龄化社会的养老服务体系", "高校课程思政建设路径", "碳达峰目标下的企业转型", "人类记忆能力的极限",
             "数字经济对传统零售业的冲击", "茶文化与中国人的生活美学", "古代书院教育的特点与启示"]
ZH_PROMPTS = ["帮我写一篇关于{t}的中文短篇论文。格式要严格按照论文格式。",
              "请写一篇题为《{t}》的学术论文，包括标题、摘要、关键词、引言、正文若干部分、结论和参考文献，约 1500 字。",
              "以“{t}”为题写一篇小论文，要有摘要和关键词，正文分几个部分论述，适当引用古今文献，最后列出参考文献。"]


def hf_rows(dataset, config, field, want, rnd):
    q = urllib.parse.quote(dataset, safe="")
    if not config:
        with urllib.request.urlopen(f"https://datasets-server.huggingface.co/splits?dataset={q}", timeout=60) as r:
            sp = json.loads(r.read())["splits"]
        config = sp[0]["config"]
        split = next((s["split"] for s in sp if s["config"] == config and s["split"] == "train"), sp[0]["split"])
    else:
        split = "train"
    out = []
    for off in rnd.sample(range(0, 20000, 100), 40):
        url = (f"https://datasets-server.huggingface.co/rows?dataset={q}&config={urllib.parse.quote(config)}"
               f"&split={split}&offset={off}&length=100")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                rows = json.loads(r.read())["rows"]
        except Exception as e:  # noqa: BLE001
            print(f"::warning title={dataset}::{e}", flush=True)
            time.sleep(3)
            continue
        for row in rows:
            t = (row["row"].get(field) or "").strip()
            cjk = len(re.findall(r"[一-鿿]", t))
            if 1000 <= len(t) and cjk >= 0.6 * len(t):
                if len(t) > 3500:                       # 在段落边界截断
                    cut = t.rfind("\n", 0, 3500)
                    t = t[:cut if cut > 1500 else 3500]
                out.append(t)
        if len(out) >= want:
            break
    return out[:want]


def gen_ai(n_per, rnd):
    docs, lock = [], threading.Lock()

    def run(name):
        env, eps = g.PROVIDERS[name]
        key = os.environ.get(env)
        if not key:
            return
        eps = g.working_endpoints(name, key, eps, limit=2)
        if not eps:
            return
        topics = rnd.sample(ZH_TOPICS, min(n_per, len(ZH_TOPICS)))
        for i, t in enumerate(topics):
            url, model = eps[i % len(eps)]
            text = g.chat(url, key, model, ZH_PROMPTS[i % len(ZH_PROMPTS)].format(t=t), 0.8, 3500)
            if text and len(text) > 600:
                text = text.replace("**", "").replace("#", "")
                with lock:
                    docs.append({"doc": f"{name}-{model}：{t}", "y": 1, "source": f"AI-{name}", "text": text})
        print(f"::notice title={name}::生成中文论文 {sum(d['source'] == 'AI-' + name for d in docs)} 篇", flush=True)

    ths = [threading.Thread(target=run, args=(n,)) for n in ("deepseek", "kimi", "wenxin")]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return docs


def call(path, body=None, headers=None):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def detect(text, tok):
    j = call("/v1/detect", {"text": text, "wait": True}, {"Authorization": "Bearer " + tok})
    while j.get("status") not in ("done", "error"):
        time.sleep(4)
        j = call(f"/v1/jobs/{j['job_id']}", headers={"Authorization": "Bearer " + tok})
    res = j.get("result") or {}
    return {"ai_rate": (res.get("summary") or {}).get("ai_rate"),
            "segments": [{"block": s.get("block"), "register": s.get("register"), "chars": s.get("chars"),
                          "counted": s.get("kind") == "body", "kind": s.get("kind"), "prob": s.get("prob"),
                          "level": s.get("level"), "threshold": s.get("threshold"),
                          "classifier": (s.get("raw") or {}).get("classifier"), "raw": s.get("raw"), "head": (s.get("text") or "")[:30]}
                         for s in res.get("segments", [])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-human", type=int, default=80)
    ap.add_argument("--n-ai", type=int, default=12)
    a = ap.parse_args()
    rnd = random.Random(7)
    docs = [dict(json.loads(l), source="用户文档") for l in open(DATA / "eval_zh_user_docs.jsonl", encoding="utf-8") if l.strip()]
    for ds, cfg, field, name in HUMAN_SOURCES:
        try:
            rows = hf_rows(ds, cfg, field, a.n_human, rnd)
        except Exception as e:  # noqa: BLE001
            print(f"::warning title={name} 下载失败::{e}", flush=True)
            continue
        print(f"::notice title={name}::{len(rows)} 篇", flush=True)
        docs += [{"doc": f"{name}-{i}", "y": 0, "source": name, "text": t} for i, t in enumerate(rows)]
    ai = gen_ai(a.n_ai, rnd)
    out_dir = DATA / "gen_zh"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "zh_papers.jsonl").open("a", encoding="utf-8") as fh:
        for d in ai:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    docs += ai
    key = call("/admin/api/keys", {"name": "zh-doc-check", "days": 1}, {"X-Admin-Token": os.environ["ADMIN_TOKEN"]})
    tok = key.get("key") or key.get("api_key") or key.get("token")
    results = []
    for i, d in enumerate(docs):
        try:
            results.append({**{k: d[k] for k in ("doc", "y", "source")}, "chars": len(d["text"]), **detect(d["text"], tok)})
        except Exception as e:  # noqa: BLE001
            print(f"::warning title=检测失败::{d['doc']} {e}", flush=True)
        if i % 20 == 0:
            print(f"已检测 {i + 1}/{len(docs)}", flush=True)
    (DATA / "zh_doc_check.json").write_text(json.dumps(results, ensure_ascii=False, indent=0), encoding="utf-8")
    by = {}
    for r in results:
        by.setdefault(r["source"], []).append(r["ai_rate"] or 0)
    print("::notice title=各来源平均 AI 率::" + json.dumps({k: [len(v), round(sum(v) / len(v), 3)] for k, v in by.items()},
                                                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
