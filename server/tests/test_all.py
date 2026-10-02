"""测试。运行：TEST_MODELS_DIR=<含 observer/performer/cls 三个小模型的目录> pytest -q
（小模型可用 tests/make_tiny_models.py 生成；它们是随机权重，只用于检查流程与公式，不代表检测效果。）"""
import os
import sys
import time
from pathlib import Path

import pytest

MD = os.environ.get("TEST_MODELS_DIR")
if not MD:
    pytest.skip("未设置 TEST_MODELS_DIR", allow_module_level=True)

os.environ.update({
    "OBSERVER_MODEL": f"{MD}/observer", "PERFORMER_MODEL": f"{MD}/performer", "CLASSIFIER_MODEL": f"{MD}/cls",
    "EN_CLASSIFIER_MODEL": f"{MD}/desklib_en", "POETRY_CLASSIFIER_MODEL": f"{MD}/desklib_en/../cls",
    "EN2_CLASSIFIER_MODEL": f"{MD}/cls",
    "ADMIN_TOKEN": "test-admin-pw", "LM_MAX_TOKENS": "128", "CALIBRATION_FILE": "/nonexistent/cal.json",
    "MAX_TEXT_CHARS": "300000",
    "USER_CALIBRATION_FILE": f"/tmp/test_user_calibration_{os.getpid()}.json",
    "USER_LABELS_FILE": f"/tmp/test_user_labels_{os.getpid()}.json", "AUTO_CALIBRATE_DELAY": "0.3",
})
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import keys, scoring  # noqa: E402
from app.segmenter import classical_ratio, detect_register, segment_text  # noqa: E402

MODERN = ("宋代的地方行政制度在很大程度上延续了唐末五代的格局，但又有所调整。据《宋史·职官志》记载，路一级机构的设置经历了反复变化，"
          "转运使司的职能也随之扩展。这一时期的文献对州县官员的任免多有记述，然而其中不乏相互矛盾之处。笔者以为，此类记载须与地方志、碑刻互相参证，方能厘清其真实面貌。")
CLASSICAL = ("太祖既受禅，患藩镇之强，乃用赵普之谋，稍夺其权。或曰：其所以然者，盖鉴于唐末五代之乱也。帝曰：善。于是诸镇皆罢，"
             "以文臣知州事，而兵权悉归于上。君子曰：此所谓强干弱枝者也，其虑深矣，岂不然哉。")
ENGLISH = ("The administrative system of the Song dynasty largely continued the structure inherited from the late Tang and "
           "the Five Dynasties, although it was adjusted in several important ways. According to the treatise on offices in "
           "the Song History, the organization of circuit-level agencies changed repeatedly, and the functions of the fiscal "
           "commissioners expanded accordingly. Sources from this period record many appointments of prefectural officials. ")


# ---------------- 公式：与官方实现逐字比对 ----------------

def ref_fast_detect(logits_ref, logits_score, labels):
    """baoguangsheng/fast-detect-gpt: get_sampling_discrepancy_analytic"""
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    return discrepancy.mean().item()


def ref_binoculars(obs_logits, perf_logits, labels):
    """ahans30/Binoculars: ppl = perplexity(encodings, performer_logits); x_ppl = entropy(observer, performer)"""
    ce = torch.nn.functional.cross_entropy(perf_logits, labels, reduction="none")
    ppl = ce.mean().item()
    p_proba = torch.softmax(obs_logits, dim=-1)
    q_scores = perf_logits
    ce2 = (p_proba * -torch.log_softmax(q_scores, dim=-1)).sum(-1)
    x_ppl = ce2.mean().item()
    return ppl / x_ppl


@pytest.fixture(scope="module")
def scorer():
    from app.detectors.lm_scorer import LMScorer
    s = LMScorer()
    s.load()
    return s


def test_formulas_match_official(scorer):
    ids = scorer.tok(MODERN, add_special_tokens=False)["input_ids"][:100]
    inp = torch.tensor([ids])
    with torch.inference_mode():
        o = scorer.observer(input_ids=inp).logits[0, :-1, : scorer.vocab].float()
        p = scorer.performer(input_ids=inp).logits[0, :-1, : scorer.vocab].float()
    labels = inp[0, 1:]
    r = scorer.score_ids(ids)
    assert r["tokens"] == len(ids) - 1
    assert r["fastdetect"] == pytest.approx(ref_fast_detect(o[None], p[None], labels[None]), rel=1e-4)
    assert r["binoculars"] == pytest.approx(ref_binoculars(o, p, labels), rel=1e-4)


def test_extra_features_match_reference(scorer):
    """Log-Rank / LRR（DetectLLM）/ 熵 / GLTR top-k 与独立实现比对。"""
    ids = scorer.tok(MODERN, add_special_tokens=False)["input_ids"][:100]
    inp = torch.tensor([ids])
    with torch.inference_mode():
        p = scorer.performer(input_ids=inp).logits[0, :-1, : scorer.vocab].float()
    labels = inp[0, 1:]
    lp = torch.log_softmax(p, -1)
    ll = lp[torch.arange(len(labels)), labels]
    # 名次：按概率从高到低排序后，实际 token 的位置（从 1 开始）
    order = torch.argsort(lp, dim=-1, descending=True)
    ranks = (order == labels.unsqueeze(-1)).nonzero()[:, 1] + 1
    ref_logrank = torch.log(ranks.float()).mean().item()
    ref_lrr = (-ll.sum() / torch.log(ranks.float()).sum()).item()
    ref_ent = (-(lp.exp() * lp).sum(-1)).mean().item()
    r = scorer.score_ids(ids)
    assert r["log_rank"] == pytest.approx(ref_logrank, rel=1e-4, abs=1e-6)
    assert r["lrr"] == pytest.approx(ref_lrr, rel=1e-4)
    assert r["entropy"] == pytest.approx(ref_ent, rel=1e-4)
    assert r["top1"] == pytest.approx((ranks == 1).float().mean().item())
    assert r["top10"] == pytest.approx((ranks <= 10).float().mean().item())
    assert r["fastdetect_norm"] == pytest.approx(r["fastdetect"] / (len(ids) - 1) ** 0.5)
    assert r["lp_burstiness"] is not None and r["lp_burstiness"] >= 0


def test_smoothing_and_levels():
    probs = [0.1, 0.95, 0.1, None, 0.9, 0.92]
    sm = scoring.smooth(probs, 0.3)
    assert sm[3] is None
    assert sm[1] < 0.95 and sm[0] > 0.1          # 孤立高分被拉低，邻居被略微拉高
    assert scoring.smooth(probs, 0) == probs
    assert scoring.level_of(0.85, 0.5) == ("high", "高度疑似")
    assert scoring.level_of(0.7, 0.5) == ("mid", "中度疑似")
    assert scoring.level_of(0.55, 0.5) == ("light", "轻度疑似")
    assert scoring.level_of(0.45, 0.5)[0] == "low"
    assert scoring.level_of(0.7, 0.75)[0] == "low"
    assert scoring.level_of(0.78, 0.75) == ("mid", "中度疑似")


def test_long_text_is_chunked(scorer):
    r = scorer.score(MODERN * 6)
    assert r and r["tokens"] > 128


# ---------------- 分段 ----------------

def test_classical_ratio_separates():
    assert classical_ratio(CLASSICAL) >= 0.03  # 学术白话也常用"其、而、以"，所以还要看现代汉语标志词
    assert detect_register(MODERN) == "zh"
    assert detect_register(CLASSICAL) == "zh_classical"
    assert detect_register(ENGLISH) == "en"
    assert detect_register(POEM) == "zh_poetry" and detect_register(COUPLET) == "zh_poetry"


def test_classical_document_is_counted_and_modern_quotes_excluded():
    # 通篇文言（文言小说、仿古文）：文言就是正文，对话多也不算引文
    story = "\n\n".join([CLASSICAL, "女曰：“妾本唐时花媪，以杜工部一诗，得窃灵气，岁久成精。郎宜自爱，勿以妾为念也。”生泣而别之。", CLASSICAL])
    segs = segment_text(story)
    assert all(s.kind == "body" and s.register == "zh_classical" for s in segs)
    # 现代汉语论文里夹一段文言：文言段落视为古籍引文，另起一段
    paper = "\n\n".join([MODERN, CLASSICAL, MODERN])
    kinds = [(s.register, s.kind) for s in segment_text(paper)]
    assert ("zh_classical", "quotation") in kinds and kinds.count(("zh", "body")) == 2


POEM = "诗·七律《咏春》\n浣花溪畔废园春，牡丹幻作红衫人。\n杜老诗魂传一脉，陈生痴念结三生。\n花馔夜饮情方炽，道士符飞梦已尘。\n青城别后重相见，溪上呼名泪满巾。"
COUPLET = "对联：\n飞檐斗拱，几回苍烟落照；\n暮鼓晨钟，一枕孤馆秋寒。"
ESSAY_PARAS = ["读完这篇故事，心中久久不能平静。窗外的风掠过枝头，花影摇曳，恍惚间仿佛也看见一位女子立于残垣之间。",
               "这是一个关于情的故事，但它的动人之处，恰恰在于那份情的不可能。她的存在本身，便是诗与花的因缘和合。",
               "而陈生呢？他明知她是异类，却始终无法割舍。这份情，早已超越了色相之惑，成了一种近乎执拗的守护。",
               "就像每年春天，溪畔的花，依旧会开。"]


def test_titles_split_works_and_short_paragraphs_merge():
    doc = "\n\n".join([POEM, COUPLET, "花落花开——读后感"] + ESSAY_PARAS)
    segs = segment_text(doc)
    assert [s.register for s in segs[:2]] == ["zh_poetry", "zh_poetry"]
    assert segs[0].title.startswith("诗·七律") and segs[1].title == "对联："
    assert len({s.block for s in segs}) == 3               # 三篇作品
    essay = [s for s in segs if s.register == "zh"]
    assert len(essay) <= 2 and all(len(s.text) >= 80 for s in essay)   # 短段落合并成窗口
    assert "依旧会开" in essay[-1].text                      # 结尾一句并入上一段，不单独成段
    assert all(s.kind == "body" for s in segs)              # 带标题的诗词是独立作品，照常计入


def test_untitled_poem_inside_modern_paper_is_quotation():
    poem = "浣花溪畔废园春，牡丹幻作红衫人。\n杜老诗魂传一脉，陈生痴念结三生。"
    paper = "\n\n".join([MODERN * 2, poem, MODERN * 2])
    kinds = [(s.register, s.kind) for s in segment_text(paper)]
    assert ("zh_poetry", "quotation") in kinds


def test_english_segments_are_longer_and_split_on_sentences():
    segs = segment_text(ENGLISH * 8)
    assert all(s.register == "en" for s in segs)
    assert all(len(s.text) >= 500 for s in segs)
    assert all(s.text.rstrip().endswith(".") for s in segs)


def test_profiles_and_merge():
    base = {"calibrated": True, "threshold": 0.5, "signals": scoring.DEFAULTS["signals"], "note": "zh"}
    en = dict(base, note="en", threshold=0.7, profile="en")
    merged = scoring.merge_profile(base, en, "en")
    assert merged["note"] == "zh" and merged["profiles"]["en"]["threshold"] == 0.7
    assert scoring.profile_for(merged, "en")[0]["threshold"] == 0.7
    assert scoring.profile_for(merged, "zh")[0]["note"] == "zh"
    # 文言没有专门校准：沿用现代汉语参数，但标记为未校准
    assert scoring.profile_for(merged, "zh_classical") == (merged, False)
    # 英文没有专门校准：退回经验值，不用中文参数
    assert scoring.profile_for(base, "en")[1] is False
    # 替换现代汉语部分时保留其他文体
    again = scoring.merge_profile(merged, dict(base, note="zh2"), "zh")
    assert again["note"] == "zh2" and again["profiles"]["en"]["threshold"] == 0.7


def test_segmenter_excludes_references_and_quotes():
    doc = "\n\n".join([MODERN * 2, CLASSICAL * 2, "“" + MODERN + "”", MODERN, "参考文献",
                       "[1] 脱脱等：《宋史》，北京：中华书局，1977年。", "[2] 李焘：《续资治通鉴长编》，北京：中华书局，2004年。"])
    segs = segment_text(doc)
    kinds = [s.kind for s in segs]
    assert "reference" in kinds and "quotation" in kinds and "body" in kinds
    assert all(s.kind == "reference" for s in segs if "中华书局" in s.text)
    assert sum(len(s.text) for s in segs) >= len(doc.replace("\n", "")) * 0.95


# ---------------- Key ----------------

def test_keys_roundtrip():
    k = keys.issue("测试", days=1, daily_chars=100)
    p = keys.verify(k["key"])
    assert p and p["i"] == k["id"] and p["q"] == 100
    assert keys.verify(k["key"][:-2] + ("AA" if not k["key"].endswith("AA") else "BB")) is None
    assert keys.verify("atc-garbage") is None
    ok, _ = keys.consume(p, 60)
    assert ok
    ok, left = keys.consume(p, 60)
    assert not ok and left == 40
    keys.revoke_runtime(k["id"])
    assert keys.verify(k["key"]) is None


def test_expired_key():
    k = keys.issue("x", days=1)
    import json, base64
    p64 = k["key"][4:].split(".")[0]
    payload = json.loads(base64.urlsafe_b64decode(p64 + "=" * (-len(p64) % 4)))
    payload["e"] = int(time.time()) - 10
    np64 = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=").decode()
    forged = f"atc-{np64}.{keys._sign(np64)}"   # 用正确签名构造一个已过期的 Key
    assert keys.verify(forged) is None


# ---------------- 校准 ----------------

def test_calibration_extended_features():
    import random
    rnd = random.Random(1)
    def mk(ai):
        return {"fastdetect": rnd.gauss(2.8 if ai else 0.6, 0.8), "binoculars": rnd.gauss(0.82 if ai else 0.98, 0.06),
                "classifier": rnd.uniform(0.4, 0.95) if ai else rnd.uniform(0.05, 0.6),
                "fastdetect_norm": rnd.gauss(0.2 if ai else 0.05, 0.05), "lrr": rnd.gauss(1.6 if ai else 1.2, 0.2),
                "log_rank": rnd.gauss(0.8 if ai else 1.4, 0.3), "entropy": rnd.gauss(2.0 if ai else 2.6, 0.4),
                "top10": rnd.gauss(0.85 if ai else 0.7, 0.05), "lp_burstiness": rnd.gauss(0.15 if ai else 0.3, 0.08),
                "style_cv": rnd.gauss(0.3 if ai else 0.6, 0.15), "style_phrases": rnd.gauss(3 if ai else 1, 1)}
    human = [mk(False) for _ in range(40)]
    ai = [mk(True) for _ in range(40)]
    res = scoring.calibrate(human, ai, 0.05)
    rep, cal = res["report"], res["calibration"]
    assert rep["cross_validated"] is True
    assert set(scoring.EXTENDED_FEATURES) == set(cal["lr"]["features"])
    assert rep["combined_auroc"] > 0.95 and rep["human_flagged_rate"] <= 0.05 + 1e-9
    assert "lrr" in rep["auroc"]
    # 旧版校准 JSON（特征名为 logit(classifier)）仍然可用
    old = dict(cal); old["lr"] = dict(cal["lr"], features=["fastdetect", "binoculars", "logit(classifier)"],
                                      w=cal["lr"]["w"][:3], mean=cal["lr"]["mean"][:3], std=cal["lr"]["std"][:3])
    assert 0 <= scoring.combine(ai[0], old)["prob"] <= 1


def test_calibration_separates_synthetic():
    import random
    rnd = random.Random(0)
    human = [{"fastdetect": rnd.gauss(0.5, 0.5), "binoculars": rnd.gauss(1.0, 0.05), "classifier": rnd.uniform(0.05, 0.5)} for _ in range(60)]
    ai = [{"fastdetect": rnd.gauss(3.0, 0.7), "binoculars": rnd.gauss(0.8, 0.05), "classifier": rnd.uniform(0.5, 0.98)} for _ in range(60)]
    res = scoring.calibrate(human, ai, 0.05)
    cal, rep = res["calibration"], res["report"]
    assert cal["calibrated"] and "lr" in cal
    assert rep["combined_auroc"] > 0.95
    assert rep["human_flagged_rate"] <= 0.05 + 1e-9
    assert cal["signals"]["binoculars"]["direction"] == -1 and cal["signals"]["fastdetect"]["direction"] == 1
    assert scoring.combine(ai[0], cal)["prob"] > scoring.combine(human[0], cal)["prob"]


# ---------------- 接口 ----------------

@pytest.fixture(scope="module")
def client():
    from app import main
    c = TestClient(main.app)
    for _ in range(300):
        if not main.engine.loading:
            break
        time.sleep(0.1)
    return c


ADMIN = {"X-Admin-Token": "test-admin-pw"}


def issue(client, **kw):
    r = client.post("/admin/api/keys", json={"name": "t", **kw}, headers=ADMIN)
    assert r.status_code == 200, r.text
    return r.json()["key"]


def poll(client, jid, headers, path="/v1/jobs/"):
    for _ in range(3000):
        j = client.get(path + jid, headers=headers).json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("timeout")


def test_health(client):
    h = client.get("/health").json()
    assert h["lm"]["ready"] and h["classifier"]["ready"] and h["classifier_en"]["ready"], h
    assert h["requires_key"] and h["key_signing_configured"]
    assert h["calibration"]["calibrated"] is False


def test_pages(client):
    assert "审读" in client.get("/").text
    assert "管理员登录" in client.get("/admin").text
    assert client.get("/static/app.js").status_code == 200


def test_auth_errors(client):
    assert client.post("/v1/detect", json={"text": MODERN}).json()["error"] == "missing_key"
    assert client.post("/v1/detect", json={"text": MODERN}, headers={"X-API-Key": "atc-bad.bad"}).json()["error"] == "invalid_key"
    assert client.post("/admin/api/keys", json={}, headers={"X-Admin-Token": "wrong"}).status_code == 401


def test_detect_sync(client):
    h = {"Authorization": "Bearer " + issue(client)}
    r = client.post("/v1/detect", json={"text": MODERN * 3}, headers=h)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "done"
    s = j["result"]["summary"]
    assert s["methods"] == {"fastdetect": True, "binoculars": True, "classifier": True, "classifier_en": True,
                            "classifier_en2": True, "classifier_poetry": True,
                            "classifier_classical": bool(os.environ.get("CLASSICAL_CLASSIFIER_MODEL"))}
    assert 0 <= s["ai_rate"] <= 1 and s["counted_chars"] > 0
    seg = j["result"]["segments"][0]
    assert set(seg["raw"]) >= {"fastdetect", "binoculars", "classifier", "ppl", "lrr", "log_rank", "entropy", "top10", "style_cv"}
    assert set(s["segments_by_level"]) == {"high", "mid", "light", "low"}
    assert "reliability_notes" in s and any("校准" in n for n in s["reliability_notes"])
    assert 0 <= seg["prob"] <= 1


def test_all_quotation_falls_back_and_excluded_get_reference_value(client):
    h = {"Authorization": "Bearer " + issue(client)}
    # 通篇文言：文言就是正文，照常计入（不再被当作引文排除）
    s = client.post("/v1/detect", json={"text": CLASSICAL * 3}, headers=h).json()["result"]
    assert s["summary"]["fallback_all_counted"] is False
    assert s["summary"]["counted_chars"] > 0 and s["summary"]["ai_rate"] is not None
    assert s["summary"]["main_register"] == "zh_classical"
    assert any("文言" in n for n in s["summary"]["reliability_notes"])
    # 只有参考文献：退回为全部计入，报告仍有数值
    refs = "参考文献\n" + "\n".join(f"[{i}] 脱脱等：《宋史》卷{i}，北京：中华书局，1977年，第{i*3}页。" for i in range(1, 12))
    s = client.post("/v1/detect", json={"text": refs}, headers=h).json()["result"]
    assert s["summary"]["fallback_all_counted"] is True
    assert all(seg["prob"] is not None for seg in s["segments"])
    # 正文 + 文言引文：引文不计入，但有参考值
    r = client.post("/v1/detect", json={"text": MODERN * 2 + "\n\n" + CLASSICAL * 2}, headers=h).json()["result"]
    assert r["summary"]["fallback_all_counted"] is False
    q = [seg for seg in r["segments"] if seg["kind"] == "quotation"]
    assert q and all(seg["prob"] is None and seg["ref_prob"] is not None for seg in q)


def test_pages_versioned_and_not_cached(client):
    r = client.get("/")
    assert "/static/app.js?v=" in r.text and "/static/library.js?v=" in r.text
    assert r.headers["cache-control"] == "no-cache"
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"


def test_detect_long_async_fast_mode(client):
    h = {"X-API-Key": issue(client)}
    doc = "\n\n".join(MODERN for _ in range(1200))   # 约 16 万字
    t0 = time.time()
    r = client.post("/v1/detect", json={"text": doc, "mode": "fast"}, headers=h)
    assert r.status_code == 202, r.text
    j = poll(client, r.json()["id"], h)
    assert j["status"] == "done", j
    s = j["result"]["summary"]
    assert s["lm_sampled"] and s["lm_scored_segments"] == 60
    assert s["segments"] > 60
    print(f"\n160k chars fast mode (tiny models): {time.time()-t0:.1f}s")


def test_job_owner_isolation(client):
    h1, h2 = {"X-API-Key": issue(client)}, {"X-API-Key": issue(client)}
    r = client.post("/v1/detect", json={"text": MODERN * 30, "wait": False}, headers=h1)
    jid = r.json()["id"]
    assert client.get("/v1/jobs/" + jid, headers=h2).status_code == 404
    assert poll(client, jid, h1)["status"] == "done"


def test_quota(client):
    h = {"X-API-Key": issue(client, daily_chars=500)}
    assert client.post("/v1/detect", json={"text": MODERN * 2}, headers=h).status_code == 200
    r = client.post("/v1/detect", json={"text": MODERN * 2}, headers=h)
    assert r.status_code == 429 and r.json()["error"] == "quota_exceeded"


def test_file_upload(client, tmp_path):
    import zipfile
    h = {"X-API-Key": issue(client)}
    p = tmp_path / "t.docx"
    body = "".join(f"<w:p><w:r><w:t>{MODERN}</w:t></w:r></w:p>" for _ in range(3))
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
                   + body + "</w:body></w:document>")
    r = client.post("/v1/detect/file", files={"file": ("t.docx", p.read_bytes())}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["result"]["summary"]["total_chars"] >= len(MODERN) * 3
    r = client.post("/v1/detect/file", files={"file": ("t.exe", b"xx")}, headers=h)
    assert r.json()["error"] == "bad_file"


def test_calibrate_endpoint_and_apply(client):
    human = [MODERN * 2] * 8
    ai = [("综上所述，该制度在很大程度上体现了中央集权的发展趋势，具有重要意义。此外，值得注意的是，这一变化对后世产生了深远影响，"
           "为理解宋代政治提供了有力支撑。总的来说，其演变是多种因素共同作用的结果。") * 3] * 8
    r = client.post("/admin/api/calibrate", json={"human": human, "ai": ai}, headers=ADMIN)
    assert r.status_code == 200, r.text
    j = poll(client, r.json()["id"], ADMIN, "/admin/api/jobs/")
    assert j["status"] == "done", j
    cal = j["result"]["calibration"]
    assert cal["calibrated"]
    assert client.post("/admin/api/calibration", json={"calibration": cal}, headers=ADMIN).json()["ok"]
    assert client.get("/health").json()["calibration"]["calibrated"] is True


def test_english_uses_english_classifier(client):
    from app import main
    h = {"Authorization": "Bearer " + issue(client)}
    r = client.post("/v1/detect", json={"text": ENGLISH * 6 + "\n\n" + MODERN * 3, "wait": True}, headers=h).json()["result"]
    regs = {seg["register"] for seg in r["segments"]}
    assert regs == {"en", "zh"}, regs
    en_seg = next(seg for seg in r["segments"] if seg["register"] == "en")
    zh_seg = next(seg for seg in r["segments"] if seg["register"] == "zh")
    # 英文段落的分类器分数来自英文分类器，与中文分类器给同一段的分数不同
    assert en_seg["raw"]["classifier"] == round(main.engine.cls_en.predict([en_seg["text"]])[0], 4)
    assert zh_seg["raw"]["classifier"] == round(main.engine.cls.predict([zh_seg["text"]])[0], 4)
    assert set(r["summary"]["chars_by_register"]) == {"en", "zh"}
    assert any("多种文体" in n for n in r["summary"]["reliability_notes"])


def test_calibrate_one_profile_keeps_others(client):
    human = [ENGLISH * 4] * 6
    ai = [("Furthermore, it is worth noting that the administrative landscape of the Song dynasty serves as a testament "
           "to the intricate interplay of central authority and local governance. Moreover, this multifaceted system "
           "played a pivotal role in fostering stability. ") * 6] * 6
    r = client.post("/admin/api/calibrate", json={"human": human, "ai": ai, "profile": "auto"}, headers=ADMIN)
    j = poll(client, r.json()["id"], ADMIN, "/admin/api/jobs/")
    assert j["status"] == "done", j
    cal = j["result"]["calibration"]
    assert cal["profile"] == "en" and j["result"]["report"]["profile_name"] == "英文"
    d = client.post("/admin/api/calibration", json={"calibration": cal}, headers=ADMIN).json()
    assert d["calibration"]["profiles"]["en"]["calibrated"]
    h = client.get("/health").json()["calibration"]["profiles"]
    assert h["en"] is True


def test_poetry_uses_poetry_classifier(client):
    from app import main
    assert main.engine.cls_poetry is not None and main.engine.cls_poetry.ready
    h = {"Authorization": "Bearer " + issue(client)}
    r = client.post("/v1/detect", json={"text": POEM + "\n\n" + MODERN * 3, "wait": True}, headers=h).json()["result"]
    poem = next(seg for seg in r["segments"] if seg["register"] == "zh_poetry")
    body = poem["text"][len(poem["title"]):].strip()             # 打分时不含标题
    assert poem["raw"]["classifier"] == round(main.engine.cls_poetry.predict([body])[0], 4)
    assert r["summary"]["methods"]["classifier_poetry"] is True
    assert client.get("/health").json()["classifier_poetry"]["ready"] is True


def test_download_models_tarball(tmp_path):
    import io, sys, tarfile
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import download_models  # noqa: F401 —— 导入时不会下载任何东西（没有参数）
    src = tmp_path / "poetry-classifier"
    src.mkdir()
    (src / "config.json").write_text("{}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(src, arcname="poetry-classifier")
    tar = tmp_path / "m.tar.gz"
    tar.write_bytes(buf.getvalue())
    out = download_models.fetch_tarball(tar.as_uri(), str(tmp_path / "dest" / "model"))
    assert (out / "config.json").exists()


def test_memorized_guard_and_bare_title():
    import math
    from app.engine import memorized
    assert memorized({"ppl": math.log(1.9), "classifier": 0.005})          # 名篇：困惑度极低、分类器判人写
    assert not memorized({"ppl": math.log(1.9), "classifier": 0.9})        # 分类器也判 AI：不算名篇
    assert not memorized({"ppl": math.log(20), "classifier": 0.01})
    # 不带标点的短标题行把文言小说与后面的作品分开；独立的文言作品照常计入
    doc = "\n\n".join(["黄四娘", CLASSICAL, CLASSICAL, "诗·七律《咏春》", POEM.split("\n", 1)[1], "读后感", MODERN * 3])
    segs = segment_text(doc)
    story = [s for s in segs if s.register == "zh_classical"]
    assert story and story[0].title == "黄四娘" and all(s.kind == "body" for s in story)


def test_calibrate_with_user_labels_merged_with_builtin(client, monkeypatch):
    import random
    from app import engine as eng
    rnd = random.Random(1)
    fake = ([{"y": 0, "s": {"fastdetect": rnd.gauss(0, 1), "binoculars": rnd.gauss(1.0, .05), "classifier": rnd.uniform(.01, .4)}} for _ in range(80)]
            + [{"y": 1, "s": {"fastdetect": rnd.gauss(2.5, 1), "binoculars": rnd.gauss(.85, .05), "classifier": rnd.uniform(.6, .99)}} for _ in range(80)])
    monkeypatch.setattr(eng, "load_builtin_calib", lambda profile: fake)
    ai_only = [ENGLISH * 4] * 6                      # 只标了 AI 一类
    r = client.post("/admin/api/calibrate", json={"human": [], "ai": ai_only, "profile": "en", "include_builtin": True}, headers=ADMIN)
    j = poll(client, r.json()["id"], ADMIN, "/admin/api/jobs/")
    assert j["status"] == "done", j
    rep = j["result"]["report"]
    assert rep["builtin_samples"] == 160 and rep["user_ai"] >= 1 and rep["user_human"] == 0
    assert "user_samples" in rep and rep["user_samples"]["n_ai"] >= 1
    assert j["result"]["calibration"]["profile"] == "en"


def test_default_calibration_profiles_name_the_deployed_models():
    """内置校准里每种文体记录的分类器必须与线上实际使用的一致，否则会被静默丢弃。"""
    import json
    from app import config
    cal = json.loads((Path(__file__).resolve().parent.parent / "app" / "default_calibration.json").read_text("utf-8"))
    expect = {"en": "desklib/ai-text-detector-v1.01", "zh_poetry": config.POETRY_CLASSIFIER_ID,
              "zh_classical": "yuchuantian/AIGC_detector_zhv3", "zh_short": "yuchuantian/AIGC_detector_zhv3"}
    # 文言：按文言专用分类器拟合时，部署工作流会一并部署它（见 deploy-server.yml），两者都是合法的
    # 英文：按 desklib + 英文第二分类器拟合时，部署工作流同样会一并部署第二分类器
    alt = {"zh_classical": config.CLASSICAL_CLASSIFIER_ID,
           "en": "desklib/ai-text-detector-v1.01+" + config.EN2_CLASSIFIER_ID,
           "en_paper": "desklib/ai-text-detector-v1.01+" + config.EN2_CLASSIFIER_ID}
    for prof, model in expect.items():
        if prof in cal.get("profiles", {}):
            assert cal["profiles"][prof]["models"]["classifier"] in (model, alt.get(prof)), prof


def test_work_level_consistency(client, monkeypatch):
    """同一篇作品大部分段落已判为 AI 时，接近阈值的段落按整篇计入；多数为人写时不受影响。"""
    from app import config
    h = {"Authorization": "Bearer " + issue(client)}
    paras = [MODERN.replace("宋代", f"第{i}段宋代") * 2 for i in range(4)]
    text = "《测试文章》\n\n" + "\n\n".join(paras)
    monkeypatch.setattr(config, "SMOOTHING", 0.0)
    monkeypatch.setattr(scoring, "profile_for", lambda cal, reg: ({"threshold": 0.5}, True))

    def run(probs):
        it = iter(probs)
        monkeypatch.setattr(scoring, "combine", lambda r, c: {"prob": next(it, 0.1), "signals": {}})
        return client.post("/v1/detect", json={"text": text}, headers=h).json()["result"]

    res = run([0.9, 0.9, 0.9, 0.42])
    body = [s for s in res["segments"] if s["prob"] is not None]
    assert len(body) == 4, [s["text"][:10] for s in res["segments"]]
    assert body[-1]["by_work"] and body[-1]["level"] == "light" and not body[-1]["near_threshold"]
    assert res["summary"]["ai_rate"] == 1.0

    res = run([0.1, 0.1, 0.9, 0.42])
    body = [s for s in res["segments"] if s["prob"] is not None]
    assert not body[-1]["by_work"] and body[-1]["near_threshold"] and body[-1]["level"] == "low"


def test_works_summary_per_title(client):
    """按标题分篇汇总（类似知网的章节 AI 率）。"""
    h = {"Authorization": "Bearer " + issue(client)}
    text = "《第一篇》\n\n" + MODERN * 2 + "\n\n《第二篇》\n\n" + CLASSICAL * 2
    res = client.post("/v1/detect", json={"text": text}, headers=h).json()["result"]
    works = res["works"]
    assert [w["title"] for w in works] == ["《第一篇》", "《第二篇》"]
    for w in works:
        assert w["counted"] and 0 <= w["ai_rate"] <= 1 and w["verdict"]
    assert works[1]["registers"] == ["zh_classical"]


def test_user_calibration_persists_and_resets(client):
    """管理页启用的标注校准写入持久文件，重启（重新加载）后仍在；恢复默认后清除。"""
    from app import config
    from app.main import engine
    h = {"X-Admin-Token": "test-admin-pw"}
    cal = dict(engine.cal)
    cal.pop("profiles", None)
    cal.update({"profile": "zh_classical", "threshold": 0.4242, "models": {
        "observer": config.OBSERVER_MODEL, "performer": config.PERFORMER_MODEL,
        "classifier": config.classifier_for("zh_classical")}})
    r = client.post("/admin/api/calibration", json={"calibration": cal}, headers=h)
    assert r.status_code == 200 and r.json()["saved"], r.text
    assert "zh_classical" in config.load_user_profiles()
    engine.reload_calibration()                      # 模拟服务重启
    assert scoring.profile_for(engine.cal, "zh_classical")[0]["threshold"] == 0.4242
    assert "标注校准" in engine.cal_source
    r = client.delete("/admin/api/calibration", headers=h)
    assert r.status_code == 200
    assert config.load_user_profiles() == {}
    assert scoring.profile_for(engine.cal, "zh_classical")[0].get("threshold") != 0.4242


def test_calibrate_trusts_labeled_register(client, monkeypatch):
    """报告页标注的诗词单独提交时，按标注时的文体（诗词）校准，而不是重新判断后被丢掉。"""
    import random
    from app import engine as eng
    rnd = random.Random(2)
    fake = ([{"y": 0, "s": {"fastdetect": rnd.gauss(0, 1), "binoculars": rnd.gauss(1.0, .05), "classifier": rnd.uniform(.01, .4)}} for _ in range(60)]
            + [{"y": 1, "s": {"fastdetect": rnd.gauss(2.5, 1), "binoculars": rnd.gauss(.85, .05), "classifier": rnd.uniform(.6, .99)}} for _ in range(60)])
    monkeypatch.setattr(eng, "load_builtin_calib", lambda profile: fake)
    poem = "词·《临江仙·黄四娘》\n浣花溪畔废园里，牡丹幻作红裙。杜诗千载赋花魂。陈生一顾，夜夜共芳樽。"
    r = client.post("/admin/api/calibrate", json={"human": [], "ai": [poem], "profile": "zh_poetry",
                                                   "include_builtin": True, "trust_register": True}, headers=ADMIN)
    j = poll(client, r.json()["id"], ADMIN, "/admin/api/jobs/")
    assert j["status"] == "done", j
    assert j["result"]["report"]["user_ai"] == 1
    assert j["result"]["calibration"]["profile"] == "zh_poetry"


@pytest.mark.skipif(not os.environ.get("CLASSICAL_CLASSIFIER_MODEL"), reason="未配置文言分类器")
def test_classical_uses_classical_classifier_and_mpu_second_opinion(client):
    h = {"Authorization": "Bearer " + issue(client)}
    res = client.post("/v1/detect", json={"text": CLASSICAL * 3}, headers=h).json()["result"]
    seg = [s for s in res["segments"] if s["register"] == "zh_classical"][0]
    assert "classifier" in seg["raw"] and "classifier_mpu" in seg["raw"]


def test_classical_text_fed_to_models_has_no_quotes():
    """古籍语料不带引号、AI 文言多带引号：文言送进模型前去掉引号，避免模型把"有引号"当成 AI 特征。"""
    from app.engine import score_text
    segs = segment_text("《某篇》\n\n" + "生曰：“善。”女曰：\"诺。\"" + CLASSICAL, True, True)
    s = [x for x in segs if x.register == "zh_classical"][0]
    t = score_text(s)
    assert not any(q in t for q in "“”\"「」") and "生曰：善。" in t


def test_paper_sections_stay_in_one_work_and_merge_to_long_windows():
    """论文的章节标题（1. Introduction / 2.2 … / Abstract）不应把一篇论文切成多篇作品；英文短节合并成 ≥1000 字符的窗口。"""
    para = ("Stock prices respond to news in ways that are partly predictable, and many studies document "
            "momentum over months and reversal over years across markets and periods. ")
    doc = "An Analysis of Markets\n\nAbstract\n" + para * 3 + "\n\n1. Introduction\n" + para * 2 + \
          "\n\n2.1 Efficient Markets\n" + para * 2 + "\n\n2.2 Anomalies\n" + para * 2 + "\n\n3. Methodology\n" + para * 2
    segs = [s for s in segment_text(doc, True, True) if s.kind == "body"]
    assert len({s.block for s in segs}) == 1
    assert all(len(s.text) >= 900 for s in segs[:-1])
    # 真正的新作品（书名号标题）仍然分开
    two = "《甲篇》\n\n" + MODERN * 2 + "\n\n《乙篇》\n\n" + MODERN * 2
    assert len({s.block for s in segment_text(two, True, True)}) == 2


def test_labels_sync_and_auto_calibrate(client, monkeypatch):
    """"标完自动生效"：管理员的标注同步到服务器，停手片刻后自动校准并启用；全部撤销后恢复默认。"""
    import random
    from app import config
    from app import engine as eng
    from app.main import autocal, engine
    rnd = random.Random(3)
    fake = ([{"y": 0, "s": {"fastdetect": rnd.gauss(0, 1), "binoculars": rnd.gauss(1.0, .05), "classifier": rnd.uniform(.01, .4)}} for _ in range(60)]
            + [{"y": 1, "s": {"fastdetect": rnd.gauss(2.5, 1), "binoculars": rnd.gauss(.85, .05), "classifier": rnd.uniform(.6, .99)}} for _ in range(60)])
    monkeypatch.setattr(eng, "load_builtin_calib", lambda profile: fake)
    config.save_user_profiles({})
    engine.reload_calibration()
    items = [{"key": f"k{i}", "text": CLASSICAL + str(i), "register": "zh_classical", "label": "ai"} for i in range(3)]
    r = client.post("/admin/api/labels", json={"items": items}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["by_register"]["zh_classical"]["ai"] == 3
    for _ in range(120):
        time.sleep(0.25)
        if "zh_classical" in config.load_user_profiles():
            break
    assert "zh_classical" in config.load_user_profiles()
    assert "标注" in engine.cal_source
    st = client.get("/admin/api/labels", headers=ADMIN).json()
    assert st["last"]["zh_classical"]["status"] in ("done", "running")
    # 撤销全部标注 → 恢复默认
    client.post("/admin/api/labels", json={"items": [dict(it, label=None) for it in items]}, headers=ADMIN)
    for _ in range(40):
        time.sleep(0.2)
        if "zh_classical" not in config.load_user_profiles():
            break
    assert "zh_classical" not in config.load_user_profiles()
    assert client.post("/admin/api/labels", json={"items": items}).status_code == 401
    autocal.clear()


def test_english_glued_sentences_are_respaced():
    from app.segmenter import normalize_english
    assert normalize_english("growth trend.However, prices rose.The yield was 3.8 kg.") == \
        "growth trend. However, prices rose. The yield was 3.8 kg."
    assert normalize_english("1.Introduction") == "1. Introduction"


def test_english_second_classifier_is_scored_and_named_in_calibration(client):
    """英文段落同时用 desklib 和英文第二分类器（国产大模型英文）打分；校准参数按两个分类器的组合名匹配。"""
    from app import config
    h = {"Authorization": "Bearer " + issue(client)}
    res = client.post("/v1/detect", json={"text": ENGLISH * 4}, headers=h).json()["result"]
    seg = [s for s in res["segments"] if s["register"] == "en"][0]
    assert "classifier" in seg["raw"] and "classifier_en2" in seg["raw"]
    assert config.classifier_for("en").endswith("+" + config.EN2_CLASSIFIER_ID)
    assert scoring.feature_value({"classifier_en2": 0.5}, "logit_classifier_en2") == 0.0
    assert scoring.feature_value({}, "logit_classifier_en2") is None


def test_english_paper_detection_and_profile_fallback():
    """有 Abstract / Introduction / Methods 等章节标题的英文文档按"英文学术论文"校准；没有该校准时沿用通用英文。"""
    from app.segmenter import is_english_paper
    paper = "Title\nAbstract\nText.\nKeywords: a; b\n1.Introduction\nText.\n2.Materials and Methods\nText.\n5. Conclusion\nText."
    assert is_english_paper(paper)
    assert not is_english_paper(ENGLISH * 3)
    cal = {"profiles": {"en": {"threshold": 0.7}}}
    assert scoring.profile_for(cal, "en_paper")[0]["threshold"] == 0.7


def test_second_paper_after_references_is_counted():
    """一个文档里放了两篇论文：第一篇的参考文献之后，第二篇的标题、摘要和正文要重新计入。"""
    body = ("Tomato is one of the most widely cultivated vegetable crops in the world. It is important in both fresh "
            "markets and food-processing industries. The profitability of tomato production depends on yield. ") * 4
    doc = ("Paper One Title\nAbstract\n" + body + "\nReferences\n[1] Smith J. Tomato. J Hort, 2020, 12(3): 1-9.\n---\n"
           "Technical Measures for Increasing Tomato Yield\nAbstract\n" + body + "\n1.Introduction\n" + body)
    segs = segment_text(doc)
    refs = [s for s in segs if s.kind == "reference"]
    assert len(refs) == 1 and "Smith" in refs[0].text and "Technical Measures" not in refs[0].text
    second = [s for s in segs if s.kind == "body" and s.block == refs[0].block + 1]
    assert second and second[0].text.startswith("Technical Measures")


def test_english_paper_whole_document_verdict(client, monkeypatch):
    """英文论文：第二分类器各段中位数达到阈值时整篇按 AI 计，单段低分的段落标"整篇判断"。"""
    from app import config
    import app.main as m
    monkeypatch.setattr(config, "EN_PAPER_DOC_THRESHOLD", 0.0)
    cal = dict(m.engine.cal)
    profs = dict(cal.get("profiles") or {})
    profs["en_paper"] = {"threshold": 0.999, "lr": None, "calibrated": True}
    monkeypatch.setattr(m.engine, "cal", dict(cal, profiles=profs))
    para = ENGLISH * 3
    doc = "Title of Paper\nAbstract\n" + para + "\n1. Introduction\n" + para + "\n2. Methods\n" + para + "\n3. Results\n" + para + "\n4. Conclusion\n" + para
    h = {"Authorization": "Bearer " + issue(client)}
    r = client.post("/v1/detect", json={"text": doc, "wait": True}, headers=h).json()
    res = r["result"]
    en = [s for s in res["segments"] if s["register"] == "en" and s["kind"] == "body"]
    assert len(en) >= 3 and all(s["label"] == "中度疑似（整篇判断）" for s in en)
    assert res["summary"]["ai_rate"] == 1.0


def test_titles_drawings_and_references_in_mixed_document():
    """英文标题（Title Case）开始新作品；"Weekly Tasks:" 这类英文小节标签不算；制表符框图不参与检测；
    参考文献后面紧跟的另一篇（没有 Abstract 等标题的清单类文章）重新计入。"""
    from app.segmenter import is_drawing_line, is_title
    assert is_title("The Little Fire Fox and the Star Stone")
    assert is_title("9-Month Staged Family Implementation Checklist (English Version)")
    assert is_title("弘扬长征精神,传承红色文化")
    assert not is_title("Weekly Tasks:") and not is_title("Stage 1: Boundary Building (Month 1-2)")
    assert not is_title("Common Cold; Prevention; Treatment; Viral Infection; Public Health")
    assert is_drawing_line("│    Smart Home     │    Industrial     │") and is_drawing_line("┌──────────┐")
    line = ("Negotiate with your child to confirm a fixed independent learning period every day after school, "
            "and write the agreed time on a visible whiteboard at home")
    doc = ("References\nAtzori, L., Iera, A., & Morabito, G.(2010).The Internet of Things: A survey.Computer Networks, 54(15), 2787-2805.\n\n"
           "9-Month Staged Family Implementation Checklist (English Version)\n"
           "This checklist fully aligns with the core logic of the previous paper and can be directly implemented by families.\n"
           + "\n".join([line] * 8))
    segs = segment_text(doc)
    assert [s.kind for s in segs][0] == "reference" and "Atzori" in segs[0].text and "Checklist" not in segs[0].text
    body = [s for s in segs if s.kind == "body"]
    assert body and body[0].text.startswith("9-Month Staged")


def test_chinese_paper_titles_keywords_and_law_references():
    body1 = "人的大脑究竟能够记住多少东西，这是一个很有意思的问题。现实生活中可以看到，有些人经过长期训练以后能够记住大量数字。" * 3
    body2 = "中国古典诗歌不仅是一种文学形式，也是传统文人涵养性情的重要途径。本文从意境、含蓄与音律三个方面梳理古典诗歌的审美特质。" * 3
    text = ("人类记忆与逻辑推理能力的潜力及其限度\n摘要\n" + body1 + "\n关键词： 记忆能力；逻辑推理；工作记忆；认知能力\n一、引言\n" + body1 +
            "\n二、人的记忆并不是简单的仓库\n" + body1 + "\n参考文献\nOberauer, K. (2016). What limits working memory capacity? "
            "Psychological Bulletin, 142(7), 758–799.\n\n论中国古典诗歌的审美特质与人格修养\n摘要：" + body2 + "\n二、古典诗歌的审美特质\n" + body2 +
            "\nReferences\nBidding Law of the People's Republic of China.\n"
            "Regulations for the Implementation of the Bidding Law of the People's Republic of China (State Council, 2011).\n"
            "Government Procurement Law of the People's Republic of China.\n"
            "Note: Please add the latest academic literature and case data from your own field before submission, and verify all legal citations.\n")
    segs = segment_text(text)
    works = {}
    for s in segs:
        works.setdefault(s.block, []).append(s)
    titles = [next((s.title for s in v if s.title), "") for v in works.values()]
    assert titles[0] == "人类记忆与逻辑推理能力的潜力及其限度"
    assert "论中国古典诗歌的审美特质与人格修养" in titles
    assert not any(s.register == "zh_poetry" for s in segs)          # "关键词：……；……" 不是诗
    assert not any(s.title.startswith("二、") for s in segs)           # 章节标题不当作品名
    assert all(s.kind == "reference" for s in segs if "Government Procurement" in s.text)
