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
    "ADMIN_TOKEN": "test-admin-pw", "LM_MAX_TOKENS": "128", "CALIBRATION_FILE": "/nonexistent/cal.json",
    "MAX_TEXT_CHARS": "300000",
})
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import keys, scoring  # noqa: E402
from app.segmenter import classical_ratio, segment_text  # noqa: E402

MODERN = ("宋代的地方行政制度在很大程度上延续了唐末五代的格局，但又有所调整。据《宋史·职官志》记载，路一级机构的设置经历了反复变化，"
          "转运使司的职能也随之扩展。这一时期的文献对州县官员的任免多有记述，然而其中不乏相互矛盾之处。笔者以为，此类记载须与地方志、碑刻互相参证，方能厘清其真实面貌。")
CLASSICAL = ("太祖既受禅，患藩镇之强，乃用赵普之谋，稍夺其权。或曰：其所以然者，盖鉴于唐末五代之乱也。帝曰：善。于是诸镇皆罢，"
             "以文臣知州事，而兵权悉归于上。君子曰：此所谓强干弱枝者也，其虑深矣，岂不然哉。")


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
    assert classical_ratio(MODERN) < 0.07 <= classical_ratio(CLASSICAL)


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
    assert h["lm"]["ready"] and h["classifier"]["ready"], h
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
    assert s["methods"] == {"fastdetect": True, "binoculars": True, "classifier": True}
    assert 0 <= s["ai_rate"] <= 1 and s["counted_chars"] > 0
    seg = j["result"]["segments"][0]
    assert set(seg["raw"]) >= {"fastdetect", "binoculars", "classifier", "ppl", "lrr", "log_rank", "entropy", "top10", "style_cv"}
    assert set(s["segments_by_level"]) == {"high", "mid", "light", "low"}
    assert "reliability_notes" in s and any("校准" in n for n in s["reliability_notes"])
    assert 0 <= seg["prob"] <= 1


def test_all_quotation_falls_back_and_excluded_get_reference_value(client):
    h = {"Authorization": "Bearer " + issue(client)}
    # 全文都是文言：应退回为全部计入，报告有数值
    s = client.post("/v1/detect", json={"text": CLASSICAL * 3}, headers=h).json()["result"]
    assert s["summary"]["fallback_all_counted"] is True
    assert s["summary"]["counted_chars"] > 0 and s["summary"]["ai_rate"] is not None
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
