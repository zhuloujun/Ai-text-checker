"""HTTP 接口（FastAPI）。

公开：
  GET  /                  网页
  GET  /admin             管理页（签发 Key、校准）
  GET  /health            运行状态
需要 API Key（请求头 Authorization: Bearer <key> 或 X-API-Key: <key>）：
  POST /v1/detect         JSON {text, mode, exclude_references, flag_quotations, wait}
  POST /v1/detect/file    表单上传 file（.docx/.pdf/.txt）+ 同样的选项
  GET  /v1/jobs/{id}      查询任务进度与结果
  GET  /v1/me             查看当前 Key 信息与今日用量
需要管理员密码（请求头 X-Admin-Token）：
  POST /admin/api/keys            签发 Key
  POST /admin/api/revoke          临时作废 Key（重启后需靠 REVOKED_KEY_IDS 保持）
  GET  /admin/api/usage           各 Key 今日用量
  POST /admin/api/calibrate       提交校准样本（后台任务）
  GET  /admin/api/jobs/{id}       查询校准任务
  POST /admin/api/calibration     立即启用一份校准参数
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, docparse, keys, scoring
from .engine import Engine, JobQueue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")

STATIC = config.BASE_DIR / "static"
app = FastAPI(title="Ai-text-checker", version="2.0", docs_url="/docs", redoc_url=None)
engine = Engine()
jobs = JobQueue(engine)
threading.Thread(target=engine.load_all, daemon=True).start()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def _no_stale_assets(request, call_next):
    # 让浏览器每次都向服务器确认网页和脚本是否更新（未更新时只返回 304，很快），
    # 避免浏览器继续用旧版 app.js，导致新功能（如本地文稿库）不生效
    resp = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path in ("/", "/admin"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


def _versioned_html(name: str) -> HTMLResponse:
    """给页面里引用的 /static/ 文件加上内容指纹 ?v=xxxx，文件一变浏览器就会重新下载。"""
    html = (STATIC / name).read_text(encoding="utf-8")

    def ver(m):
        f = STATIC / m.group(1)
        if not f.is_file():
            return m.group(0)
        return f'/static/{m.group(1)}?v={hashlib.sha1(f.read_bytes()).hexdigest()[:10]}"'
    html = re.sub(r'/static/([\w.-]+)"', ver, html)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

_admin_fail: dict[str, list[float]] = {}


def err(status: int, code: str, message: str):
    raise HTTPException(status_code=status, detail={"error": code, "message": message})


@app.exception_handler(HTTPException)
async def http_exc(_req: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"error": "http_error", "message": str(exc.detail)}
    return JSONResponse(detail, status_code=exc.status_code)


# ---------------- 鉴权 ----------------

def require_key(authorization: str | None, x_api_key: str | None) -> dict:
    if not config.REQUIRE_KEY:
        return {"i": "public", "n": "public", "q": 0}
    if not keys.signing_available():
        err(503, "not_configured", "服务尚未配置管理员密码 ADMIN_TOKEN（在 GitHub 仓库 Settings → Secrets and variables → Actions 里添加 HF_ADMIN_TOKEN 后重新部署），然后到 /admin 页面签发 API Key。")
    token = (x_api_key or "").strip()
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token:
        err(401, "missing_key", "缺少 API Key。请在请求头加 Authorization: Bearer <你的 Key>。")
    payload = keys.verify(token)
    if not payload:
        err(401, "invalid_key", "API Key 无效、已过期或已被作废。")
    return payload


def require_admin(request: Request, token: str | None):
    # 经 Cloudflare Worker 转发时，真实访客 IP 在 X-Forwarded-For 的第一个
    fwd = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    ip = fwd or (request.client.host if request.client else "?")
    now = time.time()
    fails = [t for t in _admin_fail.get(ip, []) if now - t < 600]
    if len(fails) >= 10:
        err(429, "too_many_attempts", "管理员密码错误次数过多，请 10 分钟后再试。")
    if not config.ADMIN_TOKEN:
        err(503, "not_configured", "未设置 ADMIN_TOKEN。请在 GitHub 仓库 Settings → Secrets and variables → Actions 里添加 HF_ADMIN_TOKEN 后重新部署。")
    if not token or not hmac.compare_digest(token.encode(), config.ADMIN_TOKEN.encode()):
        fails.append(now)
        _admin_fail[ip] = fails
        err(401, "bad_admin_token", "管理员密码不正确。")


def ensure_ready():
    # 模型还在加载时照常接收任务：任务会排队，等全部模型加载完再开始打分
    if not engine.loading and not engine.any_ready():
        err(503, "no_detector", "没有可用的检测模型，请查看 /health 里的错误信息。")


# ---------------- 页面 ----------------

@app.get("/", include_in_schema=False)
def index():
    return _versioned_html("index.html")


@app.get("/admin", include_in_schema=False)
def admin_page():
    return _versioned_html("admin.html")


@app.get("/health")
def health():
    return {"ok": True, "requires_key": config.REQUIRE_KEY, "key_signing_configured": keys.signing_available(),
            "queue": jobs.pending(), "limits": {"max_chars": config.MAX_TEXT_CHARS, "max_upload_mb": config.MAX_UPLOAD_MB,
                                               "fast_mode_max_segments": config.FAST_MODE_MAX_SEGMENTS},
            **engine.status()}


# ---------------- 检测 ----------------

class DetectIn(BaseModel):
    text: str = Field(..., description="待检测文本")
    mode: str = Field("full", description="full = 逐段全部检测；fast = 语言模型抽样检测")
    exclude_references: bool = True
    flag_quotations: bool = True
    wait: bool | None = Field(None, description="是否等待结果；默认短文本等待、长文本返回任务编号")


async def _submit_detect(text: str, mode: str, excl: bool, flagq: bool, wait: bool | None, key: dict):
    ensure_ready()
    text = (text or "").replace("\r\n", "\n")
    if len(text.strip()) < 50:
        err(400, "too_short", "文本太短（少于 50 字），无法可靠检测。")
    if len(text) > config.MAX_TEXT_CHARS:
        err(413, "too_long", f"文本超过 {config.MAX_TEXT_CHARS} 字上限。")
    if mode not in ("full", "fast"):
        err(400, "bad_mode", "mode 只能是 full 或 fast。")
    if jobs.pending() >= config.MAX_QUEUED_JOBS:
        err(429, "busy", "排队任务太多，请稍后再试。")
    if key["i"] != "public":
        ok, left = keys.consume(key, len(text))
        if not ok:
            err(429, "quota_exceeded", f"今日额度不足：剩余 {left} 字。额度每天（UTC）零点重置。")
    job = jobs.submit("detect", key["i"], {"text": text, "mode": mode, "exclude_references": excl,
                                          "flag_quotations": flagq})
    if wait is None:
        wait = len(text) <= config.SYNC_MAX_CHARS
    if wait:
        deadline = time.time() + 90
        while time.time() < deadline:
            j = jobs.get(job["id"])
            if j and j["status"] in ("done", "error"):
                return jobs.public(j)
            await asyncio.sleep(0.3)
    return JSONResponse(jobs.public(jobs.get(job["id"])), status_code=202)


@app.post("/v1/detect")
async def detect(body: DetectIn, authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    key = require_key(authorization, x_api_key)
    return await _submit_detect(body.text, body.mode, body.exclude_references, body.flag_quotations, body.wait, key)


@app.post("/v1/detect/file")
async def detect_file(file: UploadFile = File(...), mode: str = Form("full"),
                      exclude_references: bool = Form(True), flag_quotations: bool = Form(True),
                      wait: bool | None = Form(None),
                      authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    key = require_key(authorization, x_api_key)
    data = await file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        err(413, "file_too_large", f"文件超过 {config.MAX_UPLOAD_MB} MB。")
    try:
        text = await asyncio.to_thread(docparse.extract, file.filename or "", data)
    except ValueError as e:
        err(400, "bad_file", str(e))
    except Exception as e:  # noqa: BLE001
        err(400, "parse_failed", f"文件解析失败：{type(e).__name__}")
    return await _submit_detect(text, mode, exclude_references, flag_quotations, wait, key)


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str, authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    key = require_key(authorization, x_api_key)
    j = jobs.get(job_id)
    if not j or (j["kind"] != "detect") or (j["owner"] != key["i"]):
        err(404, "not_found", "任务不存在或已过期（结果保留 1 小时）。")
    return jobs.public(j)


@app.get("/v1/me")
def me(authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    key = require_key(authorization, x_api_key)
    usage = keys.usage_snapshot().get(key["i"], {"chars": 0, "requests": 0})
    return {"id": key["i"], "name": key.get("n"), "daily_chars": key.get("q"),
            "expires": key.get("e") or None, "today": usage}


# ---------------- 管理 ----------------

class IssueIn(BaseModel):
    name: str = ""
    days: int = 0
    daily_chars: int | None = None


class RevokeIn(BaseModel):
    id: str


class CalibrateIn(BaseModel):
    human: list[str]
    ai: list[str]
    target_fpr: float = 0.05
    profile: str = Field("auto", description="auto / zh（现代汉语）/ zh_classical（文言）/ en（英文）")


class CalibrationIn(BaseModel):
    calibration: dict


@app.post("/admin/api/keys")
def admin_issue(body: IssueIn, request: Request, x_admin_token: str | None = Header(None)):
    require_admin(request, x_admin_token)
    try:
        return keys.issue(body.name, body.days, body.daily_chars)
    except RuntimeError as e:
        err(503, "not_configured", str(e))


@app.post("/admin/api/revoke")
def admin_revoke(body: RevokeIn, request: Request, x_admin_token: str | None = Header(None)):
    require_admin(request, x_admin_token)
    keys.revoke_runtime(body.id.strip())
    return {"ok": True, "message": f"已临时作废 {body.id}。要在服务重启后仍然有效，请在 GitHub 仓库 Settings → Secrets and variables → Actions 的 Variables 里把它加入 REVOKED_KEY_IDS（逗号分隔），然后重新部署。"}


@app.get("/admin/api/usage")
def admin_usage(request: Request, x_admin_token: str | None = Header(None)):
    require_admin(request, x_admin_token)
    return {"usage": keys.usage_snapshot(), "revoked_env": sorted(config.REVOKED_KEY_IDS)}


@app.post("/admin/api/calibrate")
def admin_calibrate(body: CalibrateIn, request: Request, x_admin_token: str | None = Header(None)):
    require_admin(request, x_admin_token)
    ensure_ready()
    if body.profile not in ("auto", "zh", "zh_classical", "en"):
        err(400, "bad_profile", "profile 只能是 auto、zh、zh_classical 或 en。")
    if sum(len(t) for t in body.human + body.ai) > config.MAX_TEXT_CHARS:
        err(413, "too_long", f"校准样本总字数超过 {config.MAX_TEXT_CHARS}。")
    return jobs.submit("calibrate", "admin", body.model_dump())


@app.get("/admin/api/jobs/{job_id}")
def admin_job(job_id: str, request: Request, x_admin_token: str | None = Header(None)):
    require_admin(request, x_admin_token)
    j = jobs.get(job_id)
    if not j:
        err(404, "not_found", "任务不存在或已过期。")
    return jobs.public(j)


@app.post("/admin/api/calibration")
def admin_apply(body: CalibrationIn, request: Request, x_admin_token: str | None = Header(None)):
    require_admin(request, x_admin_token)
    cal = body.calibration
    if "signals" not in cal or "threshold" not in cal:
        err(400, "bad_calibration", "校准参数格式不正确。")
    # 只替换这次校准的文体，其他文体的校准保持不变
    cal = scoring.merge_profile(engine.cal, cal, cal.get("profile") or "zh")
    engine.set_calibration(cal, "管理页面（运行时，重启后失效）")
    try:
        Path(config.CALIBRATION_FILE).write_text(json.dumps(cal, ensure_ascii=False, indent=2), "utf-8")
    except OSError:
        pass
    return {"ok": True, "calibration": cal,
            "message": "已启用。服务重启后会恢复原设置：请在 GitHub 仓库 Settings → Secrets and variables → Actions 的 Variables 里新建 CALIBRATION_JSON，值为下面这段完整 JSON（已合并各文体的校准），然后重新部署。"}
