"""自签发 API Key。

免费 Space 的磁盘在重启后会清空，所以 Key 不存数据库，而是"自带签名"：
  atc-<载荷>.<签名>
载荷里写着 Key 编号、名称、到期时间、每日额度；签名用 KEY_SECRET（未设置时由 ADMIN_TOKEN 派生）计算。
服务端只要重新算一遍签名就能验证，重启也不会失效。
作废某个 Key：把它的编号加到环境变量 REVOKED_KEY_IDS（逗号分隔）。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from datetime import datetime, timezone

from . import config

PREFIX = "atc-"
_revoked_runtime: set[str] = set()
_usage: dict[str, dict] = {}
_usage_lock = threading.Lock()


def _secret() -> bytes:
    if config.KEY_SECRET:
        return config.KEY_SECRET.encode()
    if config.ADMIN_TOKEN:
        return hmac.new(config.ADMIN_TOKEN.encode(), b"ai-text-checker/key-secret/v1", hashlib.sha256).digest()
    return b""


def signing_available() -> bool:
    return bool(_secret())


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload_b64: str) -> str:
    return _b64(hmac.new(_secret(), payload_b64.encode(), hashlib.sha256).digest()[:24])


def issue(name: str, days: int = 0, daily_chars: int | None = None) -> dict:
    if not signing_available():
        raise RuntimeError("未设置 ADMIN_TOKEN 或 KEY_SECRET，无法签发 Key")
    kid = secrets.token_hex(4)
    exp = int(time.time() + days * 86400) if days and days > 0 else 0
    payload = {"i": kid, "n": (name or "")[:40], "e": exp,
               "q": int(daily_chars if daily_chars is not None else config.DEFAULT_DAILY_CHARS),
               "t": int(time.time())}
    p64 = _b64(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode())
    return {"key": f"{PREFIX}{p64}.{_sign(p64)}", "id": kid, "name": payload["n"],
            "expires": None if not exp else datetime.fromtimestamp(exp, timezone.utc).isoformat(),
            "daily_chars": payload["q"]}


def verify(key: str) -> dict | None:
    """有效则返回载荷，否则返回 None。"""
    if not key or not key.startswith(PREFIX) or not signing_available():
        return None
    try:
        p64, sig = key[len(PREFIX):].split(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(_sign(p64), sig):
        return None
    try:
        payload = json.loads(_unb64(p64))
    except (ValueError, json.JSONDecodeError):
        return None
    kid = payload.get("i", "")
    if kid in config.REVOKED_KEY_IDS or kid in _revoked_runtime:
        return None
    if payload.get("e") and payload["e"] < time.time():
        return None
    return payload


def revoke_runtime(kid: str):
    _revoked_runtime.add(kid)


def consume(payload: dict, chars: int) -> tuple[bool, int]:
    """记录用量；超出每日额度返回 (False, 剩余)。用量只保存在内存里，服务重启后清零。"""
    kid = payload["i"]
    quota = int(payload.get("q") or config.DEFAULT_DAILY_CHARS)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    with _usage_lock:
        u = _usage.setdefault(kid, {"day": day, "chars": 0, "requests": 0, "name": payload.get("n", "")})
        if u["day"] != day:
            u.update(day=day, chars=0, requests=0)
        if quota > 0 and u["chars"] + chars > quota:
            return False, max(0, quota - u["chars"])
        u["chars"] += chars
        u["requests"] += 1
        return True, (quota - u["chars"]) if quota > 0 else -1


def usage_snapshot() -> dict:
    with _usage_lock:
        return {k: dict(v) for k, v in _usage.items()}
