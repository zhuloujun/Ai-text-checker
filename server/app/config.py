"""运行配置：全部来自环境变量（由部署平台注入：Modal 的 Secret、Hugging Face Space 的 Variables and secrets 等）。"""
import json
import os
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


BASE_DIR = Path(__file__).resolve().parent.parent

# ---------- 模型 ----------
# 打分用的两个语言模型：必须共用同一个分词器（同一系列的基础版 + 对话版）。
OBSERVER_MODEL = os.getenv("OBSERVER_MODEL", "Qwen/Qwen2.5-0.5B")
PERFORMER_MODEL = os.getenv("PERFORMER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# 中文 AI 文本分类器（MPU，ICLR 2024）。留空则不启用。
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "yuchuantian/AIGC_detector_zhv3")
# 分类器中哪个标签表示"AI 生成"。auto = 按标签名自动判断，判断不了时取下标 1。
CLASSIFIER_AI_LABEL = os.getenv("CLASSIFIER_AI_LABEL", "auto")

ENABLE_LM = _bool("ENABLE_LM", True)
ENABLE_CLASSIFIER = _bool("ENABLE_CLASSIFIER", True)

TORCH_THREADS = _int("TORCH_THREADS", os.cpu_count() or 2)
LM_MAX_TOKENS = _int("LM_MAX_TOKENS", 512)          # 单段送入语言模型的最大 token 数
CLS_MAX_TOKENS = _int("CLS_MAX_TOKENS", 512)

# ---------- 分段 ----------
SEGMENT_TARGET_CHARS = _int("SEGMENT_TARGET_CHARS", 400)
SEGMENT_MIN_CHARS = _int("SEGMENT_MIN_CHARS", 80)
FAST_MODE_MAX_SEGMENTS = _int("FAST_MODE_MAX_SEGMENTS", 60)  # 快速模式下语言模型最多检测多少段
# 相邻段落平滑强度（0 = 不平滑，0.3 = 本段 70% + 相邻段 30%）
SMOOTHING = float(os.getenv("SMOOTHING", "0.3") or 0.3)
# 文言虚词（之乎者也矣焉哉曰…）占汉字比例超过此值的段落，视为以古籍引文为主，不计入 AI 率
CLASSICAL_THRESHOLD = float(os.getenv("CLASSICAL_THRESHOLD", "0.07") or 0.07)

# ---------- 限制 ----------
MAX_TEXT_CHARS = _int("MAX_TEXT_CHARS", 300_000)
SYNC_MAX_CHARS = _int("SYNC_MAX_CHARS", 3_000)       # 小于这个长度的请求直接同步返回
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 30)
MAX_QUEUED_JOBS = _int("MAX_QUEUED_JOBS", 20)
JOB_TTL_SECONDS = _int("JOB_TTL_SECONDS", 3600)

# ---------- API Key ----------
# KEY_SECRET：签发 Key 的签名密钥；ADMIN_TOKEN：管理页面密码。两者都应设为 Secret。
KEY_SECRET = os.getenv("KEY_SECRET", "")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
REVOKED_KEY_IDS = {k.strip() for k in os.getenv("REVOKED_KEY_IDS", "").split(",") if k.strip()}
REQUIRE_KEY = _bool("REQUIRE_KEY", True)               # False = 网页和接口都无需 Key（不建议公开 Space 这样做）
DEFAULT_DAILY_CHARS = _int("DEFAULT_DAILY_CHARS", 1_000_000)

# ---------- 校准 ----------
# 可以直接把校准结果 JSON 填在 CALIBRATION_JSON 变量里；否则读 calibration.json；都没有就用内置默认值。
CALIBRATION_FILE = Path(os.getenv("CALIBRATION_FILE", str(BASE_DIR / "calibration.json")))


def load_calibration_override():
    raw = os.getenv("CALIBRATION_JSON", "").strip()
    if raw:
        try:
            return json.loads(raw), "环境变量 CALIBRATION_JSON"
        except json.JSONDecodeError:
            pass
    if CALIBRATION_FILE.exists():
        try:
            return json.loads(CALIBRATION_FILE.read_text("utf-8")), f"文件 {CALIBRATION_FILE.name}"
        except (OSError, json.JSONDecodeError):
            pass
    return None, "内置默认值（未校准）"
