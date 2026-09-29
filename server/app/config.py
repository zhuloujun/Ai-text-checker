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
# 英文 AI 文本分类器（desklib，DeBERTa-v3-large，RAID 基准）。留空则英文只用语言模型特征。
EN_CLASSIFIER_MODEL = os.getenv("EN_CLASSIFIER_MODEL", "desklib/ai-text-detector-v1.01")
ENABLE_EN_CLASSIFIER = _bool("ENABLE_EN_CLASSIFIER", True)
EN_CLS_MAX_TOKENS = _int("EN_CLS_MAX_TOKENS", 512)
# 诗词专用分类器（tools/train_poetry.py 在 ChangAn 上微调，发布在本仓库 Release）。填本地目录；留空则诗词用通用中文分类器。
POETRY_CLASSIFIER_MODEL = os.getenv("POETRY_CLASSIFIER_MODEL", "")
POETRY_CLASSIFIER_ID = os.getenv("POETRY_CLASSIFIER_ID", "poetry-classifier-v1")   # 校准参数按这个名字匹配模型
POETRY_CLASSIFIER_URL = os.getenv(
    "POETRY_CLASSIFIER_URL",
    "https://github.com/zhuloujun/ceshi/releases/download/poetry-classifier-v1/poetry-classifier.tar.gz")

# 文言专用分类器（tools/train_classical.py 用古籍人写 vs DeepSeek / Kimi / 文心一言等生成的文言微调，发布在本仓库 Release）。
CLASSICAL_CLASSIFIER_MODEL = os.getenv("CLASSICAL_CLASSIFIER_MODEL", "")
CLASSICAL_CLASSIFIER_ID = os.getenv("CLASSICAL_CLASSIFIER_ID", "classical-classifier-v2")

ENABLE_LM = _bool("ENABLE_LM", True)
ENABLE_CLASSIFIER = _bool("ENABLE_CLASSIFIER", True)

TORCH_THREADS = _int("TORCH_THREADS", os.cpu_count() or 2)
LM_MAX_TOKENS = _int("LM_MAX_TOKENS", 512)          # 单段送入语言模型的最大 token 数
LM_DTYPE = os.getenv("LM_DTYPE", "float32")          # float32（默认）/ bfloat16：大模型省一半内存
CLS_MAX_TOKENS = _int("CLS_MAX_TOKENS", 512)

# ---------- 分段 ----------
SEGMENT_TARGET_CHARS = _int("SEGMENT_TARGET_CHARS", 400)
SEGMENT_MIN_CHARS = _int("SEGMENT_MIN_CHARS", 80)
SEGMENT_TARGET_CHARS_EN = _int("SEGMENT_TARGET_CHARS_EN", 1000)   # 英文约 170 词
SEGMENT_MIN_CHARS_EN = _int("SEGMENT_MIN_CHARS_EN", 200)
# 遇到段落分隔时，窗口不足这么长就与下一段合并（同一作品、同一文体内）
SEGMENT_FLUSH_CHARS = _int("SEGMENT_FLUSH_CHARS", 200)
SEGMENT_FLUSH_CHARS_CLASSICAL = _int("SEGMENT_FLUSH_CHARS_CLASSICAL", 120)
SEGMENT_FLUSH_CHARS_EN = _int("SEGMENT_FLUSH_CHARS_EN", 600)
# 现代汉语段落短于这个字数时，改用"短段"校准（有的话）：短文本信号弱，需要单独的阈值
SHORT_SEGMENT_CHARS = _int("SHORT_SEGMENT_CHARS", 200)
# 篇幅短的提示（Turnitin 要求英文至少 300 词才给结果；这里对单段放宽，只做标注）
SHORT_CHARS_ZH = _int("SHORT_CHARS_ZH", 100)
SHORT_WORDS_EN = _int("SHORT_WORDS_EN", 150)
# "疑似名篇"判定：语言模型困惑度低于此值且分类器判为人写
MEMORIZED_PPL = float(os.getenv("MEMORIZED_PPL", "3.5") or 3.5)
# 管理页校准时，用户样本与内置公开数据合并，用户样本合计所占的权重比例
USER_SAMPLE_SHARE = float(os.getenv("USER_SAMPLE_SHARE", "0.3") or 0.3)
FAST_MODE_MAX_SEGMENTS = _int("FAST_MODE_MAX_SEGMENTS", 60)  # 快速模式下语言模型最多检测多少段
# 相邻段落平滑强度（0 = 不平滑，0.3 = 本段 70% + 相邻段 30%）
SMOOTHING = float(os.getenv("SMOOTHING", "0.3") or 0.3)
WORK_MAJORITY = float(os.getenv("WORK_MAJORITY", "0.6") or 0.6)   # 同篇已判 AI 的文字占比达到此值，接近阈值的段落按整篇计入
# 文言虚词（之乎者也矣焉哉曰…）占汉字比例超过此值的段落，视为以古籍引文为主，不计入 AI 率
CLASSICAL_THRESHOLD = float(os.getenv("CLASSICAL_THRESHOLD", "0.03") or 0.03)
MODERN_MAX_RATIO = float(os.getenv("MODERN_MAX_RATIO", "0.015") or 0.015)

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


def classifier_for(register: str) -> str:
    if register == "en":
        return EN_CLASSIFIER_MODEL
    if register == "zh_poetry" and POETRY_CLASSIFIER_MODEL:
        return POETRY_CLASSIFIER_ID
    if register == "zh_classical" and CLASSICAL_CLASSIFIER_MODEL:
        return CLASSICAL_CLASSIFIER_ID
    return CLASSIFIER_MODEL


def _models_match(cal: dict, register: str) -> bool:
    m = cal.get("models") or {}
    if not m:
        return True
    return (m.get("observer"), m.get("performer"), m.get("classifier")) == (
        OBSERVER_MODEL, PERFORMER_MODEL, classifier_for(register))


def _filter_profiles(cal: dict) -> dict:
    """去掉与当前模型不一致的文体校准（换了模型，旧参数就不适用了）。"""
    profs = {k: v for k, v in (cal.get("profiles") or {}).items() if isinstance(v, dict) and _models_match(v, k)}
    out = dict(cal)
    if profs:
        out["profiles"] = profs
    else:
        out.pop("profiles", None)
    return out


def load_default_calibration():
    """随代码发布的默认校准（由 tools/evaluate.py 用公开数据集生成）。"""
    default = Path(__file__).resolve().parent / "default_calibration.json"
    if default.exists():
        try:
            cal = json.loads(default.read_text("utf-8"))
            cal = _filter_profiles(cal)
            if not _models_match(cal, "zh"):
                # 现代汉语部分不适用时，只保留仍适用的文体校准
                return ({"profiles": cal["profiles"]} if cal.get("profiles") else None)
            return cal
        except (OSError, json.JSONDecodeError):
            pass
    return None


# 管理页“用我的标注校准”后启用的各文体校准，永久保存在这个文件里（线上放在 Modal 持久卷上，服务重启后仍有效）。
# 只保存用户自己校准过的文体；其余文体始终跟随内置默认校准的更新。
USER_CALIBRATION_FILE = Path(os.getenv("USER_CALIBRATION_FILE", str(BASE_DIR / "user_calibration.json")))
CALIBRATION_VOLUME = os.getenv("CALIBRATION_VOLUME", "")


def load_user_profiles() -> dict:
    try:
        d = json.loads(USER_CALIBRATION_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return {p: c for p, c in (d.get("profiles") or {}).items()
            if isinstance(c, dict) and _models_match(c, p)}


def save_user_profiles(profiles: dict) -> bool:
    """写入并提交到持久卷。返回是否成功永久保存。"""
    try:
        USER_CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        USER_CALIBRATION_FILE.write_text(json.dumps({"format": "user_profiles_v1", "profiles": profiles},
                                                    ensure_ascii=False, indent=1), "utf-8")
    except OSError:
        return False
    if CALIBRATION_VOLUME:
        try:
            import modal
            modal.Volume.from_name(CALIBRATION_VOLUME).commit()
        except Exception:  # noqa: BLE001  提交失败时，容器正常退出时 Modal 也会自动提交
            pass
    return True


def load_calibration_override():
    """管理员自己的校准：环境变量 CALIBRATION_JSON 优先，其次 calibration.json。"""
    raw = os.getenv("CALIBRATION_JSON", "").strip()
    if raw:
        try:
            return _filter_profiles(json.loads(raw)), "环境变量 CALIBRATION_JSON"
        except json.JSONDecodeError:
            pass
    if CALIBRATION_FILE.exists():
        try:
            return _filter_profiles(json.loads(CALIBRATION_FILE.read_text("utf-8"))), f"文件 {CALIBRATION_FILE.name}"
        except (OSError, json.JSONDecodeError):
            pass
    return None, None
