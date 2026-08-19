"""
كل الإعدادات القابلة للتعديل بمكان واحد.

كل قيمة هون ممكن تتغير بثلاث طرق (الأولوية من الأضعف للأقوى):
  1. القيمة الافتراضية المكتوبة تحت.
  2. متغير بيئة بملف .env  (مثال: LAWS_TOP_N=7).
  3. قيمة مُرسلة بجسم الطلب نفسه (per-request override) — شوف الـ routes.
"""

import os
import sys
from pathlib import Path

# ── إجبار stdout/stderr على UTF-8 ───────────────────────────────────────────────
# كونسول ويندوز بيشتغل بترميز cp1256، وهاد ما بيقدر يطبع «→» ولا رموز الجداول،
# فأي print فيه رموز بيرمي UnicodeEncodeError ويوقّف السكربت. هون منصلّحها مرة
# وحدة لكل المشروع (config بينستورد من كل مكان).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # stream مش قابل لإعادة الضبط
        pass

# ── تحميل .env إن وُجد (اختياري — ما بيفشل إذا python-dotenv مش منصّب) ──────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
UTILS_DIR = Path(os.getenv("UTILS_DIR", BASE_DIR / "utils"))


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════ النماذج ══════════════════════════════════

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# "cpu" | "cuda" | "auto"  — auto = cuda إذا متوفرة، وإلا cpu
DEVICE = os.getenv("DEVICE", "auto")

# fp16 بيشتغل بس على GPU؛ على CPU بينتجل بطء أو أخطاء، فمنطفيه تلقائياً
USE_FP16 = _b("USE_FP16", True)

# مكان تخزين أوزان HuggingFace — بتنزل مرة وحدة وبتنعاد استخدامها كل مرة
HF_HOME = os.getenv("HF_HOME", str(BASE_DIR / ".hf_cache"))
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "0"))


# ═══════════════════════ Part A — RAG المواد القانونية ═══════════════════════

LAWS_DIR = Path(os.getenv("LAWS_DIR", DATA_DIR / "laws"))
LAWS_CORPUS_FILE = Path(os.getenv("LAWS_CORPUS_FILE", LAWS_DIR / "articles_unified.jsonl"))
LAWS_QDRANT_PATH = str(os.getenv("LAWS_QDRANT_PATH", LAWS_DIR / "qdrant_db"))
LAWS_COLLECTION = os.getenv("LAWS_COLLECTION", "syrian_law_articles")

LAWS_DEFAULTS = {
    # كم مادة ترجع بالنتيجة النهائية (بعد إعادة الترتيب)
    "top_n": _i("LAWS_TOP_N", 7),
    # كم مرشّح يُسحب من البحث الهجين قبل إعادة الترتيب.
    # ⚠️ هاد أهم بارامتر للسرعة: كل مرشّح = تمريرة cross-encoder إضافية.
    # النوتبوك كان 30 (على GPU). على CPU جرّب 12–15.
    "hybrid_top_k": _i("LAWS_HYBRID_TOP_K", 15),
    # استبعاد المواد الملغاة من النتائج
    "exclude_repealed": _b("LAWS_EXCLUDE_REPEALED", False),
    # عتبة الامتناع (abstention) — نتيجة تحت هالرقم بتنرمى. 0.0 = بدون عتبة
    "min_score": _f("LAWS_MIN_SCORE", 0.0),
    # دمج المواد المرتبطة (dependency merge)
    "with_dependencies": _b("LAWS_WITH_DEPENDENCIES", True),
    "dep_depth": _i("LAWS_DEP_DEPTH", 2),
    "dep_max": _i("LAWS_DEP_MAX", 12),
    "dep_dedupe_vs_hits": _b("LAWS_DEP_DEDUPE_VS_HITS", True),
    # تفكيك الاستعلامات الطويلة لمسائل قانونية منفصلة (يحتاج مفتاح LLM)
    "decompose": _b("LAWS_DECOMPOSE", False),
    "decompose_min_words": _i("LAWS_DECOMPOSE_MIN_WORDS", 25),
    # طول نافذة الـ cross-encoder. أصغر = أسرع، وأكبر = بيشوف نص المادة كامل
    "rerank_max_length": _i("LAWS_RERANK_MAX_LENGTH", 512),
}

LAWS_BUILD_BATCH = _i("LAWS_BUILD_BATCH", 8)
LAWS_EMBED_MAX_LENGTH = _i("LAWS_EMBED_MAX_LENGTH", 1024)


# ═══════════════════════ Part B — RAG السوابق القضائية ═══════════════════════

CASES_DIR = Path(os.getenv("CASES_DIR", DATA_DIR / "cases"))
CASES_INDEX_FILE = str(os.getenv("CASES_INDEX_FILE", CASES_DIR / "legal_facts_hybrid.faiss"))
CASES_METADATA_FILE = str(os.getenv("CASES_METADATA_FILE", CASES_DIR / "legal_metadata_hybrid.pkl"))
CASES_SOURCE_JSON = str(os.getenv("CASES_SOURCE_JSON", CASES_DIR / "standard_cases.json"))

CASES_DEFAULTS = {
    "top_k": _i("CASES_TOP_K", 5),
    # نفس ملاحظة السرعة يلي فوق
    "hybrid_top_k": _i("CASES_HYBRID_TOP_K", 15),
    # عتبة التشابه 0.0–1.0 (بترجع بالنتيجة كنسبة مئوية)
    "threshold": _f("CASES_THRESHOLD", 0.50),
    "rerank_max_length": _i("CASES_RERANK_MAX_LENGTH", 2048),
    "encode_batch_size": _i("CASES_ENCODE_BATCH_SIZE", 32),
}


# ══════════════════ Part C — إصدار حكم أولي (Judgment) ══════════════════

CLASSIFIER_DIR = Path(os.getenv("CLASSIFIER_DIR", DATA_DIR / "classifier"))

JUDGMENT_DEFAULTS = {
    # كم مادة/سابقة تُمرَّر للـ LLM كسياق
    "top_k_laws": _i("JUDGMENT_TOP_K_LAWS", 5),
    "top_k_cases": _i("JUDGMENT_TOP_K_CASES", 3),
    "exclude_repealed_laws": _b("JUDGMENT_EXCLUDE_REPEALED", True),
    "min_law_score": _f("JUDGMENT_MIN_LAW_SCORE", 0.0),
    "case_threshold": _f("JUDGMENT_CASE_THRESHOLD", 0.45),
    "max_quote_words": _i("JUDGMENT_MAX_QUOTE_WORDS", 12),
    # المصنّف الإحصائي (SVM) — بينشتغل بس إذا ملفات joblib موجودة بـ data/classifier
    "use_domain_classifier": _b("JUDGMENT_USE_CLASSIFIER", True),
    "top_k_classifier_labels": _i("JUDGMENT_TOP_K_CLASSIFIER_LABELS", 5),
    "classifier_min_score": _f("JUDGMENT_CLASSIFIER_MIN_SCORE", -1.0),
    "precedent_pool_size": _i("JUDGMENT_PRECEDENT_POOL_SIZE", 15),
    # v2 — إعادة تنظيم الوقائع قبل الاسترجاع (استدعاء LLM إضافي)
    "use_fact_reorganization": _b("JUDGMENT_USE_FACT_REORG", True),
    "show_reorganized_facts_to_llm": _b("JUDGMENT_SHOW_REORG_TO_LLM", True),
    "fact_reorg_max_tokens": _i("JUDGMENT_FACT_REORG_MAX_TOKENS", 512),
    # v3 — تمييز المواد المتشابهة لفظياً (استدعاء LLM لكل زوج)
    "use_statute_discrimination": _b("JUDGMENT_USE_DISCRIMINATION", True),
    "confusable_overlap_threshold": _f("JUDGMENT_CONFUSABLE_OVERLAP", 0.15),
    "max_confusable_pairs": _i("JUDGMENT_MAX_CONFUSABLE_PAIRS", 3),
    "discrimination_max_tokens": _i("JUDGMENT_DISCRIMINATION_MAX_TOKENS", 400),
    # الـ LLM
    "groq_model": os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
    "max_output_tokens": _i("JUDGMENT_MAX_OUTPUT_TOKENS", 3072),
    "temperature": _f("JUDGMENT_TEMPERATURE", 0.1),
}

# ⚠️ المفتاح ما بينكتب بالكود أبداً — بينقرأ من .env فقط
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


# ═══════════════════════════════ إقلاع السيرفر ═══════════════════════════════

# أي ميزات تُحمّل نماذجها عند الإقلاع بدل أول طلب.
# فاضي (الافتراضي) = تحميل كسول → السيرفر بيقلع بثانية، وأول طلب لكل ميزة بياخد وقت.
# مثال: WARMUP=laws,cases,search
WARMUP = [s.strip() for s in os.getenv("WARMUP", "").split(",") if s.strip()]

PORT = _i("PORT", 5000)
DEBUG = _b("DEBUG", True)


def resolve_device() -> str:
    """يحوّل DEVICE='auto' لقيمة فعلية."""
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def use_fp16() -> bool:
    """fp16 على CPU بيعطي نتائج غلط/بطيئة — منسمحله بس على GPU."""
    return USE_FP16 and resolve_device() == "cuda"
