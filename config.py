"""
جميع الإعدادات القابلة للتعديل في مكان واحد.

تُضبط كل قيمة بثلاث طرق، والأولوية من الأضعف إلى الأقوى:
  1. القيمة الافتراضية المذكورة أدناه.
  2. متغير بيئة في ملف .env  (مثال: LAWS_TOP_N=7).
  3. قيمة مُرسلة ضمن جسم الطلب، وتسري على ذلك الطلب وحده.
"""

import os
import sys
from pathlib import Path

# إجبار stdout/stderr على ترميز UTF-8.
# الترميز الافتراضي لكونسول ويندوز هو cp1256 ولا يدعم كثيراً من الرموز، فيرمي
# أي print يحتوي عليها UnicodeEncodeError. تُضبط هنا مرة واحدة للمشروع كله.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # مجرى غير قابل لإعادة الضبط
        pass

# تحميل .env إن وُجد؛ لا يفشل إذا لم تكن python-dotenv مثبّتة.
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


# النماذج

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# "cpu" | "cuda" | "auto" — auto يختار cuda إن توفّرت، وإلا cpu.
DEVICE = os.getenv("DEVICE", "auto")

# fp16 مدعوم على الـ GPU فقط؛ يُعطَّل تلقائياً على الـ CPU.
USE_FP16 = _b("USE_FP16", True)

# مكان تخزين أوزان HuggingFace؛ تُنزَّل مرة واحدة ويُعاد استخدامها.
HF_HOME = os.getenv("HF_HOME", str(BASE_DIR / ".hf_cache"))
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "0"))


# المواد القانونية

LAWS_DIR = Path(os.getenv("LAWS_DIR", DATA_DIR / "laws"))
LAWS_CORPUS_FILE = Path(os.getenv("LAWS_CORPUS_FILE", LAWS_DIR / "articles_unified.jsonl"))
LAWS_QDRANT_PATH = str(os.getenv("LAWS_QDRANT_PATH", LAWS_DIR / "qdrant_db"))
LAWS_COLLECTION = os.getenv("LAWS_COLLECTION", "syrian_law_articles")

LAWS_DEFAULTS = {
    # عدد المواد في النتيجة النهائية بعد إعادة الترتيب.
    "top_n": _i("LAWS_TOP_N", 7),
    # عدد المرشحين المسحوبين من البحث الهجين قبل إعادة الترتيب.
    # أهم بارامتر في الأداء: كل مرشح يقابل تمريرة إضافية على الـ cross-encoder.
    "hybrid_top_k": _i("LAWS_HYBRID_TOP_K", 15),
    # استبعاد المواد الملغاة من النتائج
    "exclude_repealed": _b("LAWS_EXCLUDE_REPEALED", False),
    # عتبة الامتناع: تُستبعد أي نتيجة دون هذا الرقم. 0.0 يعني بلا عتبة.
    "min_score": _f("LAWS_MIN_SCORE", 0.0),
    # دمج المواد المرتبطة (dependency merge)
    "with_dependencies": _b("LAWS_WITH_DEPENDENCIES", True),
    "dep_depth": _i("LAWS_DEP_DEPTH", 2),
    "dep_max": _i("LAWS_DEP_MAX", 12),
    "dep_dedupe_vs_hits": _b("LAWS_DEP_DEDUPE_VS_HITS", True),
    # تفكيك الاستعلامات الطويلة إلى مسائل قانونية منفصلة؛ يتطلب مفتاح نموذج لغوي.
    "decompose": _b("LAWS_DECOMPOSE", False),
    "decompose_min_words": _i("LAWS_DECOMPOSE_MIN_WORDS", 25),
    # طول نافذة الـ cross-encoder. القيمة الأصغر أسرع، والأكبر تغطي نص المادة كاملاً.
    "rerank_max_length": _i("LAWS_RERANK_MAX_LENGTH", 512),
}

LAWS_BUILD_BATCH = _i("LAWS_BUILD_BATCH", 8)
LAWS_EMBED_MAX_LENGTH = _i("LAWS_EMBED_MAX_LENGTH", 1024)


# السوابق القضائية

CASES_DIR = Path(os.getenv("CASES_DIR", DATA_DIR / "cases"))
CASES_INDEX_FILE = str(os.getenv("CASES_INDEX_FILE", CASES_DIR / "legal_facts_hybrid.faiss"))
CASES_METADATA_FILE = str(os.getenv("CASES_METADATA_FILE", CASES_DIR / "legal_metadata_hybrid.pkl"))
CASES_SOURCE_JSON = str(os.getenv("CASES_SOURCE_JSON", CASES_DIR / "standard_cases.json"))

CASES_DEFAULTS = {
    "top_k": _i("CASES_TOP_K", 5),
    # تنطبق عليه ملاحظة الأداء المذكورة أعلاه.
    "hybrid_top_k": _i("CASES_HYBRID_TOP_K", 15),
    # عتبة التشابه 0.0–1.0، وتُعاد في النتيجة كنسبة مئوية.
    "threshold": _f("CASES_THRESHOLD", 0.50),
    "rerank_max_length": _i("CASES_RERANK_MAX_LENGTH", 2048),
    "encode_batch_size": _i("CASES_ENCODE_BATCH_SIZE", 32),
}


# إصدار الحكم الأولي

CLASSIFIER_DIR = Path(os.getenv("CLASSIFIER_DIR", DATA_DIR / "classifier"))

JUDGMENT_DEFAULTS = {
    # عدد المواد والسوابق المُمرَّرة إلى النموذج اللغوي كسياق.
    "top_k_laws": _i("JUDGMENT_TOP_K_LAWS", 5),
    "top_k_cases": _i("JUDGMENT_TOP_K_CASES", 3),
    "exclude_repealed_laws": _b("JUDGMENT_EXCLUDE_REPEALED", True),
    "min_law_score": _f("JUDGMENT_MIN_LAW_SCORE", 0.0),
    "case_threshold": _f("JUDGMENT_CASE_THRESHOLD", 0.45),
    "max_quote_words": _i("JUDGMENT_MAX_QUOTE_WORDS", 12),
    # المصنّف الإحصائي (SVM)؛ يعمل فقط عند توفّر ملفات joblib في data/classifier.
    "use_domain_classifier": _b("JUDGMENT_USE_CLASSIFIER", True),
    "top_k_classifier_labels": _i("JUDGMENT_TOP_K_CLASSIFIER_LABELS", 5),
    "classifier_min_score": _f("JUDGMENT_CLASSIFIER_MIN_SCORE", -1.0),
    "precedent_pool_size": _i("JUDGMENT_PRECEDENT_POOL_SIZE", 15),
    # إعادة تنظيم الوقائع قبل الاسترجاع؛ تستهلك استدعاءً إضافياً للنموذج اللغوي.
    "use_fact_reorganization": _b("JUDGMENT_USE_FACT_REORG", True),
    "show_reorganized_facts_to_llm": _b("JUDGMENT_SHOW_REORG_TO_LLM", True),
    "fact_reorg_max_tokens": _i("JUDGMENT_FACT_REORG_MAX_TOKENS", 512),
    # تمييز المواد المتشابهة لفظياً؛ تستهلك استدعاءً لكل زوج.
    "use_statute_discrimination": _b("JUDGMENT_USE_DISCRIMINATION", True),
    "confusable_overlap_threshold": _f("JUDGMENT_CONFUSABLE_OVERLAP", 0.15),
    "max_confusable_pairs": _i("JUDGMENT_MAX_CONFUSABLE_PAIRS", 3),
    "discrimination_max_tokens": _i("JUDGMENT_DISCRIMINATION_MAX_TOKENS", 400),
    # النموذج اللغوي
    "groq_model": os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
    "max_output_tokens": _i("JUDGMENT_MAX_OUTPUT_TOKENS", 3072),
    "temperature": _f("JUDGMENT_TEMPERATURE", 0.1),
}

# لا تُكتب المفاتيح في الكود إطلاقاً؛ تُقرأ من .env حصراً.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


# نماذج العقود

CONTRACTS_DIR = Path(os.getenv("CONTRACTS_DIR", DATA_DIR / "contracts"))
CONTRACTS_SOURCE_JSON = str(os.getenv(
    "CONTRACTS_SOURCE_JSON", CONTRACTS_DIR / "hammurabi_templates_flat.json"))
CONTRACTS_INDEX_FILE = str(os.getenv("CONTRACTS_INDEX_FILE", CONTRACTS_DIR / "contracts.faiss"))
CONTRACTS_METADATA_FILE = str(os.getenv(
    "CONTRACTS_METADATA_FILE", CONTRACTS_DIR / "contracts_meta.pkl"))

# القيمة الافتراضية هي نموذج المشروع نفسه (BGE-M3) تفادياً لتحميل نموذج ثانٍ
# في الذاكرة. للاعتماد على نموذج آخر:
#     CONTRACTS_EMBEDDING_MODEL=intfloat/multilingual-e5-large
# يستلزم أي تغيير هنا إعادة بناء الفهرس:
#     python scripts/build_contracts_index.py --force
CONTRACTS_EMBEDDING_MODEL = os.getenv("CONTRACTS_EMBEDDING_MODEL", EMBEDDING_MODEL)

# مزوّد النموذج اللغوي لطبقة الاقتراح:
#   groq → استدعاء API (لا يحتاج GPU؛ يخدم Llama 3.1 وغيره)
#   hf   → تحميل الأوزان محلياً بـ transformers (يحتاج GPU وHF_TOKEN للنماذج المقيّدة)
#   none → تعطيل الاقتراح كلياً، بحث دلالي صرف
CONTRACTS_LLM_BACKEND = os.getenv("CONTRACTS_LLM_BACKEND", "groq").strip().lower()

# النموذج المُحمَّل محلياً عند CONTRACTS_LLM_BACKEND=hf.
CONTRACTS_HF_MODEL = os.getenv("CONTRACTS_HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# مطلوب لتنزيل النماذج المقيّدة الوصول (Llama 3.1 من Meta مثلاً).
HF_TOKEN = os.getenv("HF_TOKEN", "") or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)

CONTRACTS_DEFAULTS = {
    # عدد نماذج العقود في قائمة المرشحين.
    "top_k": _i("CONTRACTS_TOP_K", 5),
    # عتبة تشابه cosine بين 0.0 و1.0. القيمة 0.0 تعني بلا عتبة.
    "min_score": _f("CONTRACTS_MIN_SCORE", 0.0),
    # طبقة النموذج اللغوي التي تقترح الأنسب من المرشحين؛ تستهلك استدعاءً واحداً.
    # القيمة false تعني بحثاً دلالياً صرفاً بلا أي استدعاء.
    "suggest": _b("CONTRACTS_SUGGEST", True),
    "llm_backend": CONTRACTS_LLM_BACKEND,
    # Llama 3.1 عبر Groq: لا يحتاج GPU، والمهمة هنا اختيار رقم من قائمة قصيرة.
    "groq_model": os.getenv("CONTRACTS_GROQ_MODEL", "llama-3.1-8b-instant"),
    "hf_model": CONTRACTS_HF_MODEL,
    "suggest_max_tokens": _i("CONTRACTS_SUGGEST_MAX_TOKENS", 200),
    "temperature": _f("CONTRACTS_TEMPERATURE", 0.1),
}

CONTRACTS_BUILD_BATCH = _i("CONTRACTS_BUILD_BATCH", 16)


# إقلاع الخادم

# الميزات التي تُحمَّل نماذجها عند الإقلاع بدل أول طلب.
# القيمة الفارغة (الافتراضي) تعني تحميلاً كسولاً: يقلع الخادم فوراً ويستغرق أول
# طلب لكل ميزة وقتاً أطول. مثال: WARMUP=laws,cases,search
WARMUP = [s.strip() for s in os.getenv("WARMUP", "").split(",") if s.strip()]

PORT = _i("PORT", 5000)
DEBUG = _b("DEBUG", True)


def resolve_device() -> str:
    """يحوّل DEVICE='auto' إلى قيمة فعلية."""
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def use_fp16() -> bool:
    """fp16 يعطي نتائج خاطئة أو بطيئة على الـ CPU، فيُسمح به على الـ GPU فقط."""
    return USE_FP16 and resolve_device() == "cuda"
