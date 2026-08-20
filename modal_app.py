"""
نشر المشروع على Modal.

    pip install modal && modal setup

    # مرة واحدة فقط:
    modal run modal_app.py::download_models      # ~4.6GB إلى Volume
    modal run modal_app.py::build_laws_index     # فهرس Qdrant إلى Volume
    modal run modal_app.py::build_contracts_index  # بعد رفع ملف النماذج

    # النشر:
    modal deploy modal_app.py

كل ما يُبنى مرة واحدة يُحفظ في Volumes، ولا يُعاد بناؤه عند كل تشغيل:
  monsif-hf-cache  → أوزان BGE-M3 والـ reranker
  monsif-indexes   → فهرس Qdrant للمواد + فهرس العقود
أما الفهارس الجاهزة في git (utils/*.faiss و data/cases/*) فتُحزم داخل الصورة.
"""

import os
import shutil
from pathlib import Path

import modal

APP_NAME = "monsif-back"

# "T4" أسرع بـ 5–20 مرة وبكلفة مشابهة لأن زمن التنفيذ ينكمش. None = CPU فقط.
GPU = os.environ.get("MONSIF_GPU", "T4") or None

HF_CACHE = "/cache/hf"          # Volume: أوزان HuggingFace
IDX = "/indexes"                # Volume: الفهارس المبنية
APP_DIR = "/root/app"

hf_cache_vol = modal.Volume.from_name("monsif-hf-cache", create_if_missing=True)
indexes_vol = modal.Volume.from_name("monsif-indexes", create_if_missing=True)

# GROQ_API_KEY و GOOGLE_API_KEY:
#   modal secret create monsif-secrets GROQ_API_KEY=gsk_... GOOGLE_API_KEY=...
secrets = [modal.Secret.from_name("monsif-secrets")]

# متغيرات البيئة — تقابل تماماً ما كان في .env محلياً.
ENV = {
    "HF_HOME": HF_CACHE,
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "PYTHONUNBUFFERED": "1",
    "DEBUG": "false",
    "WARMUP": "",                       # التحميل يتم في @modal.enter بدل ذلك
    "DEVICE": "cuda" if GPU else "cpu",
    # ملفات المصنّف موجودة أصلاً في utils/؛ لا حاجة لنسخها من Drive.
    "CLASSIFIER_DIR": f"{APP_DIR}/utils/classification_model",
    # فهرس Qdrant يُنسخ إلى /tmp عند الإقلاع (انظر Server.start).
    "LAWS_QDRANT_PATH": "/tmp/qdrant_db",
    "CONTRACTS_SOURCE_JSON": f"{IDX}/contracts/hammurabi_templates_flat.json",
    "CONTRACTS_INDEX_FILE": f"{IDX}/contracts/contracts.faiss",
    "CONTRACTS_METADATA_FILE": f"{IDX}/contracts/contracts_meta.pkl",
}

IGNORE = [
    "venv", "venv/**", ".git", ".git/**", "__pycache__", "**/__pycache__/**",
    "*.pyc", ".hf_cache", ".hf_cache/**", ".env", "data/laws/qdrant_db",
    "data/laws/qdrant_db/**", "*.ipynb",
]

_base = modal.Image.debian_slim(python_version="3.11")
if not GPU:
    # عجلة torch الخاصة بالـ CPU: ~200MB بدل ~2.5GB لنسخة CUDA.
    _base = _base.pip_install(
        "torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cpu"
    )

image = (
    _base
    .pip_install("hf_transfer")
    .pip_install_from_requirements("requirements.txt")
    # NLTK ينزّل stopwords عند الاستيراد؛ نخبزه في الصورة كي لا يلمس الشبكة وقت الطلب.
    .run_commands("python -c \"import nltk; nltk.download('stopwords')\"")
    .env(ENV)
    .add_local_dir(".", APP_DIR, ignore=IGNORE)
)

app = modal.App(APP_NAME)


# ═══════════════ خطوات البناء — تُنفَّذ مرة واحدة ═══════════════

@app.function(image=image, volumes={HF_CACHE: hf_cache_vol}, timeout=3600)
def download_models():
    """تنزيل أوزان BGE-M3 والـ reranker إلى Volume (~4.6GB). مرة واحدة فقط."""
    import sys

    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    from huggingface_hub import snapshot_download

    import config

    for repo in (config.EMBEDDING_MODEL, config.RERANKER_MODEL):
        print(f"→ {repo}")
        snapshot_download(
            repo_id=repo,
            ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "onnx/*", "*.ot"],
            max_workers=8,
        )
    hf_cache_vol.commit()
    print("تمت الأوزان؛ محفوظة في Volume ولن تُنزَّل ثانية.")


@app.function(
    image=image, gpu=GPU, volumes={HF_CACHE: hf_cache_vol, IDX: indexes_vol},
    secrets=secrets, timeout=3600, memory=16384,
)
def build_laws_index(force: bool = False, limit: int = 0):
    """بناء فهرس Qdrant للمواد ثم حفظه في Volume.

    على T4 يستغرق دقائق معدودة بدل 20–60 دقيقة على CPU.
    """
    import subprocess
    import sys

    target = f"{IDX}/laws/qdrant_db"
    os.makedirs(f"{IDX}/laws", exist_ok=True)
    # يُبنى محلياً على قرص الحاوية ثم يُنسخ: الكتابة المباشرة على Volume بطيئة
    # ويأخذ Qdrant قفلاً على المجلد.
    os.environ["LAWS_QDRANT_PATH"] = "/tmp/build_qdrant"

    cmd = [sys.executable, "scripts/build_laws_index.py"]
    if force:
        cmd.append("--force")
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, cwd=APP_DIR, check=True)

    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree("/tmp/build_qdrant", target)
    hf_cache_vol.commit()
    indexes_vol.commit()
    print(f"الفهرس محفوظ في Volume: {target}")


@app.function(
    image=image, gpu=GPU, volumes={HF_CACHE: hf_cache_vol, IDX: indexes_vol},
    secrets=secrets, timeout=3600,
)
def build_contracts_index(force: bool = False):
    """بناء فهرس العقود. يتطلب رفع الملف أولاً:

        modal volume put monsif-indexes hammurabi_templates_flat.json \
            /contracts/hammurabi_templates_flat.json
    """
    import subprocess
    import sys

    os.makedirs(f"{IDX}/contracts", exist_ok=True)
    cmd = [sys.executable, "scripts/build_contracts_index.py"]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, cwd=APP_DIR, check=True)
    indexes_vol.commit()


# ═══════════════ الخادم ═══════════════

@app.cls(
    image=image,
    gpu=GPU,
    volumes={HF_CACHE: hf_cache_vol, IDX: indexes_vol},
    secrets=secrets,
    cpu=2.0,
    memory=8192,
    # مدة بقاء الحاوية خاملة بعد آخر طلب. أطول = بدايات باردة أقل وكلفة أعلى.
    scaledown_window=300,
    # 0 = التصفير عند الخمول (لا كلفة بلا طلبات). 1 = جاهز دائماً وكلفة شهرية ثابتة.
    min_containers=0,
    max_containers=4,
    timeout=900,
)
class Server:
    @modal.enter()
    def start(self):
        import sys

        sys.path.insert(0, APP_DIR)
        os.chdir(APP_DIR)

        # لا يُسمح بأي اتصال بـ HuggingFace وقت الخدمة؛ كل شيء من الـ Volume.
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        # Qdrant المحلي يأخذ قفلاً حصرياً على المجلد، والـ Volume مشترك بين كل
        # الحاويات. تُنسخ نسخة محلية لكل حاوية كي تعمل عدة حاويات معاً.
        src = f"{IDX}/laws/qdrant_db"
        dst = os.environ["LAWS_QDRANT_PATH"]
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"[modal] نُسخ فهرس Qdrant → {dst}", flush=True)

        import app as flask_module

        self.web = flask_module.app

        # تحميل النماذج والفهارس هنا: لا تُعدّ الحاوية جاهزة قبل انتهاء enter،
        # فلا يصطدم أول مستخدم بزمن التحميل.
        for name, warm in (
            ("laws", "services.laws_rag"),
            ("cases", "services.cases_rag"),
            ("search", "services.law_and_jurisprudence_search"),
            ("contracts", "services.contracts_rag"),
        ):
            try:
                __import__(warm, fromlist=["warmup"]).warmup()
                print(f"[modal] جاهز: {name}", flush=True)
            except Exception as e:
                print(f"[modal] تُخطّيت ميزة «{name}»: {type(e).__name__}: {e}", flush=True)

    @modal.wsgi_app()
    def flask_app(self):
        return self.web
