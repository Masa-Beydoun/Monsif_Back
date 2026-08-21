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

# طبقة اقتراح العقد: "groq" استدعاء API، أو "hf" أوزان محلية (Llama 3.1).
# الافتراضي أوزان محلية؛ MONSIF_CONTRACTS_BACKEND=groq يعيدها إلى استدعاء API.
CONTRACTS_BACKEND = os.environ.get("MONSIF_CONTRACTS_BACKEND", "hf").strip().lower()

# "T4" أسرع بـ 5–20 مرة من الـ CPU وبكلفة مشابهة لأن زمن التنفيذ ينكمش.
# لكن Llama-3.1-8B بدقة fp16 يشغل ~16GB، وذاكرة T4 ست عشرة أيضاً فلا يبقى
# متسع للتنشيطات — لذلك يقفز الافتراضي إلى L4 (24GB) عند اختيار المزوّد hf.
_DEFAULT_GPU = "L4" if CONTRACTS_BACKEND == "hf" else "T4"
GPU = os.environ.get("MONSIF_GPU", _DEFAULT_GPU) or None

HF_CACHE = "/cache/hf"          # Volume: أوزان HuggingFace
IDX = "/indexes"                # Volume: الفهارس المبنية
APP_DIR = "/root/app"

hf_cache_vol = modal.Volume.from_name("monsif-hf-cache", create_if_missing=True)
indexes_vol = modal.Volume.from_name("monsif-indexes", create_if_missing=True)

# سرّ واحد يحمل كل المفاتيح:
#   GROQ_API_KEY  — لمزوّد groq
#   GOOGLE_API_KEY— لتفكيك الاستعلامات الطويلة (اختياري)
#   HF_TOKEN      — لأوزان Llama المقيّدة الوصول عند مزوّد hf
#
#   modal secret create monsif-secrets --force GROQ_API_KEY=gsk_... GOOGLE_API_KEY= HF_TOKEN=hf_...
#
# قائمة الأسرار ثابتة عمداً ولا تعتمد على أي شرط: يُعاد تنفيذ هذا الملف داخل
# الحاوية أيضاً، فلو اختلف عدد الكائنات بين المحلي والبعيد فشل التشغيل بـ
# «Function has N dependencies but container got M object ids».
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
    # ملف النماذج موجود في git، أما الفهرس فيُبنى/يُستورد مرة واحدة إلى Volume.
    "CONTRACTS_INDEX_FILE": f"{IDX}/contracts/contracts.faiss",
    "CONTRACTS_METADATA_FILE": f"{IDX}/contracts/contracts_meta.pkl",
    # الفهرس المستورد من النوتبوك مبني بـ e5-large؛ أي اختلاف هنا = نتائج عشوائية.
    "CONTRACTS_EMBEDDING_MODEL": os.environ.get(
        "MONSIF_CONTRACTS_MODEL", "intfloat/multilingual-e5-large"),
    "CONTRACTS_LLM_BACKEND": CONTRACTS_BACKEND,
    "CONTRACTS_GROQ_MODEL": "llama-3.1-8b-instant",
    "CONTRACTS_HF_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
    # الميزات التي تُحمَّل عند الإقلاع؛ الباقي يبقى معطَّلاً بلا كلفة.
    "MONSIF_WARM": os.environ.get("MONSIF_WARM", "laws,cases,search,contracts"),
    # تُخبز هذه أيضاً حتى يقرأ الملفُ داخلَ الحاوية القيمَ نفسها التي قُرئت
    # محلياً وقت النشر، فلا يختلف تقييم الوحدة بين الجهتين.
    "MONSIF_CONTRACTS_BACKEND": CONTRACTS_BACKEND,
    "MONSIF_GPU": GPU or "",
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

def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


app = modal.App(APP_NAME)


# ═══════════════ خطوات البناء — تُنفَّذ مرة واحدة ═══════════════

# secrets هنا ضرورية: نماذج Llama مقيّدة الوصول ولا تُنزَّل بلا HF_TOKEN.
@app.function(image=image, volumes={HF_CACHE: hf_cache_vol},
              secrets=secrets, timeout=7200)
def download_models():
    """تنزيل أوزان النماذج إلى Volume. مرة واحدة لكل نموذج.

    BGE-M3 والـ reranker وe5-large ≈ 6.8GB، ويُضاف Llama-3.1-8B (~16GB) عند
    اختيار المزوّد hf.
    """
    import sys

    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    from huggingface_hub import snapshot_download

    import config

    repos = [config.EMBEDDING_MODEL, config.RERANKER_MODEL]
    # نموذج العقود منفصل عن نموذج المشروع (e5-large بدل BGE-M3) لأن الفهرس
    # الجاهز مبني به؛ إعادة بنائه بـ BGE-M3 تعني إعادة الترميز من الصفر.
    if config.CONTRACTS_EMBEDDING_MODEL not in repos:
        repos.append(config.CONTRACTS_EMBEDDING_MODEL)
    # الأوزان المحلية للنموذج اللغوي تُنزَّل فقط عند اختيار مزوّد hf (~16GB).
    if config.CONTRACTS_LLM_BACKEND == "hf":
        repos.append(config.CONTRACTS_HF_MODEL)

    # النماذج المقيّدة تفشل بـ 401/403 بلا توكن. الفشل هنا صريح ومبكر أفضل من
    # اكتشافه لاحقاً على شكل «تعذّر الاتصال بـ huggingface.co» داخل الخادم.
    gated = [r for r in repos if r.lower().startswith(("meta-llama/", "mistralai/"))]
    if gated and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN غير موجود داخل الحاوية، والنماذج التالية مقيّدة الوصول: "
            + ", ".join(gated) + "\n"
            "  أضِف التوكن إلى السرّ نفسه ثم أعد المحاولة:\n"
            "    modal secret create monsif-secrets --force "
            "GROQ_API_KEY=... GOOGLE_API_KEY= HF_TOKEN=hf_...\n"
            "    modal run modal_app.py::download_models"
        )

    print(f"المطلوب تنزيله: {', '.join(repos)}\n")
    for repo in repos:
        print(f"→ {repo}")
        snapshot_download(
            repo_id=repo,
            # original/ في مستودعات Llama نسخة ثانية من الأوزان بصيغة .pth؛
            # تجاهلها يوفّر ~16GB تنزيلاً وتخزيناً بلا أي أثر على التشغيل.
            ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "onnx/*", "*.ot",
                             "original/*", "*.pth"],
            max_workers=8,
            token=os.environ.get("HF_TOKEN") or None,
        )
    hf_cache_vol.commit()

    # جرد ما استقر فعلاً في الـ Volume — يكشف أي تنزيل ناقص فوراً.
    hub = os.path.join(HF_CACHE, "hub")
    present = sorted(d for d in os.listdir(hub)
                     if d.startswith("models--")) if os.path.isdir(hub) else []
    print("\nالموجود في الـ Volume الآن:")
    for d in present:
        print(f"  {d}")

    missing = [r for r in repos
               if "models--" + r.replace("/", "--") not in present]
    if missing:
        raise RuntimeError("لم يكتمل تنزيل: " + ", ".join(missing))
    print("\nتمت الأوزان؛ محفوظة في Volume ولن تُنزَّل ثانية.")


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


@app.function(image=image, volumes={IDX: indexes_vol}, timeout=900)
def import_contracts_index(force: bool = False):
    """استيراد فهرس العقود الجاهز من النوتبوك — بلا تحميل نموذج وبلا إعادة ترميز.

    ارفعي الملفين أولاً إلى الـ Volume:

        modal volume put monsif-indexes contracts.index    /contracts/contracts.index
        modal volume put monsif-indexes contracts_df.pkl   /contracts/contracts_df.pkl

    ثم:  modal run modal_app.py::import_contracts_index
    """
    import subprocess
    import sys

    src_dir = f"{IDX}/contracts"
    os.makedirs(src_dir, exist_ok=True)

    raw_index = f"{src_dir}/contracts.index"
    raw_pickle = f"{src_dir}/contracts_df.pkl"
    for f in (raw_index, raw_pickle):
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"لم يُعثر على {f}. ارفعي الملفين إلى الـ Volume أولاً "
                f"(انظري docstring هذه الدالة)."
            )

    cmd = [sys.executable, "scripts/import_contracts_index.py",
           "--index", raw_index, "--pickle", raw_pickle]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, cwd=APP_DIR, check=True)
    indexes_vol.commit()
    print("فهرس العقود جاهز في الـ Volume.")


# ═══════════════ الخادم ═══════════════

@app.cls(
    image=image,
    gpu=GPU,
    volumes={HF_CACHE: hf_cache_vol, IDX: indexes_vol},
    secrets=secrets,
    cpu=2.0,
    # ثلاثة نماذج قد تجتمع في الذاكرة: BGE-M3 (نسختان) + reranker + e5-large
    # الخاص بالعقود ≈ 9GB. الفوترة على المستهلك فعلياً، والرقم هنا للجدولة.
    memory=16384,
    # مدة بقاء الحاوية خاملة بعد آخر طلب (2–1200 ثانية).
    # أوزان Llama ~16GB وتحميلها يستغرق دقيقتين أو ثلاثاً، فيوم العرض:
    #     MONSIF_SCALEDOWN=1200  → تبقى الحاوية حيّة ٢٠ دقيقة بعد آخر طلب
    scaledown_window=_i("MONSIF_SCALEDOWN", 300),
    # 0 = التصفير عند الخمول (لا كلفة بلا طلبات، لكن أول طلب بارد).
    # 1 = جاهزة دائماً بلا أي انتظار، وتُحتسب بالساعة طوال الشهر.
    min_containers=_i("MONSIF_MIN_CONTAINERS", 0),
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
        # MONSIF_WARM=contracts يقلع بميزة واحدة فقط — أسرع وأرخص أثناء التجربة.
        wanted = [f.strip() for f in os.environ.get(
            "MONSIF_WARM", "laws,cases,search,contracts").split(",") if f.strip()]

        for name, warm in (
            ("laws", "services.laws_rag"),
            ("cases", "services.cases_rag"),
            ("search", "services.law_and_jurisprudence_search"),
            ("contracts", "services.contracts_rag"),
        ):
            if name not in wanted:
                continue
            try:
                mod = __import__(warm, fromlist=["warmup"])
                mod.warmup()
                # auto_load للعقود يقرأ الفهرس فقط؛ نموذج الترميز يُحمَّل عند أول
                # بحث. نحمّله هنا كي لا يدفع أول مستخدم ثمن التحميل.
                if name == "contracts":
                    mod._get_encoder()
                    # أوزان Llama ~16GB؛ نقلها إلى الـ GPU يستغرق دقيقة أو أكثر،
                    # فتُحمَّل هنا كي لا تقع على أول طلب اقتراح.
                    mod.warmup_llm()
                print(f"[modal] جاهز: {name}", flush=True)
            except Exception as e:
                print(f"[modal] تُخطّيت ميزة «{name}»: {type(e).__name__}: {e}", flush=True)

    @modal.wsgi_app()
    def flask_app(self):
        return self.web
