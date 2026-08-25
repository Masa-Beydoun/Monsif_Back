import config
from flask import Flask, jsonify
from flask_cors import CORS

from routes.cases_rag_routes import cases_rag_bp
from routes.judgment_routes import judgment_bp
from routes.law_and_jurisprudence_search_routes import legal_search_bp
from routes.classification_routes import classification_bp
from routes.contracts_routes import contracts_bp
from routes.laws_catalog_routes import laws_catalog_bp
from routes.laws_rag_routes import laws_rag_bp
from routes.summarization_routes import legal_summarization
from services import model_registry

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # إرجاع النص العربي كما هو بدل \uXXXX
app.json.ensure_ascii = False
CORS(app)

app.register_blueprint(legal_summarization, url_prefix="/api/legal")  # /summarize
app.register_blueprint(legal_search_bp, url_prefix="/api/legal")  # /search
app.register_blueprint(laws_rag_bp, url_prefix="/api/legal")  # /laws/search
app.register_blueprint(laws_catalog_bp, url_prefix="/api/legal")  # /laws · /laws/<id>/articles
app.register_blueprint(cases_rag_bp, url_prefix="/api/legal")  # /cases/search
app.register_blueprint(judgment_bp, url_prefix="/api/legal")  # /judgment/predict
app.register_blueprint(contracts_bp, url_prefix="/api/legal")  # /contracts/search

app.register_blueprint(classification_bp, url_prefix="/api/legal")


@app.route("/")
def home():
    return jsonify(
        {
            "message": "Server is running successfully!",
            "endpoints": {
                "POST /api/legal/summarize": "تلخيص نص قضية واستخراج الحقول",
                "POST /api/legal/search": "بحث المواد + الاجتهادات (النظام القديم)",
                "GET  /api/legal/laws": "قائمة القوانين مع عدد المواد والتصنيفات",
                "GET  /api/legal/laws/<law_id>/articles": "مواد قانون معيّن مجمَّعة ضمن تصنيفاتها",
                "GET  /api/legal/laws/article/<article_id>": "مادة واحدة بكل معلوماتها",
                "POST /api/legal/laws/search": "RAG المواد القانونية",
                "POST /api/legal/cases/search": "RAG السوابق القضائية",
                "POST /api/legal/judgment/predict": "إصدار حكم أولي (يعتمد على الاتنين فوق)",
                "GET  /api/legal/judgment/config": "جاهزية الميزة + إعداداتها الافتراضية",
                "GET  /api/legal/judgment/llm/ping": "فحص سريع لمفتاح النموذج اللغوي",
                "GET  /api/legal/contracts": "تصفّح كل نماذج العقود مع فلترة بالفئة/نص",
                "GET  /api/legal/contracts/<doc_id>": "عرض نموذج عقد كامل بالـ doc_id",
                "POST /api/legal/contracts/search": "RAG نماذج العقود + اقتراح الأنسب",
                "POST /api/legal/contracts/get": "عرض نموذج عقد كامل بالـ doc_id",
                "GET  /api/health": "حالة النماذج المحمّلة",
            },
        }
    )


@app.route("/api/health")
def health():
    """النماذج المحمّلة حالياً."""
    from services import cases_rag, contracts_rag, laws_rag

    return jsonify(
        {
            "status": "ok",
            "models": model_registry.status(),
            "features": {
                "laws_rag_loaded": laws_rag.is_loaded(),
                "cases_rag_loaded": cases_rag.is_loaded(),
                "contracts_rag_loaded": contracts_rag.is_loaded(),
            },
            "warmup_on_start": config.WARMUP,
        }
    )


def _warmup() -> None:
    """تحميل مسبق اختياري وفق قيمة WARMUP في ملف .env.

    القيمة الفارغة (الافتراضي) تعني تحميلاً كسولاً: يقلع الخادم فوراً وتُحمَّل
    نماذج كل ميزة عند أول طلب يصلها.
    """
    if not config.WARMUP:
        print("[app] تحميل كسول: تُحمَّل النماذج عند أول طلب لكل ميزة.", flush=True)
        return

    for feature in config.WARMUP:
        try:
            if feature == "laws":
                from services import laws_rag

                laws_rag.warmup()
            elif feature == "cases":
                from services import cases_rag

                cases_rag.warmup()
            elif feature == "contracts":
                from services import contracts_rag

                contracts_rag.warmup()
            elif feature == "search":
                from services.law_and_jurisprudence_search import warmup

                warmup()
            else:
                print(f"[app] ميزة غير معروفة في WARMUP: {feature}", flush=True)
        except Exception as e:
            print(f"[app] فشل التحميل المسبق لميزة «{feature}»: {e}", flush=True)


if __name__ == "__main__":
    _warmup()
    # use_reloader=False ضروري: الـ reloader يشغّل العملية مرتين، فتُحمَّل النماذج
    # مرتين ويرمي فهرس Qdrant المحلي خطأ قفل (already locked).
    app.run(debug=config.DEBUG, port=config.PORT, use_reloader=False, threaded=True)
