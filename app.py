import config
from flask import Flask, jsonify
from flask_cors import CORS

from routes.cases_rag_routes import cases_rag_bp
from routes.judgment_routes import judgment_bp
from routes.law_and_jurisprudence_search_routes import legal_search_bp
from routes.classification_routes import classification_bp
from routes.laws_rag_routes import laws_rag_bp
from routes.summarization_routes import legal_summarization
from services import model_registry

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # حتى العربي يرجع مقروء مش \uXXXX
app.json.ensure_ascii = False
CORS(app)

app.register_blueprint(legal_summarization, url_prefix="/api/legal")  # /summarize
app.register_blueprint(legal_search_bp, url_prefix="/api/legal")  # /search
app.register_blueprint(laws_rag_bp, url_prefix="/api/legal")  # /laws/search
app.register_blueprint(cases_rag_bp, url_prefix="/api/legal")  # /cases/search
app.register_blueprint(judgment_bp, url_prefix="/api/legal")  # /judgment/predict

app.register_blueprint(classification_bp, url_prefix="/api/legal")


@app.route("/")
def home():
    return jsonify(
        {
            "message": "Server is running successfully!",
            "endpoints": {
                "POST /api/legal/summarize": "تلخيص نص قضية واستخراج الحقول",
                "POST /api/legal/search": "بحث المواد + الاجتهادات (النظام القديم)",
                "POST /api/legal/laws/search": "RAG المواد القانونية",
                "POST /api/legal/cases/search": "RAG السوابق القضائية",
                "POST /api/legal/judgment/predict": "إصدار حكم أولي (يعتمد على الاتنين فوق)",
                "GET  /api/health": "حالة النماذج المحمّلة",
            },
        }
    )


@app.route("/api/health")
def health():
    """شو محمّل هلق — مفيد لتعرفي إذا أول طلب رح يكون بطيء."""
    from services import cases_rag, laws_rag

    return jsonify(
        {
            "status": "ok",
            "models": model_registry.status(),
            "features": {
                "laws_rag_loaded": laws_rag.is_loaded(),
                "cases_rag_loaded": cases_rag.is_loaded(),
            },
            "warmup_on_start": config.WARMUP,
        }
    )


def _warmup() -> None:
    """تحميل مسبق اختياري حسب WARMUP بملف .env.

    فاضي (الافتراضي) = تحميل كسول: السيرفر بيقلع فوراً، وكل ميزة بتحمّل نماذجها
    أول ما توصلها أول طلب. حطّي WARMUP=laws,cases إذا بدك أول طلب يكون سريع
    وما بتهمك مدة الإقلاع.
    """
    if not config.WARMUP:
        print("[app] تحميل كسول — النماذج بتنحمّل عند أول طلب لكل ميزة.", flush=True)
        return

    for feature in config.WARMUP:
        try:
            if feature == "laws":
                from services import laws_rag

                laws_rag.warmup()
            elif feature == "cases":
                from services import cases_rag

                cases_rag.warmup()
            elif feature == "search":
                from services.law_and_jurisprudence_search import warmup

                warmup()
            else:
                print(f"[app] ميزة غير معروفة بـ WARMUP: {feature}", flush=True)
        except Exception as e:
            print(f"[app] ✗ فشل تسخين «{feature}»: {e}", flush=True)


if __name__ == "__main__":
    _warmup()
    # ⚠️ use_reloader=False مهم: الـ reloader بيشغّل العملية مرتين، يعني كل نموذج
    #    بينحمّل مرتين، وفهرس Qdrant المحلي بيرمي خطأ قفل (already locked).
    app.run(debug=config.DEBUG, port=config.PORT, use_reloader=False, threaded=True)
