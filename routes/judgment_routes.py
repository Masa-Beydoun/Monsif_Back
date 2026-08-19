# routes/judgment_routes.py — Part C: إصدار حكم أولي
from flask import Blueprint, jsonify, request

import config
from services import cases_rag, judgment, laws_rag

judgment_bp = Blueprint("judgment_bp", __name__)


@judgment_bp.route("/judgment/predict", methods=["POST"])
def predict():
    """
    POST /api/legal/judgment/predict
    {
      "text": "وقائع القضية ...",
      "config": {                     // اختياري — أي حقل من JudgmentConfig
        "top_k_laws": 5,
        "top_k_cases": 3,
        "case_threshold": 0.45,
        "use_fact_reorganization": true,
        "use_statute_discrimination": true,
        "use_domain_classifier": true
      }
    }

    ⚠️ هالمسار بيستدعي الـ LLM حتى 5 مرات (إعادة تنظيم + تمييز لكل زوج + الحكم)،
    وبيشغّل الـ RAG مرتين. متوقّع ياخد وقت أطول من مسارات البحث المفردة.
    """
    data = request.get_json(silent=True) or {}
    facts = (data.get("text") or "").strip()
    if not facts:
        return jsonify({"status": "error",
                        "error": "يرجى إرسال حقل 'text' غير فارغ ضمن الـ JSON"}), 400
    if len(facts) < 20:
        return jsonify({"status": "error",
                        "error": "نص الوقائع قصير جداً ولا يكفي لإصدار حكم أولي."}), 400

    overrides = data.get("config") if isinstance(data.get("config"), dict) else None

    try:
        result = judgment.predict_judgment(facts, overrides)
        if result.get("error"):
            return jsonify({"status": "error", "error": result["message"],
                            "data": result}), 502
        return jsonify({"status": "success", "data": result}), 200
    except judgment.MissingAPIKey as e:
        return jsonify({"status": "error", "error": str(e)}), 503
    except (laws_rag.IndexNotBuilt, FileNotFoundError) as e:
        return jsonify({"status": "error", "error": str(e)}), 503
    except Exception as e:
        return jsonify({"status": "error", "error": f"حدث خطأ غير متوقع: {e}"}), 500


@judgment_bp.route("/judgment/config", methods=["GET"])
def judgment_config():
    return jsonify({
        "status": "success",
        "data": {
            "defaults": config.JUDGMENT_DEFAULTS,
            "tunable_in_request": sorted(judgment.JudgmentConfig.__dataclass_fields__),
            "groq_api_key_set": bool(config.GROQ_API_KEY),
            "classifier_available": judgment.classifier_available(),
            "classifier_dir": str(config.CLASSIFIER_DIR),
            "depends_on": {
                "laws_rag_loaded": laws_rag.is_loaded(),
                "cases_rag_loaded": cases_rag.is_loaded(),
            },
        },
    }), 200
