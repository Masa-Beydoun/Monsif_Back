# routes/judgment_routes.py — إصدار الحكم الأولي
from flask import Blueprint, jsonify, request

import config
from services import cases_rag, judgment, laws_rag, llm_client

judgment_bp = Blueprint("judgment_bp", __name__)


@judgment_bp.route("/judgment/predict", methods=["POST"])
def predict():
    """
    POST /api/legal/judgment/predict
    {
      "text": "وقائع القضية ...",
      "config": {                     // اختياري: أي حقل من JudgmentConfig
        "top_k_laws": 5,
        "top_k_cases": 3,
        "case_threshold": 0.45,
        "llm_backend": "hf",          // hf | groq
        "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
        "use_fact_reorganization": true,
        "use_statute_discrimination": true,
        "use_domain_classifier": true
      }
    }

    ينفّذ هذا المسار حتى خمسة استدعاءات للنموذج اللغوي ودورتَي استرجاع،
    لذلك يستغرق وقتاً أطول من مسارات البحث المفردة.

    بنية data موثّقة في README؛ الحقول الثابتة التي تعتمد عليها الواجهة:
      ok · outcome · final_charge · candidate_charges[] · reasoning ·
      verification · evidence{statutes, cases, classifier_candidates,
      reorganized_facts, discrimination} · meta
    """
    data = request.get_json(silent=True) or {}
    facts = (data.get("text") or "").strip()
    if not facts:
        return jsonify({"status": "error",
                        "error": "يرجى إرسال حقل 'text' غير فارغ ضمن جسم الطلب."}), 400
    if len(facts) < 20:
        return jsonify({"status": "error",
                        "error": "نص الوقائع قصير جداً ولا يكفي لإصدار حكم أولي."}), 400

    overrides = data.get("config") if isinstance(data.get("config"), dict) else None

    try:
        result = judgment.predict_judgment(facts, overrides)
        if not result.get("ok"):
            return jsonify({"status": "error", "error": result.get("message"),
                            "data": result}), 502
        return jsonify({"status": "success", "data": result}), 200
    except judgment.MissingAPIKey as e:
        return jsonify({"status": "error", "error": str(e)}), 503
    except (laws_rag.IndexNotBuilt, FileNotFoundError) as e:
        return jsonify({"status": "error", "error": str(e)}), 503
    except judgment.LLMError as e:
        return jsonify({"status": "error", "error": str(e)}), 502
    except Exception as e:
        return jsonify({"status": "error",
                        "error": f"حدث خطأ غير متوقع أثناء معالجة الطلب: {e}"}), 500


@judgment_bp.route("/judgment/config", methods=["GET"])
def judgment_config():
    """الإعدادات الافتراضية وحالة كل ما تعتمد عليه الميزة.

    تستدعيه الواجهة عند فتح الشاشة لتعرف هل الميزة جاهزة قبل أن يكتب المستخدم
    وقائع ويصطدم بخطأ بعد دقيقة انتظار.
    """
    llm = llm_client.status()
    depends_on = {
        "laws_rag_loaded": laws_rag.is_loaded(),
        "cases_rag_loaded": cases_rag.is_loaded(),
        "classifier_available": judgment.classifier_available(),
        "llm_key_set": llm["configured"],
    }
    return jsonify({
        "status": "success",
        "data": {
            "ready": llm["configured"],
            "blocking_reason": None if llm["configured"] else
                               f"مفتاح النموذج اللغوي غير مضبوط لواجهة «{llm['active_backend']}».",
            "defaults": config.JUDGMENT_DEFAULTS,
            "tunable_in_request": sorted(judgment.JudgmentConfig.__dataclass_fields__),
            "outcomes": judgment.OUTCOMES,
            "llm": llm,
            "classifier_dir": str(config.CLASSIFIER_DIR),
            "depends_on": depends_on,
        },
    }), 200


@judgment_bp.route("/judgment/llm/ping", methods=["GET", "POST"])
def llm_ping():
    """استدعاء تجريبي قصير للنموذج اللغوي.

    يفصل عطل المفتاح/النموذج عن عطل الاسترجاع، فلا يلزم انتظار مسار الحكم
    الكامل لاكتشاف أن الرمز خاطئ.
    """
    data = request.get_json(silent=True) or {}
    result = llm_client.ping(backend=data.get("backend"), model=data.get("model"))
    return jsonify({"status": "success" if result["ok"] else "error",
                    "data": result}), (200 if result["ok"] else 502)
