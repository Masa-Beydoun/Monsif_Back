# routes/contracts_routes.py — Part D: RAG نماذج العقود
from flask import Blueprint, jsonify, request

import config
from services import contracts_rag

contracts_bp = Blueprint("contracts_bp", __name__)

TUNABLE = {
    "top_k": int,
    "min_score": float,
    "suggest": bool,
    "groq_model": str,
    "suggest_max_tokens": int,
    "temperature": float,
}


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _coerce(data: dict) -> dict:
    out = {}
    for key, caster in TUNABLE.items():
        if key in data and data[key] is not None:
            out[key] = _to_bool(data[key]) if caster is bool else caster(data[key])
    return out


@contracts_bp.route("/contracts/search", methods=["POST"])
def search_contracts():
    """
    POST /api/legal/contracts/search
    {
      "text": "بدي عقد إيجار محل تجاري",
      "top_k": 5,          // اختياري
      "min_score": 0.0,    // اختياري — عتبة cosine
      "suggest": true      // اختياري — طبقة الـ LLM (بتكلّف استدعاء Groq)
    }

    بترجع قائمة المرشحين + اقتراح الأنسب. لعرض العقد كامل، ابعتي الـ doc_id
    يلي اختاره المستخدم لـ /api/legal/contracts/get.
    """
    data = request.get_json(silent=True) or {}
    query_text = (data.get("text") or "").strip()
    if not query_text:
        return jsonify({"status": "error",
                        "error": "يرجى إرسال حقل 'text' غير فارغ ضمن الـ JSON"}), 400

    try:
        params = _coerce(data)
    except (TypeError, ValueError) as e:
        return jsonify({"status": "error", "error": f"قيمة بارامتر غير صالحة: {e}"}), 400

    try:
        result = contracts_rag.search_contracts(query_text, **params)
        return jsonify({"status": "success", "data": result}), 200
    except contracts_rag.MissingAPIKey as e:
        return jsonify({"status": "error", "error": str(e)}), 503
    except FileNotFoundError as e:
        return jsonify({"status": "error", "error": str(e)}), 503
    except Exception as e:
        return jsonify({"status": "error", "error": f"حدث خطأ غير متوقع: {e}"}), 500


@contracts_bp.route("/contracts/get", methods=["POST"])
def get_contract():
    """
    POST /api/legal/contracts/get
    {"doc_id": "..."}          أو          {"subject": "عنوان العقد بالضبط"}

    doc_id أضمن — العنوان لازمه يطابق حرفياً متل ما رجع بنتيجة البحث.
    """
    data = request.get_json(silent=True) or {}
    doc_id = (data.get("doc_id") or "").strip()
    subject = (data.get("subject") or "").strip()
    if not doc_id and not subject:
        return jsonify({"status": "error",
                        "error": "يرجى إرسال 'doc_id' أو 'subject' ضمن الـ JSON"}), 400

    try:
        contract = contracts_rag.get_contract(doc_id=doc_id or None, subject=subject or None)
    except FileNotFoundError as e:
        return jsonify({"status": "error", "error": str(e)}), 503
    except Exception as e:
        return jsonify({"status": "error", "error": f"حدث خطأ غير متوقع: {e}"}), 500

    if contract is None:
        return jsonify({"status": "error",
                        "error": "ما في نموذج عقد بهالمعرّف/العنوان."}), 404
    return jsonify({"status": "success", "data": contract}), 200


@contracts_bp.route("/contracts/<path:doc_id>", methods=["GET"])
def get_contract_by_id(doc_id: str):
    """نفس /contracts/get بس بـ GET — GET /api/legal/contracts/<doc_id>"""
    try:
        contract = contracts_rag.get_contract(doc_id=doc_id)
    except FileNotFoundError as e:
        return jsonify({"status": "error", "error": str(e)}), 503
    except Exception as e:
        return jsonify({"status": "error", "error": f"حدث خطأ غير متوقع: {e}"}), 500

    if contract is None:
        return jsonify({"status": "error", "error": "ما في نموذج عقد بهالمعرّف."}), 404
    return jsonify({"status": "success", "data": contract}), 200


@contracts_bp.route("/contracts/config", methods=["GET"])
def contracts_config():
    data = {
        "defaults": config.CONTRACTS_DEFAULTS,
        "tunable_in_request": sorted(TUNABLE),
        "loaded": contracts_rag.is_loaded(),
        "embedding_model": config.CONTRACTS_EMBEDDING_MODEL,
        "index_file": config.CONTRACTS_INDEX_FILE,
    }
    # توزيع الفئات متوفر بس إذا الفهرس محمّل — ما منحمّله لمجرد طلب إعدادات
    if contracts_rag.is_loaded():
        rag = contracts_rag.get_rag()
        data["total_contracts"] = len(rag.records)
        data["categories"] = rag.categories()
    return jsonify({"status": "success", "data": data}), 200
