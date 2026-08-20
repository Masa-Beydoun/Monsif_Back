# routes/cases_rag_routes.py — Part B: RAG السوابق القضائية
from flask import Blueprint, jsonify, request

import config
from services import cases_rag

cases_rag_bp = Blueprint("cases_rag_bp", __name__)

TUNABLE = {
    "top_k": int,
    "hybrid_top_k": int,
    "threshold": float,
    "rerank_max_length": int,
}


def _coerce(data: dict) -> dict:
    out = {}
    for key, caster in TUNABLE.items():
        if key in data and data[key] is not None:
            out[key] = caster(data[key])
    return out


@cases_rag_bp.route("/cases/search", methods=["POST"])
def search_cases():
    """
    POST /api/legal/cases/search
    {
      "text": "وقائع القضية ...",
      "top_k": 5,            // اختياري
      "hybrid_top_k": 15,
      "threshold": 0.5       // 0.0–1.0
    }
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
        result = cases_rag.search_cases(query_text, **params)
        return jsonify({"status": "success", "data": result}), 200
    except FileNotFoundError as e:
        return jsonify({"status": "error", "error": str(e)}), 503
    except Exception as e:
        return jsonify({"status": "error", "error": f"حدث خطأ غير متوقع: {e}"}), 500


@cases_rag_bp.route("/cases/config", methods=["GET"])
def cases_config():
    return jsonify({
        "status": "success",
        "data": {
            "defaults": config.CASES_DEFAULTS,
            "tunable_in_request": sorted(TUNABLE),
            "loaded": cases_rag.is_loaded(),
            "index_file": config.CASES_INDEX_FILE,
        },
    }), 200
