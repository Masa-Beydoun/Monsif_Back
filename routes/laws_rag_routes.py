# routes/laws_rag_routes.py — الاسترجاع الدلالي للمواد القانونية
from flask import Blueprint, jsonify, request

import config
from services import laws_rag

laws_rag_bp = Blueprint("laws_rag_bp", __name__)

# البارامترات المسموح تعديلها من جسم الطلب؛ ما عداها يُتجاهل.
TUNABLE = {
    "top_n": int,
    "hybrid_top_k": int,
    "exclude_repealed": bool,
    "min_score": float,
    "with_dependencies": bool,
    "dep_depth": int,
    "dep_max": int,
    "dep_dedupe_vs_hits": bool,
    "decompose": bool,
    "rerank_max_length": int,
}


def _coerce(data: dict) -> dict:
    """يقرأ البارامترات المسموحة من جسم الطلب ويحوّلها لأنواعها الصحيحة."""
    out = {}
    for key, caster in TUNABLE.items():
        if key not in data or data[key] is None:
            continue
        v = data[key]
        if caster is bool:
            out[key] = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes")
        else:
            out[key] = caster(v)
    return out


@laws_rag_bp.route("/laws/search", methods=["POST"])
def search_laws():
    """
    POST /api/legal/laws/search
    {
      "text": "السرقة الموصوفة",
      "top_n": 7,                 // جميع البارامترات اختيارية
      "hybrid_top_k": 15,
      "exclude_repealed": true,
      "min_score": 0.3,
      "with_dependencies": true,
      "dep_depth": 2,
      "dep_max": 12,
      "decompose": false
    }
    """
    data = request.get_json(silent=True) or {}
    query_text = (data.get("text") or "").strip()
    if not query_text:
        return jsonify({"status": "error",
                        "error": "يرجى إرسال حقل 'text' غير فارغ ضمن جسم الطلب."}), 400

    try:
        params = _coerce(data)
    except (TypeError, ValueError) as e:
        return jsonify({"status": "error",
                        "error": f"قيمة غير صالحة لأحد البارامترات: {e}"}), 400

    try:
        result = laws_rag.search_laws(query_text, **params)
        return jsonify({"status": "success", "data": result}), 200
    except (laws_rag.IndexNotBuilt, FileNotFoundError) as e:
        return jsonify({"status": "error", "error": str(e)}), 503
    except Exception as e:
        return jsonify({"status": "error",
                        "error": f"حدث خطأ غير متوقع أثناء معالجة الطلب: {e}"}), 500


@laws_rag_bp.route("/laws/config", methods=["GET"])
def laws_config():
    """القيم الافتراضية الحالية وأسماء البارامترات المسموح إرسالها في الطلب."""
    return jsonify({
        "status": "success",
        "data": {
            "defaults": config.LAWS_DEFAULTS,
            "tunable_in_request": sorted(TUNABLE),
            "loaded": laws_rag.is_loaded(),
            "corpus_file": str(config.LAWS_CORPUS_FILE),
            "index_path": config.LAWS_QDRANT_PATH,
        },
    }), 200
