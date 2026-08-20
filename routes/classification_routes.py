# routes/classification_routes.py — تصنيف الجرائم انطلاقاً من الوقائع
from flask import Blueprint, jsonify, request

from services.classification_service import predict_crimes

classification_bp = Blueprint("classification", __name__)


@classification_bp.route("/classify", methods=["POST"])
def classify_case():
    """
    POST /api/legal/classify
    {"facts": "وقائع القضية ..."}
    """
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "status": "error",
            "error": "يجب أن يكون جسم الطلب بصيغة JSON."
        }), 400

    facts = data.get("facts")

    if facts is None:
        return jsonify({
            "status": "error",
            "error": "حقل 'facts' مطلوب."
        }), 400

    if not isinstance(facts, str):
        return jsonify({
            "status": "error",
            "error": "حقل 'facts' يجب أن يكون نصاً."
        }), 400

    if not facts.strip():
        return jsonify({
            "status": "error",
            "error": "حقل 'facts' لا يجوز أن يكون فارغاً."
        }), 400

    try:
        result = predict_crimes(facts)
        return jsonify({"status": "success", "data": result}), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"تعذّر إتمام التصنيف: {e}"
        }), 500
