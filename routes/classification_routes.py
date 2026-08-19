from flask import Blueprint, request, jsonify
from services.classification_service import predict_crimes

classification_bp = Blueprint(
    "classification",
    __name__
)

@classification_bp.route("/classify", methods=["POST"])
def classify_case():

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "error": "Request body must be JSON."
        }), 400

    facts = data.get("facts")

    if facts is None:
        return jsonify({
            "success": False,
            "error": "The 'facts' field is required."
        }), 400

    if not isinstance(facts, str):
        return jsonify({
            "success": False,
            "error": "The 'facts' field must be a string."
        }), 400

    if not facts.strip():
        return jsonify({
            "success": False,
            "error": "The 'facts' field cannot be empty."
        }), 400

    try:

        result = predict_crimes(facts)

        return jsonify({
            "success": True,
            **result
        }), 200

    except Exception as e:

        return jsonify({
            "success": False,
            "error": f"Classification failed: {str(e)}"
        }), 500