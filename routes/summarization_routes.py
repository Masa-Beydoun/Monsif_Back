# routes/summarization_routes.py — تلخيص نص القضية واستخراج الحقول
from flask import Blueprint, jsonify, request

from services.summarization import IntelligentLegalPipeline

legal_summarization = Blueprint("legal_summarization", __name__)

_pipeline = None


def get_pipeline() -> IntelligentLegalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IntelligentLegalPipeline(
            enable_ai=False
        )  # تم التعديل هنا لضمان الإعدادات
    return _pipeline


@legal_summarization.route("/summarize", methods=["POST"])
def analyze_case():
    """
    POST /api/legal/summarize
    {"text": "نص القضية ..."}
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return (
            jsonify(
                {"status": "error", "error": "يرجى إرسال حقل 'text' ضمن جسم الطلب."}
            ),
            400,
        )

    raw_text = data.get("text", "")

    # الحقل يجب أن يكون نصاً (وإلا سيفشل التحليل بخطأ 500)
    if not isinstance(raw_text, str):
        return (
            jsonify({"status": "error", "error": "حقل 'text' يجب أن يكون نصاً."}),
            400,
        )

    try:
        analysis_result = get_pipeline().analyze(raw_text)

        if analysis_result.get("status") == "error":
            return jsonify(analysis_result), 400

        # ✅ التعديل تم هنا: أضفنا ["analysis"] للوصول إلى الحقول الصحيحة
        return (
            jsonify(
                {
                    "status": "success",
                    "data": {
                        "summary": analysis_result["analysis"]["extractive_summary"],
                        "extracted_fields": analysis_result["analysis"][
                            "structured_fields"
                        ],
                        "original_length": analysis_result["original_length"],
                    },
                }
            ),
            200,
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": f"حدث خطأ غير متوقع أثناء معالجة الطلب: {e}",
                }
            ),
            500,
        )
