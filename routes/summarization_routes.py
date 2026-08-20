# routes/summarization_routes.py — تلخيص نص القضية واستخراج الحقول
from flask import Blueprint, jsonify, request

from services.summarization import IntelligentLegalPipeline

legal_summarization = Blueprint('legal_summarization', __name__)

# تهيئة كسولة: أول طلب يجهّز الـ Pipeline ثم يُعاد استخدامها.
_pipeline = None


def get_pipeline() -> IntelligentLegalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IntelligentLegalPipeline()
    return _pipeline


@legal_summarization.route('/summarize', methods=['POST'])
def analyze_case():
    """
    POST /api/legal/summarize
    {"text": "نص القضية ..."}
    """
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({
            "status": "error",
            "error": "يرجى إرسال حقل 'text' ضمن جسم الطلب."
        }), 400

    raw_text = data.get('text', '')

    try:
        analysis_result = get_pipeline().analyze(raw_text)

        if analysis_result.get("status") == "error":
            return jsonify(analysis_result), 400

        return jsonify({
            "status": "success",
            "data": {
                "summary": analysis_result["summary"],
                "extracted_fields": analysis_result["structured_fields"],
                "original_length": analysis_result["original_length"]
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"حدث خطأ غير متوقع أثناء معالجة الطلب: {e}"
        }), 500
