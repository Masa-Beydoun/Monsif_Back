# routes/legal_routes.py
from flask import Blueprint, request, jsonify
from services.summarization import IntelligentLegalPipeline

# 1. إنشاء Blueprint
legal_summarization = Blueprint('legal_summarization', __name__)

# 2. تهيئة الـ Pipeline (كسولة — أول طلب بيجهّزها، وبعدين بتنعاد استخدامها)
_pipeline = None


def get_pipeline() -> IntelligentLegalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IntelligentLegalPipeline()
    return _pipeline

# 3. تعريف الـ Endpoints
@legal_summarization.route('/summarize', methods=['POST'])
def analyze_case():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({
            "status": "error",
            "error": "يرجى إرسال حقل 'text' ضمن الـ JSON"
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
            "error": f"حدث خطأ غير متوقع: {str(e)}"
        }), 500