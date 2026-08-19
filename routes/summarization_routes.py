# routes/legal_routes.py
from flask import Blueprint, request, jsonify
from services.summarization import IntelligentLegalPipeline

legal_summarization = Blueprint('legal_summarization', __name__)

pipeline = IntelligentLegalPipeline()

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
        analysis_result = pipeline.analyze(raw_text)

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