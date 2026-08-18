# routes/legal_search_routes.py
from flask import Blueprint, request, jsonify
from services.law_and_jurisprudence_search import search_legal, warmup

# 1. إنشاء Blueprint
legal_search_bp = Blueprint('legal_search_bp', __name__)

# 2. تحميل نماذج الـ Embedding/Re-ranker والفهارس مرة واحدة عند إقلاع السيرفر
#    (وليس عند كل طلب — هذا هو نفس مبدأ تهيئة pipeline في legal_routes.py)
warmup()

VALID_MODES = (None, 'articles', 'jurisprudence', 'both')


# 3. تعريف الـ Endpoint
@legal_search_bp.route('/search', methods=['POST'])
def search_case():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({
            "status": "error",
            "error": "يرجى إرسال حقل 'text' ضمن الـ JSON"
        }), 400

    query_text = (data.get('text') or '').strip()
    if not query_text:
        return jsonify({
            "status": "error",
            "error": "النص المدخل فارغ."
        }), 400

    top_k = data.get('top_k', 3)
    mode = data.get('mode')  # اختياري: 'articles' | 'jurisprudence' | 'both' | None (اكتشاف تلقائي)

    if mode not in VALID_MODES:
        return jsonify({
            "status": "error",
            "error": "قيمة 'mode' غير صالحة. القيم المسموحة: articles, jurisprudence, both."
        }), 400

    try:
        result = search_legal(query_text, top_k=top_k, mode=mode)

        return jsonify({
            "status": "success",
            "data": result
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"حدث خطأ غير متوقع: {str(e)}"
        }), 500