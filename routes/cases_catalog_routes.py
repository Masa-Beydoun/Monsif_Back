# routes/cases_catalog_routes.py — تصفّح القضايا وتصنيفاتها (بدون بحث دلالي)
from flask import Blueprint, jsonify, request

from services import cases_catalog

cases_catalog_bp = Blueprint("cases_catalog_bp", __name__)


@cases_catalog_bp.route("/cases", methods=["GET"])
def list_cases():
    """
    GET /api/legal/cases
        ?q=نشل                بحث نصي في رقم القضية أو الجرائم أو معاينة النص (اختياري)
        &crime=النشل          تصفية حسب تصنيف الجريمة (اختياري)
        &year=2022            تصفية حسب سنة القرار (اختياري)
        &outcome=إدانة         تصفية حسب نتيجة الحكم (اختياري)
        &page=1&per_page=20   ترقيم صفحات اختياري

    يعيد ملخّص كل قضية مع تجميع الجرائم (التصنيفات) وقيمها المتاحة للتصفية.
    """
    args = request.args
    try:
        page = int(args.get("page", 1))
        per_page = int(args["per_page"]) if args.get("per_page") else None
    except (TypeError, ValueError) as e:
        return jsonify({"status": "error",
                        "error": f"قيمة غير صالحة لأحد البارامترات: {e}"}), 400

    try:
        data = cases_catalog.list_cases(
            q=args.get("q") or None,
            crime=args.get("crime") or None,
            year=args.get("year") or None,
            outcome=args.get("outcome") or None,
            page=page,
            per_page=per_page,
        )
        return jsonify({"status": "success", "data": data}), 200
    except FileNotFoundError as e:
        return jsonify({"status": "error",
                        "error": f"ملف القضايا غير موجود: {e}"}), 503
    except Exception as e:
        return jsonify({"status": "error",
                        "error": f"حدث خطأ غير متوقع أثناء معالجة الطلب: {e}"}), 500


@cases_catalog_bp.route("/cases/<path:case_uid>", methods=["GET"])
def single_case(case_uid: str):
    """قضية واحدة بمعرّفها (file_name أو case_number) مع كل أقسامها."""
    try:
        case = cases_catalog.get_case(case_uid)
    except FileNotFoundError as e:
        return jsonify({"status": "error",
                        "error": f"ملف القضايا غير موجود: {e}"}), 503
    except Exception as e:
        return jsonify({"status": "error",
                        "error": f"حدث خطأ غير متوقع أثناء معالجة الطلب: {e}"}), 500

    if case is None:
        return jsonify({"status": "error",
                        "error": "لم يُعثر على قضية بالمعرّف المحدد. "
                                 "استعرض القضايا المتاحة عبر GET /api/legal/cases"}), 404
    return jsonify({"status": "success", "data": case}), 200
