# routes/laws_catalog_routes.py — تصفّح القوانين وموادها (بدون بحث دلالي)
from flask import Blueprint, jsonify, request

from services import laws_catalog

laws_catalog_bp = Blueprint("laws_catalog_bp", __name__)


def _to_bool(v, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@laws_catalog_bp.route("/laws", methods=["GET"])
def list_laws():
    """
    GET /api/legal/laws
        ?q=عقوبات            بحث نصي في اسم القانون (اختياري)
        &category=تشريعات العفو العام   تصفية حسب تصنيف القانون (اختياري)

    يعيد كل القوانين مع عدد موادها وتصنيفاتها وتوزّع حالات المواد.
    """
    q = request.args.get("q") or None
    category = request.args.get("category")

    try:
        data = laws_catalog.list_laws(q=q, category=category)
        return jsonify({"status": "success", "data": data}), 200
    except FileNotFoundError as e:
        return jsonify({"status": "error",
                        "error": f"ملف المواد القانونية غير موجود: {e}"}), 503
    except Exception as e:
        return jsonify({"status": "error",
                        "error": f"حدث خطأ غير متوقع أثناء معالجة الطلب: {e}"}), 500


@laws_catalog_bp.route("/laws/<path:law_id>/articles", methods=["GET"])
def law_articles(law_id: str):
    """
    GET /api/legal/laws/<law_id>/articles
        ?grouped=true        تجميع المواد ضمن تصنيفاتها الداخلية (الافتراضي)
        &include_body=true   تضمين نص المادة (الافتراضي؛ false يصغّر الرد كثيراً)
        &status=ملغاة        تصفية حسب حالة المادة
        &category=<مسار>     تصفية حسب تصنيف داخلي بعينه
        &q=سرقة              بحث نصي في رقم المادة أو نصها
        &page=1&per_page=50  ترقيم صفحات اختياري

    معرّف القانون يحتوي مسافات وحروفاً عربية — استعمل encodeURIComponent.
    """
    return _articles_response(law_id)


@laws_catalog_bp.route("/laws/articles", methods=["GET"])
def law_articles_by_query():
    """مكافئ للمسار السابق مع تمرير المعرّف كوسيط: /laws/articles?law_id=..."""
    law_id = request.args.get("law_id") or ""
    if not law_id.strip():
        return jsonify({"status": "error",
                        "error": "يرجى تمرير 'law_id' ضمن وسائط الاستعلام."}), 400
    return _articles_response(law_id)


def _articles_response(law_id: str):
    args = request.args
    try:
        page = int(args.get("page", 1))
        per_page = int(args["per_page"]) if args.get("per_page") else None
    except (TypeError, ValueError) as e:
        return jsonify({"status": "error",
                        "error": f"قيمة غير صالحة لأحد البارامترات: {e}"}), 400

    try:
        data = laws_catalog.get_law_articles(
            law_id,
            grouped=_to_bool(args.get("grouped"), True),
            include_body=_to_bool(args.get("include_body"), True),
            status=args.get("status") or None,
            category=args.get("category"),
            q=args.get("q") or None,
            page=page,
            per_page=per_page,
        )
    except FileNotFoundError as e:
        return jsonify({"status": "error",
                        "error": f"ملف المواد القانونية غير موجود: {e}"}), 503
    except Exception as e:
        return jsonify({"status": "error",
                        "error": f"حدث خطأ غير متوقع أثناء معالجة الطلب: {e}"}), 500

    if data is None:
        return jsonify({"status": "error",
                        "error": "لم يُعثر على قانون بالمعرّف المحدد. "
                                 "استعرض القوانين المتاحة عبر GET /api/legal/laws"}), 404
    return jsonify({"status": "success", "data": data}), 200


@laws_catalog_bp.route("/laws/article/<path:article_id>", methods=["GET"])
def single_article(article_id: str):
    """مادة واحدة بمعرّفها الكامل — مفيد لفتح المواد المرتبطة من حقل dependencies."""
    try:
        article = laws_catalog.get_article(article_id)
    except FileNotFoundError as e:
        return jsonify({"status": "error",
                        "error": f"ملف المواد القانونية غير موجود: {e}"}), 503
    except Exception as e:
        return jsonify({"status": "error",
                        "error": f"حدث خطأ غير متوقع أثناء معالجة الطلب: {e}"}), 500

    if article is None:
        return jsonify({"status": "error",
                        "error": "لم يُعثر على مادة بالمعرّف المحدد."}), 404
    return jsonify({"status": "success", "data": article}), 200
