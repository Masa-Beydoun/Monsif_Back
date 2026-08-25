"""تصفّح القوانين وموادها — قراءة مباشرة من articles_unified.jsonl.

هذه الوحدة للتصفّح لا للبحث الدلالي: لا تُحمَّل أي نماذج ولا يُفتح فهرس Qdrant،
لذلك كل الطلبات فورية ولا تتعارض مع قفل الفهرس الحصري.

الملف يُقرأ مرة واحدة ويُخزَّن في الذاكرة (2278 مادة، حوالي 6 ميغابايت).
"""

import json
import re
import threading
from typing import Dict, List, Optional

import config

# التحميل الكسول للمجموعة

_articles: Optional[List[Dict]] = None
_by_law: Optional[Dict[str, List[Dict]]] = None
_by_article_id: Optional[Dict[str, Dict]] = None
_load_lock = threading.Lock()

NO_CATEGORY = "بدون تصنيف"

# علامات اقتباس قد يلصقها المستخدم بالقيمة عند كتابة الرابط يدوياً
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("«", "»"), ("“", "”"), ("‘", "’"))


def _clean(value: Optional[str]) -> Optional[str]:
    """تنظيف قيمة قادمة من الرابط: مسافات زائدة وعلامات اقتباس محيطة.

    كتابة ?category="..." في المتصفح تجعل علامتي الاقتباس جزءاً من القيمة،
    فلا تطابق أي شيء. نزيلها هنا بدل أن يعود الطلب فارغاً بلا سبب واضح.
    """
    if value is None:
        return None
    v = value.strip()
    changed = True
    while changed and len(v) >= 2:
        changed = False
        for opening, closing in _QUOTE_PAIRS:
            if v.startswith(opening) and v.endswith(closing):
                v = v[1:-1].strip()
                changed = True
                break
    return v


def _load() -> None:
    """قراءة الملف مرة واحدة وبناء الفهارس في الذاكرة."""
    global _articles, _by_law, _by_article_id
    if _articles is not None:
        return
    with _load_lock:
        if _articles is not None:
            return

        articles: List[Dict] = []
        by_law: Dict[str, List[Dict]] = {}
        by_id: Dict[str, Dict] = {}

        with open(config.LAWS_CORPUS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                articles.append(d)
                by_law.setdefault(d.get("law_id", ""), []).append(d)
                aid = d.get("article_id")
                if aid and aid not in by_id:
                    by_id[aid] = d

        for law_articles in by_law.values():
            law_articles.sort(key=_article_sort_key)

        _articles, _by_law, _by_article_id = articles, by_law, by_id


def _article_sort_key(d: Dict):
    """ترتيب المواد برقمها العددي؛ ما لا يبدأ برقم يُدفع إلى الآخر."""
    num = str(d.get("article_number", ""))
    m = re.match(r"\d+", num)
    return (0, int(m.group()), num) if m else (1, 0, num)


# التصنيفات (البنية الداخلية للقانون: كتاب / باب / فصل)

def _category_key(d: Dict) -> str:
    return (d.get("law_category_raw") or "").strip()


def _category_label(d: Dict) -> str:
    """مسار التصنيف بصيغة مقروءة: «الكتاب الثاني › الباب الحادي عشر › السرقة»."""
    hierarchy = d.get("hierarchy") or []
    names = [h.get("name") for h in hierarchy if h.get("name")]
    if names:
        return " › ".join(names)
    raw = _category_key(d)
    if raw:
        # مسار خام على شكل «01. الكتاب الأول - كذا\02. الباب الثاني - كذا»
        parts = [p.strip() for p in raw.split("\\") if p.strip()]
        return " › ".join(re.sub(r"^\d+\.\s*", "", p) for p in parts)
    return NO_CATEGORY


def _law_display(d: Dict) -> Dict:
    return {
        "law_id": d.get("law_id", ""),
        "law_name": d.get("law_name", ""),
        "short_name": d.get("short_name"),
        "category": d.get("category"),
    }


def _status_counts(articles: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for a in articles:
        s = a.get("status") or "غير محدد"
        counts[s] = counts.get(s, 0) + 1
    return counts


# تحويل مادة واحدة إلى قاموس الاستجابة

def article_to_dict(d: Dict, include_body: bool = True) -> Dict:
    """كل معلومات المادة. body_normalized هو النص المفضّل للعرض."""
    out = {
        "article_id": d.get("article_id", ""),
        "law_id": d.get("law_id", ""),
        "law_name": d.get("law_name", ""),
        "short_name": d.get("short_name"),
        "article_number": d.get("article_number", ""),
        "status": d.get("status", ""),
        "category": d.get("category"),
        "category_key": _category_key(d),
        "category_label": _category_label(d),
        "hierarchy": d.get("hierarchy") or [],
        "references": d.get("references") or [],
        "referenced_by": d.get("referenced_by") or [],
        "dependencies": d.get("dependencies") or [],
        "source_article_id": d.get("source_article_id"),
    }
    if include_body:
        out["body"] = d.get("body_normalized") or d.get("body_raw") or ""
        out["body_raw"] = d.get("body_raw") or ""
        out["body_normalized"] = d.get("body_normalized") or ""
    return out


# قائمة القوانين

def list_laws(q: Optional[str] = None, category: Optional[str] = None) -> Dict:
    """ملخّص كل قانون: عدد مواده وتصنيفاته وتوزّع حالات مواده."""
    _load()

    q = _clean(q) or None
    category = _clean(category)

    laws: List[Dict] = []
    for law_id, articles in _by_law.items():
        head = articles[0]
        if category is not None and (head.get("category") or "") != category:
            continue
        if q:
            needle = q.strip()
            haystack = f"{head.get('law_name', '')} {head.get('short_name') or ''} {law_id}"
            if needle not in haystack:
                continue

        category_keys = {_category_key(a) for a in articles}
        laws.append({
            **_law_display(head),
            "article_count": len(articles),
            "category_count": len({k for k in category_keys if k}),
            "has_categories": any(category_keys - {""}),
            "status_counts": _status_counts(articles),
            "first_article_number": articles[0].get("article_number", ""),
            "last_article_number": articles[-1].get("article_number", ""),
        })

    laws.sort(key=lambda x: -x["article_count"])

    # تجميع القوانين حسب تصنيفها الأعلى (قد يكون null لأغلبها)
    groups: Dict[str, Dict] = {}
    for law in laws:
        key = law["category"] or NO_CATEGORY
        g = groups.setdefault(key, {"category": law["category"], "category_label": key,
                                    "law_count": 0, "article_count": 0, "law_ids": []})
        g["law_count"] += 1
        g["article_count"] += law["article_count"]
        g["law_ids"].append(law["law_id"])

    # القيم الصالحة لوسيط category — محسوبة من كل القوانين لا من المصفّاة،
    # حتى يعرف المستخدم سبب الرد الفارغ بدل أن يخمّن.
    available = sorted({(a[0].get("category") or "") for a in _by_law.values()})
    available_categories = [c for c in available if c]

    payload = {
        "count": len(laws),
        "total_articles": sum(law["article_count"] for law in laws),
        "laws": laws,
        "categories": list(groups.values()),
        "available_categories": available_categories,
        "filters_used": {"q": q, "category": category},
    }

    if category is not None and not laws:
        payload["message"] = (
            "لا يوجد قانون ضمن هذا التصنيف. الوسيط 'category' يصفّي حسب تصنيف "
            "القانون لا حسب اسمه؛ القيم المتاحة: "
            + ("، ".join(available_categories) or "لا يوجد")
            + ". للبحث باسم القانون استعمل الوسيط 'q'."
        )
    elif not laws:
        payload["message"] = "لا يوجد قانون مطابق."

    return payload


# مواد قانون واحد

def get_law_articles(law_id: str, grouped: bool = True, include_body: bool = True,
                     status: Optional[str] = None, category: Optional[str] = None,
                     q: Optional[str] = None, page: int = 1,
                     per_page: Optional[int] = None) -> Optional[Dict]:
    """مواد القانون كاملة، مجمَّعة ضمن تصنيفاتها الداخلية.

    يعيد None إذا لم يوجد قانون بهذا المعرّف (ليحوّله المسار إلى 404).
    """
    _load()

    status = _clean(status) or None
    category = _clean(category)
    q = _clean(q) or None

    key = _clean(law_id) or ""
    articles = _by_law.get(key)
    if articles is None:
        # سماحية: المطابقة بالاسم الكامل أو المختصر
        for candidate in _by_law.values():
            head = candidate[0]
            if key in (head.get("law_name", "").strip(), (head.get("short_name") or "").strip()):
                articles = candidate
                break
    if not articles:
        return None

    head = articles[0]
    total_in_law = len(articles)

    selected = articles
    if status:
        selected = [a for a in selected if (a.get("status") or "") == status]
    if category is not None:
        selected = [a for a in selected if _category_key(a) == category]
    if q:
        needle = q.strip()
        selected = [a for a in selected
                    if needle in str(a.get("article_number", ""))
                    or needle in (a.get("body_normalized") or "")
                    or needle in (a.get("body_raw") or "")]

    total_matched = len(selected)

    # ترقيم الصفحات اختياري: بدون per_page تُعاد كل المواد
    page = max(1, int(page or 1))
    if per_page:
        per_page = max(1, int(per_page))
        total_pages = max(1, (total_matched + per_page - 1) // per_page)
        start = (page - 1) * per_page
        selected = selected[start:start + per_page]
    else:
        total_pages = 1

    payload = {
        "law": {
            **_law_display(head),
            "article_count": total_in_law,
            "status_counts": _status_counts(articles),
        },
        "count": len(selected),
        "total_matched": total_matched,
        "grouped": bool(grouped),
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total": total_matched,
        },
        "filters_used": {"status": status, "category": category, "q": q,
                         "include_body": include_body},
    }

    if grouped:
        order: List[str] = []
        buckets: Dict[str, Dict] = {}
        for a in selected:
            ckey = _category_key(a)
            if ckey not in buckets:
                order.append(ckey)
                buckets[ckey] = {
                    "category_key": ckey,
                    "category_label": _category_label(a),
                    "hierarchy": a.get("hierarchy") or [],
                    "article_count": 0,
                    "articles": [],
                }
            buckets[ckey]["article_count"] += 1
            buckets[ckey]["articles"].append(article_to_dict(a, include_body))
        payload["categories"] = [buckets[k] for k in order]
        payload["category_count"] = len(order)
    else:
        payload["articles"] = [article_to_dict(a, include_body) for a in selected]

    return payload


def get_article(article_id: str) -> Optional[Dict]:
    """مادة واحدة بمعرّفها الكامل — يستعملها المسار لفكّ ارتباطات المواد."""
    _load()
    d = _by_article_id.get(_clean(article_id) or "")
    return article_to_dict(d) if d else None


def is_loaded() -> bool:
    return _articles is not None


def stats() -> Dict:
    _load()
    return {
        "total_articles": len(_articles),
        "total_laws": len(_by_law),
        "corpus_file": str(config.LAWS_CORPUS_FILE),
    }
