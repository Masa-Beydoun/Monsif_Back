"""تصفّح السوابق القضائية وتصنيفاتها — قراءة مباشرة من standard_cases.json.

هذه الوحدة للتصفّح لا للبحث الدلالي (قارن services/cases_rag.py): لا تُحمَّل
أي نماذج ولا يُفتح فهرس FAISS، لذلك كل الطلبات فورية.

"التصنيف" هنا هو حقل crimes (الجرائم المنسوبة للقضية) لأنه الحقل الوحيد
القابل للاستعمال كتصنيف فعلي؛ حقل case_category قيمته ثابتة لكل القضايا.
الملف يُقرأ مرة واحدة ويُخزَّن في الذاكرة.
"""

import json
import threading
from typing import Dict, List, Optional

import config

_cases: Optional[List[Dict]] = None
_by_uid: Optional[Dict[str, Dict]] = None
_by_crime: Optional[Dict[str, List[Dict]]] = None
_load_lock = threading.Lock()

NO_CRIME = "بدون تصنيف"


def _uid(d: Dict) -> str:
    return d.get("file_name") or str(d.get("case_number", ""))


def _load() -> None:
    global _cases, _by_uid, _by_crime
    if _cases is not None:
        return
    with _load_lock:
        if _cases is not None:
            return

        with open(config.CASES_SOURCE_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cases: List[Dict] = raw if isinstance(raw, list) else [raw]

        by_uid: Dict[str, Dict] = {}
        by_crime: Dict[str, List[Dict]] = {}
        for d in cases:
            by_uid[_uid(d)] = d
            crimes = d.get("crimes") or []
            if not crimes:
                by_crime.setdefault(NO_CRIME, []).append(d)
            for c in crimes:
                by_crime.setdefault(c, []).append(d)

        _cases, _by_uid, _by_crime = cases, by_uid, by_crime


def _case_summary(d: Dict) -> Dict:
    outcome = d.get("outcome")
    return {
        "case_uid": _uid(d),
        "case_number": d.get("case_number", ""),
        "file_name": d.get("file_name", ""),
        "type": d.get("type"),
        "decision_year": d.get("decision_year", ""),
        "outcome": " | ".join(outcome) if isinstance(outcome, list) else str(outcome or ""),
        "crimes": d.get("crimes") or [],
        "penal_code_articles": d.get("penal_code_articles") or [],
        "other_laws": d.get("other_laws") or [],
        "text_preview": d.get("text_preview") or "",
    }


def case_to_dict(d: Dict) -> Dict:
    """كل معلومات القضية بما فيها الأقسام الكاملة (sections)."""
    out = _case_summary(d)
    out.update({
        "is_public_prosecution": d.get("is_public_prosecution"),
        "criminal_proceedings_articles": d.get("criminal_proceedings_articles") or [],
        "fees_law": d.get("fees_law") or [],
        "sections": d.get("sections") or {},
    })
    return out


def list_cases(q: Optional[str] = None, crime: Optional[str] = None,
               year: Optional[str] = None, outcome: Optional[str] = None,
               page: int = 1, per_page: Optional[int] = None) -> Dict:
    """قائمة القضايا (ملخّص) مع إمكانية التصفية حسب الجريمة/السنة/نتيجة الحكم،
    وتجميعها حسب الجريمة."""
    _load()

    q = (q or "").strip() or None
    crime = (crime or "").strip() or None
    year = (year or "").strip() or None
    outcome = (outcome or "").strip() or None

    selected = _cases
    if crime is not None:
        selected = [d for d in selected if crime in (d.get("crimes") or []) or
                    (crime == NO_CRIME and not d.get("crimes"))]
    if year is not None:
        selected = [d for d in selected if str(d.get("decision_year", "")) == year]
    if outcome is not None:
        selected = [d for d in selected
                    if outcome in (d.get("outcome") if isinstance(d.get("outcome"), list)
                                   else [str(d.get("outcome") or "")])]
    if q:
        needle = q.strip()
        selected = [d for d in selected
                    if needle in str(d.get("case_number", ""))
                    or needle in (d.get("text_preview") or "")
                    or needle in " ".join(d.get("crimes") or [])]

    total_matched = len(selected)

    page = max(1, int(page or 1))
    if per_page:
        per_page = max(1, int(per_page))
        total_pages = max(1, (total_matched + per_page - 1) // per_page)
        start = (page - 1) * per_page
        page_items = selected[start:start + per_page]
    else:
        total_pages = 1
        page_items = selected

    # التصنيفات (الجرائم) — محسوبة من كل القضايا لا من المصفّاة، حتى تُعرض
    # قائمة كاملة بالخيارات المتاحة للتصفية.
    categories = sorted(
        (
            {"crime": c, "case_count": len(items)}
            for c, items in _by_crime.items()
        ),
        key=lambda x: -x["case_count"],
    )
    available_years = sorted({str(d.get("decision_year", "")) for d in _cases} - {""})
    available_outcomes = sorted({
        o for d in _cases
        for o in (d.get("outcome") if isinstance(d.get("outcome"), list)
                  else [str(d.get("outcome") or "")])
        if o
    })

    payload = {
        "count": len(page_items),
        "total_matched": total_matched,
        "total_cases": len(_cases),
        "cases": [_case_summary(d) for d in page_items],
        "categories": categories,
        "available_years": available_years,
        "available_outcomes": available_outcomes,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total": total_matched,
        },
        "filters_used": {"q": q, "crime": crime, "year": year, "outcome": outcome},
    }

    if not page_items:
        payload["message"] = "لا توجد قضايا مطابقة لمعايير التصفية."

    return payload


def get_case(case_uid: str) -> Optional[Dict]:
    """قضية واحدة بمعرّفها (file_name أو case_number)، مع كل الأقسام."""
    _load()
    key = (case_uid or "").strip()
    d = _by_uid.get(key)
    if d is None:
        for candidate in _cases:
            if str(candidate.get("case_number", "")) == key:
                d = candidate
                break
    return case_to_dict(d) if d else None


def is_loaded() -> bool:
    return _cases is not None


def stats() -> Dict:
    _load()
    return {
        "total_cases": len(_cases),
        "total_crimes": len(_by_crime),
        "source_file": str(config.CASES_SOURCE_JSON),
    }
