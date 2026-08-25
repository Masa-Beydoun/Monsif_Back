"""
اختبار سريع لجميع المسارات؛ يتطلب خادماً قيد التشغيل.

    python scripts/smoke_test.py                    # جميع الميزات
    python scripts/smoke_test.py --only laws        # ميزة واحدة
    python scripts/smoke_test.py --skip judgment    # تخطّي الحكم الأولي
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# الترميز الافتراضي لكونسول ويندوز cp1256 ولا يدعم العربية؛ يُجبَر على UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = "http://127.0.0.1:5000"

FACTS = (
    "تقدم المدعي بادعائه أنه سلّم المدعى عليه مبلغاً من المال على سبيل الأمانة "
    "بموجب سند موقع ومزيل ببصمة إصبعه، وقد وجّه إليه إنذاراً عدلياً مطالباً "
    "بإعادة المبلغ إلا أن المدعى عليه رفض الرد ولم يردّ الأمانة حتى تاريخ الادعاء."
)

TESTS = [
    ("health", "GET", "/api/health", None),
    ("summarize", "POST", "/api/legal/summarize", {"text": FACTS}),
    ("laws", "POST", "/api/legal/laws/search",
     {"text": "السرقة الموصوفة", "top_n": 3, "hybrid_top_k": 10}),
    ("cases", "POST", "/api/legal/cases/search",
     {"text": FACTS, "top_k": 3, "hybrid_top_k": 10}),
    ("search", "POST", "/api/legal/search", {"text": "السرقة الموصوفة", "top_k": 2}),
    ("judgment", "POST", "/api/legal/judgment/predict", {"text": FACTS}),
    # suggest=false كي لا يستهلك الاختبار السريع حصة النموذج اللغوي.
    ("contracts", "POST", "/api/legal/contracts/search",
     {"text": "بدي عقد إيجار محل تجاري", "top_k": 3, "suggest": False}),
]


def call(method: str, path: str, body=None, timeout: int = 900):
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8")), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}"), time.time() - t0
    except Exception as e:
        return None, {"error": str(e)}, time.time() - t0


def summarize(name: str, payload: dict) -> str:
    """سطر واحد يصف النتيجة بحسب نوع الميزة."""
    d = payload.get("data", payload)
    if name == "health":
        return f"device={d.get('models', {}).get('device')} loaded={d.get('features')}"
    if name == "summarize":
        return f"ملخّص {len(d.get('summary', ''))} حرف"
    if name in ("laws", "cases"):
        n = d.get("count", 0)
        top = d.get("results", [{}])[0] if d.get("results") else {}
        label = top.get("article_number") or top.get("case_number") or "—"
        score = top.get("similarity_score", "—")
        return f"{n} نتيجة | الأولى: {label} ({score}%) | {d.get('took_ms')}ms"
    if name == "search":
        parts = []
        for key in ("articles", "jurisprudence"):
            if key in d:
                parts.append(f"{key}={len(d[key].get('results', []))}")
        return " ".join(parts) or "—"
    if name == "contracts":
        n = d.get("count", 0)
        top = d.get("results", [{}])[0] if d.get("results") else {}
        sug = (d.get("suggestion") or {}).get("subject") or "—"
        return (f"{n} مرشح | الأول: {top.get('subject', '—')} ({top.get('score', '—')}) "
                f"| اقتراح: {sug} | {d.get('took_ms')}ms")
    if name == "judgment":
        v = d.get("verification", {})
        llm = d.get("meta", {}).get("llm", {})
        flagged = sum(1 for c in d.get("candidate_charges", []) if c.get("flagged"))
        return (f"outcome={d.get('outcome')} | تهم={len(d.get('candidate_charges', []))} "
                f"(موسومة {flagged}) | موثّق={v.get('fully_grounded')} "
                f"| {llm.get('backend')}:{llm.get('model')} | {d.get('meta', {}).get('took_ms')}ms")
    return ""


def main() -> int:
    global BASE

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    BASE = args.base

    failures = 0
    for name, method, path, body in TESTS:
        if args.only and name not in args.only:
            continue
        if name in args.skip:
            print(f"-  {name:<10} تخطٍّ")
            continue

        print(f".  {name:<10} {method} {path}", flush=True)
        status, payload, secs = call(method, path, body)

        if status == 200:
            print(f"✓  {name:<10} {secs:6.1f}s  {summarize(name, payload)}")
        else:
            failures += 1
            err = payload.get("error") or payload
            print(f"✗  {name:<10} {secs:6.1f}s  HTTP {status}: {str(err)[:200]}")

    print()
    if failures:
        print(f"فشل {failures} اختبار.")
    else:
        print("نجحت جميع الاختبارات.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
