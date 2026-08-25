"""معايرة عتبات نتائج البحث في المواد القانونية.

تُجرّب مجموعة استعلامات مصنّفة يدوياً (صالحة / خارج المجال) وتعرض توزّع
النتائج، ثم تمسح قيم min_score وتقيس كم استعلاماً صالحاً يبقى وكم استعلاماً
خارج المجال يُرفَض عند كل قيمة.

    python scripts/tune_laws_scores.py                     عيّنة مدمجة
    python scripts/tune_laws_scores.py --file queries.json ملفك أنت
    python scripts/tune_laws_scores.py --drop-sweep        معايرة score_drop_ratio

صيغة الملف:
    {
      "relevant":   ["استعلام قانوني صحيح", "..."],
      "irrelevant": ["سؤال خارج المجال", "..."]
    }

يحتاج السيرفر شغّالاً: python app.py
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_BASE = "http://127.0.0.1:5000"

# عيّنة أولية — استبدلها باستعلاماتك الحقيقية، فهي أساس المعايرة الصحيحة.
SAMPLE = {
    "relevant": [
        "السرقة الموصوفة ليلاً مع الكسر",
        "القتل العمد",
        "عقوبة الرشوة للموظف العام",
        "التزوير في سند رسمي",
        "جريمة الاحتيال",
        "عقوبة حيازة المخدرات",
        "اليوم انا ونازل من الباص اجت سيارة وقطعت الباص من ناحية الباب يلي نزلت منو، هل هاد الشي قانوني",
        "واحد سرق موبايلي من جيبي بالسرفيس شو بصير فيه",
        "ضربني واحد بالشارع وكسر ايدي شو حقي",
        "شفت حادث سير وما وقفت لأساعد، في عقوبة؟",
    ],
    "irrelevant": [
        "وصفة طبخ الكبة الحلبية",
        "كيف اتعلم البرمجة بايثون",
        "ما هو افضل هاتف ذكي",
        "متى يبدا دوري ابطال اوروبا",
        "شو بصير اذا كبيت دهان على الجامع الاموي",
        "بدي احجز تذكرة طيران على دبي",
        "شو احسن مطعم بدمشق",
    ],
}

SWEEP = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
DROP_SWEEP = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def search(base: str, query: str, **overrides):
    """استعلام بكل العتبات مطفأة، لنرى النتائج الخام قبل أي تصفية."""
    body = {
        "text": query,
        "top_n": overrides.pop("top_n", 8),
        "with_dependencies": False,
        "min_score": 0,
        "min_score_ratio": 0,
        "score_drop_ratio": 0,
    }
    body.update(overrides)
    req = urllib.request.Request(
        base + "/api/legal/laws/search",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))["data"]["results"]


def collect(base: str, queries: dict) -> dict:
    out = {}
    titles = {"relevant": "استعلامات صالحة:", "irrelevant": "استعلامات خارج المجال:"}
    for label in ("relevant", "irrelevant"):
        print(titles[label])
        rows = []
        for q in queries.get(label, []):
            scores = [r["score"] for r in search(base, q)]
            rows.append((q, scores))
            top = scores[0] if scores else 0.0
            print(f"  {top:8.4f}  {q[:66]}")
        out[label] = rows
        print()
    return out


def sweep_min_score(data: dict) -> None:
    rel = [s[0] if s else 0.0 for _, s in data["relevant"]]
    irr = [s[0] if s else 0.0 for _, s in data["irrelevant"]]
    n_rel, n_irr = len(rel), len(irr)

    print("مسح min_score — كم استعلاماً يمر عند كل عتبة:")
    print(f"  {'عتبة':>8} | {'صالح يمر':>12} | {'خارج المجال يمر':>16} | الحكم")
    print("  " + "-" * 62)
    best = None
    for t in SWEEP:
        kept_rel = sum(1 for s in rel if s >= t)
        kept_irr = sum(1 for s in irr if s >= t)
        # نريد أكبر عدد من الصالح مع أقل عدد من غير الصالح
        score = (kept_rel / n_rel) - (kept_irr / n_irr)
        if best is None or score > best[1]:
            best = (t, score)
        verdict = ""
        if kept_rel == n_rel and kept_irr == 0:
            verdict = "فصل تام"
        elif kept_irr == 0:
            verdict = "يرفض كل الضجيج لكنه يخسر أسئلة صالحة"
        elif kept_rel == n_rel:
            verdict = "يبقي كل الصالح لكنه يمرّر ضجيجاً"
        print(f"  {t:8.3f} | {kept_rel:>4}/{n_rel:<7} | {kept_irr:>6}/{n_irr:<9} | {verdict}")
    print()
    print(f"  أفضل مقايضة في هذه العيّنة: min_score = {best[0]}")

    overlap = [s for s in rel if s <= max(irr, default=0.0)]
    if overlap:
        print()
        print(f"  ⚠ تداخل: {len(overlap)} استعلاماً صالحاً نتيجته ≤ أعلى نتيجة خارج المجال "
              f"({max(irr):.4f}).")
        print("    لا توجد عتبة تفصل بينهما. السبب أن مقياس إعادة الترتيب يقيس تطابق")
        print("    الصياغة: السؤال بالعامية يعطي نتيجة منخفضة ولو كانت المادة صحيحة.")
        print("    الحل: أعد صياغة السؤال بمصطلحات قانونية قبل البحث.")


def sweep_drop(base: str, queries: dict) -> None:
    print("مسح score_drop_ratio — متوسط عدد المواد المُعادة للاستعلامات الصالحة:")
    print(f"  {'النسبة':>8} | {'متوسط العدد':>12}")
    print("  " + "-" * 26)
    for d in DROP_SWEEP:
        counts = []
        for q in queries.get("relevant", []):
            counts.append(len(search(base, q, score_drop_ratio=d, min_score_ratio=0.15)))
        print(f"  {d:8.2f} | {sum(counts) / max(1, len(counts)):12.2f}")
    print()
    print("  النسبة الأعلى تقصّ أكثر. 0 تُعطّل القصّ.")


def main() -> int:
    ap = argparse.ArgumentParser(description="معايرة عتبات البحث في المواد القانونية")
    ap.add_argument("--base", default=DEFAULT_BASE, help="عنوان السيرفر")
    ap.add_argument("--file", help="ملف JSON فيه relevant و irrelevant")
    ap.add_argument("--drop-sweep", action="store_true",
                    help="معايرة score_drop_ratio بدلاً من min_score")
    args = ap.parse_args()

    if args.file:
        queries = json.loads(Path(args.file).read_text(encoding="utf-8"))
    else:
        queries = SAMPLE
        print("تُستعمل العيّنة المدمجة. مرّر --file لمعايرة استعلاماتك أنت.\n")

    try:
        urllib.request.urlopen(args.base + "/api/health", timeout=10).read()
    except (urllib.error.URLError, OSError):
        print(f"لا يمكن الوصول إلى السيرفر على {args.base} — شغّل: python app.py")
        return 1

    print("=" * 70)
    print("النتائج الخام (أعلى نتيجة لكل استعلام، بلا أي عتبة)")
    print("=" * 70)
    data = collect(args.base, queries)

    if args.drop_sweep:
        sweep_drop(args.base, queries)
    else:
        sweep_min_score(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
