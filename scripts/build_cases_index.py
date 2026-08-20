"""
بناء فهرس السوابق القضائية (FAISS و BM25) من standard_cases.json.

يوجد فهرس جاهز في data/cases/، فلا حاجة لهذا السكربت إلا عند تغيّر ملف السوابق
أو عند إعادة البناء من الصفر.

    python scripts/build_cases_index.py
    python scripts/build_cases_index.py --force
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from services.cases_rag import CasesRAG  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="بناء فهرس السوابق القضائية")
    ap.add_argument("--force", action="store_true", help="إعادة البناء حتى مع وجود الفهرس")
    ap.add_argument("--json", default=config.CASES_SOURCE_JSON,
                    help="مسار ملف السوابق بصيغة JSON")
    args = ap.parse_args()

    exists = (os.path.exists(config.CASES_INDEX_FILE)
              and os.path.exists(config.CASES_METADATA_FILE))
    if exists and not args.force:
        print(f"الفهرس موجود مسبقاً: {config.CASES_INDEX_FILE}")
        print("  استخدم --force لإعادة البناء.")
        return 0

    if not os.path.exists(args.json):
        print(f"ملف السوابق غير موجود: {args.json}")
        return 1

    print(f"بناء فهرس السوابق من {args.json} ...")
    print("تحميل BGE-M3 (ينزّل نحو 2.3GB أول مرة) ...")
    rag = CasesRAG()
    rag.build_from_json(args.json, save=True)
    print("\nاكتمل البناء.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
