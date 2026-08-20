"""
بناء فهرس القضايا (FAISS + BM25) من standard_cases.json — يُشغَّل **مرة وحدة**.

عادةً ما بتحتاجيه: فهرس جاهز موجود أصلاً بـ data/cases/. استعمليه بس إذا
تغيّر ملف القضايا أو بدك تعيدي البناء من الصفر.

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
    ap = argparse.ArgumentParser(description="بناء فهرس القضايا")
    ap.add_argument("--force", action="store_true", help="أعد البناء حتى لو الفهرس موجود")
    ap.add_argument("--json", default=config.CASES_SOURCE_JSON, help="مسار ملف القضايا JSON")
    args = ap.parse_args()

    exists = (os.path.exists(config.CASES_INDEX_FILE)
              and os.path.exists(config.CASES_METADATA_FILE))
    if exists and not args.force:
        print(f"✓ الفهرس موجود أصلاً: {config.CASES_INDEX_FILE}")
        print("  استعملي --force لإعادة البناء.")
        return 0

    if not os.path.exists(args.json):
        print(f"✗ ملف القضايا غير موجود: {args.json}")
        return 1

    print(f"بناء فهرس القضايا من {args.json} ...")
    print("تحميل BGE-M3 (أول مرة بينزّل ~2.3GB) ...")
    rag = CasesRAG()
    rag.build_from_json(args.json, save=True)
    print("\n✓ تم.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
