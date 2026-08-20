"""
بناء فهرس نماذج العقود (FAISS) من hammurabi_templates_flat.json.

    python scripts/build_contracts_index.py
    python scripts/build_contracts_index.py --force
    python scripts/build_contracts_index.py --json path/to/templates.json

عند تغيير CONTRACTS_EMBEDDING_MODEL في ملف .env تجب إعادة البناء بـ --force،
لأن فهرساً مبنياً بنموذج وبحثاً بنموذج آخر يعطي نتائج عشوائية.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from services.contracts_rag import ContractsRAG  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="بناء فهرس نماذج العقود")
    ap.add_argument("--force", action="store_true", help="إعادة البناء حتى مع وجود الفهرس")
    ap.add_argument("--json", default=config.CONTRACTS_SOURCE_JSON,
                    help="مسار ملف نماذج العقود بصيغة JSON")
    args = ap.parse_args()

    exists = (os.path.exists(config.CONTRACTS_INDEX_FILE)
              and os.path.exists(config.CONTRACTS_METADATA_FILE))
    if exists and not args.force:
        print(f"الفهرس موجود مسبقاً: {config.CONTRACTS_INDEX_FILE}")
        print("  استخدم --force لإعادة البناء.")
        return 0

    if not os.path.exists(args.json):
        print(f"ملف نماذج العقود غير موجود: {args.json}")
        print("  ضع hammurabi_templates_flat.json في data/contracts/ وأعد المحاولة.")
        return 1

    print(f"بناء فهرس العقود من {args.json} ...")
    print(f"نموذج الـ Embedding: {config.CONTRACTS_EMBEDDING_MODEL}")
    rag = ContractsRAG()
    rag.build_from_json(args.json, save=True)
    print(f"\nاكتمل البناء: {len(rag.records)} نموذج عقد.")

    counts = rag.categories()
    print("\nتوزيع الفئات:")
    for cat, n in list(counts.items())[:20]:
        print(f"  {n:>5}  {cat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
