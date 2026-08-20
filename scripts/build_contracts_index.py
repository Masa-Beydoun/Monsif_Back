"""
بناء فهرس نماذج العقود (FAISS) من hammurabi_templates_flat.json — مرة وحدة.

    python scripts/build_contracts_index.py
    python scripts/build_contracts_index.py --force
    python scripts/build_contracts_index.py --json path/to/templates.json

⚠️ إذا غيّرتي CONTRACTS_EMBEDDING_MODEL بملف .env لازم تعيدي البناء بـ --force،
   لأن فهرس مبني بنموذج وبحث بنموذج تاني بيعطي نتائج عشوائية.
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
    ap.add_argument("--force", action="store_true", help="أعد البناء حتى لو الفهرس موجود")
    ap.add_argument("--json", default=config.CONTRACTS_SOURCE_JSON,
                    help="مسار ملف نماذج العقود JSON")
    args = ap.parse_args()

    exists = (os.path.exists(config.CONTRACTS_INDEX_FILE)
              and os.path.exists(config.CONTRACTS_METADATA_FILE))
    if exists and not args.force:
        print(f"✓ الفهرس موجود أصلاً: {config.CONTRACTS_INDEX_FILE}")
        print("  استعملي --force لإعادة البناء.")
        return 0

    if not os.path.exists(args.json):
        print(f"✗ ملف نماذج العقود غير موجود: {args.json}")
        print("  حطّي hammurabi_templates_flat.json بـ data/contracts/ وأعيدي المحاولة.")
        return 1

    print(f"بناء فهرس العقود من {args.json} ...")
    print(f"نموذج الـ Embedding: {config.CONTRACTS_EMBEDDING_MODEL}")
    rag = ContractsRAG()
    rag.build_from_json(args.json, save=True)
    print(f"\n✓ تم — {len(rag.records)} نموذج عقد.")

    counts = rag.categories()
    print("\nتوزيع الفئات:")
    for cat, n in list(counts.items())[:20]:
        print(f"  {n:>5}  {cat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
