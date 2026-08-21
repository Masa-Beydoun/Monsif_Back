"""
استيراد فهرس العقود الجاهز من النوتبوك بدل إعادة الترميز.

النوتبوك (contract_RAG.ipynb) حفظ ملفين على Drive:
    contracts.index    — فهرس FAISS من نوع IndexFlatIP بأبعاد 1024
    contracts_df.pkl   — DataFrame بأعمدة مطابقة تماماً لسجلات المشروع

هذا السكربت يحوّلهما إلى الصيغة التي يقرأها services/contracts_rag.py، دون
تحميل أي نموذج ودون إعادة حساب أي متجه (ثوانٍ معدودة بدل عشرات الدقائق).

    python scripts/import_contracts_index.py \
        --index  path/to/contracts.index \
        --pickle path/to/contracts_df.pkl

الفهرس مبني بنموذج intfloat/multilingual-e5-large، فيُكتب اسم النموذج داخل
ملف البيانات الوصفية. يتحقق contracts_rag.load_index() من تطابقه مع
CONTRACTS_EMBEDDING_MODEL وينبّه عند الاختلاف، لأن فهرساً مبنياً بنموذج
وبحثاً بنموذج آخر يعطي نتائج عشوائية بلا أي خطأ ظاهر.
"""

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# الأعمدة التي يتوقعها contracts_rag؛ أي عمود ناقص يوقف الاستيراد بدل أن يظهر
# الخلل لاحقاً على شكل حقول فارغة في الـ API.
REQUIRED = ("doc_id", "category", "subject", "index_keywords",
            "search_text", "body", "raw_body", "placeholders")

NOTEBOOK_MODEL = "intfloat/multilingual-e5-large"


def main() -> int:
    ap = argparse.ArgumentParser(description="استيراد فهرس العقود الجاهز")
    ap.add_argument("--index", required=True, help="مسار contracts.index")
    ap.add_argument("--pickle", required=True, help="مسار contracts_df.pkl")
    ap.add_argument("--model", default=NOTEBOOK_MODEL,
                    help=f"النموذج الذي بُني به الفهرس (افتراضي {NOTEBOOK_MODEL})")
    ap.add_argument("--force", action="store_true", help="الكتابة فوق فهرس موجود")
    args = ap.parse_args()

    import faiss
    import pandas as pd

    out_index = config.CONTRACTS_INDEX_FILE
    out_meta = config.CONTRACTS_METADATA_FILE

    if os.path.exists(out_index) and not args.force:
        print(f"الفهرس موجود مسبقاً: {out_index}")
        print("  استخدم --force للكتابة فوقه.")
        return 0

    for p in (args.index, args.pickle):
        if not os.path.exists(p):
            print(f"الملف غير موجود: {p}")
            return 1

    # الفهرس
    index = faiss.read_index(args.index)
    print(f"فهرس FAISS: {index.ntotal} متجه، بعد {index.d}")

    # السجلات
    df = pd.read_pickle(args.pickle)
    print(f"DataFrame: {df.shape[0]} صف، {df.shape[1]} عمود")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"أعمدة ناقصة في الـ DataFrame: {', '.join(missing)}")
        print(f"  الأعمدة الموجودة: {', '.join(map(str, df.columns))}")
        return 1

    if len(df) != index.ntotal:
        print(f"عدم تطابق: الفهرس {index.ntotal} متجه بينما الـ DataFrame {len(df)} صف.")
        print("  الترتيب بين الاثنين هو ما يربط المتجه بسجله؛ لا يمكن المتابعة.")
        return 1

    records = []
    for row in df[list(REQUIRED)].to_dict("records"):
        ph = row.get("placeholders")
        # قد تُخزَّن الحقول القابلة للتعبئة كمصفوفة numpy أو None حسب مصدرها.
        row["placeholders"] = list(ph) if ph is not None and not isinstance(ph, float) else []
        records.append({k: ("" if v is None else v) if k != "placeholders" else v
                        for k, v in row.items()})

    os.makedirs(os.path.dirname(out_index) or ".", exist_ok=True)

    # يُنسخ ملف الفهرس كما هو؛ لا داعي لإعادة كتابته عبر faiss.
    if os.path.abspath(args.index) != os.path.abspath(out_index):
        shutil.copyfile(args.index, out_index)

    with open(out_meta, "wb") as f:
        pickle.dump({"records": records, "model": args.model}, f)

    print(f"\nتم:\n  {out_index}\n  {out_meta}")
    print(f"\nالنموذج المسجَّل: {args.model}")
    if config.CONTRACTS_EMBEDDING_MODEL != args.model:
        print("\n  تنبيه: أضف هذا السطر إلى .env وإلا كانت نتائج البحث عشوائية:")
        print(f"      CONTRACTS_EMBEDDING_MODEL={args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
