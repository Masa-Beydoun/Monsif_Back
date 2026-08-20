"""
بناء فهرس Qdrant لـ RAG المواد القانونية — يُشغَّل **مرة وحدة**.

هاد هو الشي الغالي الوحيد بالمشروع: تحويل كل مادة قانونية لمتجه dense + sparse
بنموذج BGE-M3. بعد ما يخلص، الفهرس بينحفظ على القرص وما بينبنى مرة تانية —
السيرفر بس بيفتحه.

    python scripts/build_laws_index.py            # يبني إذا ما في فهرس
    python scripts/build_laws_index.py --force    # يهدم ويعيد البناء
    python scripts/build_laws_index.py --limit 50 # تجربة سريعة على 50 مادة

⚠️ لازم يكون السيرفر **مطفي** وقت التشغيل: Qdrant المحلي بياخد قفل حصري
   على مجلد الفهرس، وعمليتين ما بيقدروا يفتحوه سوا.
"""

import argparse
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from services import laws_rag, model_registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="بناء فهرس Qdrant للمواد القانونية")
    ap.add_argument("--force", action="store_true", help="احذف الفهرس الموجود وأعد البناء")
    ap.add_argument("--limit", type=int, default=0, help="ابنِ أول N مادة فقط (للتجربة)")
    ap.add_argument("--batch", type=int, default=config.LAWS_BUILD_BATCH,
                    help=f"حجم دفعة الترميز (افتراضي {config.LAWS_BUILD_BATCH})")
    ap.add_argument("--law-contains", default="",
                    help="ابنِ فقط المواد التي يحتوي اسم قانونها هذا النص")
    args = ap.parse_args()

    from qdrant_client import QdrantClient, models

    qdrant_path = config.LAWS_QDRANT_PATH
    collection = config.LAWS_COLLECTION

    if args.force and os.path.exists(qdrant_path):
        print(f"--force → حذف الفهرس الموجود: {qdrant_path}")
        shutil.rmtree(qdrant_path)

    # ── 1) تحميل المجموعة ──────────────────────────────────────────────────
    if not config.LAWS_CORPUS_FILE.exists():
        print(f"✗ ملف المجموعة غير موجود: {config.LAWS_CORPUS_FILE}")
        return 1

    print(f"قراءة المجموعة من {config.LAWS_CORPUS_FILE} ...")
    articles = laws_rag.load_corpus()
    print(f"  {len(articles)} مادة قابلة للفهرسة")

    if args.law_contains:
        articles = [a for a in articles if args.law_contains in a.law_name]
        print(f"  بعد فلتر «{args.law_contains}»: {len(articles)} مادة")
    if args.limit:
        articles = articles[:args.limit]
        print(f"  --limit → {len(articles)} مادة")

    if not articles:
        print("✗ ما في ولا مادة للفهرسة.")
        return 1

    # ── 2) هل الفهرس الموجود مطابق أصلاً؟ ────────────────────────────────
    Path(qdrant_path).parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=qdrant_path)
    if client.collection_exists(collection):
        n_cached = client.count(collection).count
        if n_cached == len(articles):
            print(f"✓ الفهرس موجود ومطابق ({n_cached} نقطة) — ما في شي للبناء.")
            print("  استعملي --force إذا بدك تعيدي البناء رغم هيك.")
            client.close()
            return 0
        print(f"الفهرس الحالي فيه {n_cached} نقطة بس المجموعة {len(articles)} مادة → إعادة بناء.")
        client.delete_collection(collection)

    # ── 3) تحميل النموذج ──────────────────────────────────────────────────
    print("\nتحميل BGE-M3 (أول مرة بينزّل ~2.3GB — بعدها بينقرأ من الكاش المحلي) ...")
    embedder = model_registry.get_bgem3_flag()

    # ── 4) إنشاء المجموعة ─────────────────────────────────────────────────
    client.create_collection(
        collection_name=collection,
        vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    client.create_payload_index(collection, "status", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection, "law_id", models.PayloadSchemaType.KEYWORD)

    # ── 5) الترميز والفهرسة ────────────────────────────────────────────────
    device = config.resolve_device()
    print(f"\nترميز {len(articles)} مادة على {device} (batch={args.batch}) ...")
    if device == "cpu":
        print("⚠️ ما في GPU — على CPU هالخطوة ممكن تاخد 20–60 دقيقة. شغّليها ورّوحي اعملي شي تاني.")

    t0 = time.time()
    done = 0
    for start in range(0, len(articles), args.batch):
        batch = articles[start:start + args.batch]
        enc = embedder.encode(
            [a.body_normalized for a in batch],
            batch_size=len(batch), max_length=config.LAWS_EMBED_MAX_LENGTH,
            return_dense=True, return_sparse=True, return_colbert_vecs=False,
        )
        points = []
        for a, dense, lw in zip(batch, enc["dense_vecs"], enc["lexical_weights"]):
            points.append(models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, a.article_id)),
                vector={
                    "dense": dense.tolist(),
                    "sparse": models.SparseVector(
                        indices=[int(k) for k in lw],
                        values=[float(v) for v in lw.values()],
                    ),
                },
                payload={
                    "article_id": a.article_id, "law_id": a.law_id, "law_name": a.law_name,
                    "short_name": a.short_name, "article_number": a.article_number,
                    "status": a.status, "category": a.category, "body_raw": a.body_raw,
                    "hierarchy": a.hierarchy, "references": a.references,
                    "dependencies": a.dependencies,
                },
            ))
        client.upsert(collection, points=points)

        done += len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0
        eta = (len(articles) - done) / rate if rate else 0
        print(f"  {done}/{len(articles)}  ({rate:.1f} مادة/ثا، متبقّي ~{eta/60:.1f} دقيقة)",
              end="\r", flush=True)

    print()
    n = client.count(collection).count
    client.close()
    print(f"\n✓ تم — {n} نقطة بـ {(time.time() - t0)/60:.1f} دقيقة")
    print(f"  الفهرس محفوظ بـ: {qdrant_path}")
    print("  السيرفر رح يفتحه مباشرة بدون إعادة بناء.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
