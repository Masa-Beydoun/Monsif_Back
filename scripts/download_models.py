"""
تنزيل أوزان النماذج مرة وحدة إلى الكاش المحلي (.hf_cache).

ليش سكربت مستقل: التنزيل ~2.5GB وممكن ينقطع بنص الطريق. هون منعيد المحاولة
تلقائياً، والتنزيل بيكمّل من وين وقف (resume) مش من الصفر.

    python scripts/download_models.py
    python scripts/download_models.py --retries 20
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def fetch(repo_id: str, retries: int, wait: int) -> bool:
    from huggingface_hub import snapshot_download

    for attempt in range(1, retries + 1):
        try:
            print(f"  محاولة {attempt}/{retries} ...", flush=True)
            path = snapshot_download(
                repo_id=repo_id,
                # ملفات ما منحتاجها — بتوفّر جيجابايتات (أوزان TF/Flax/ONNX)
                ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "onnx/*", "*.ot"],
                max_workers=4,
            )
            print(f"  ✓ جاهز: {path}", flush=True)
            return True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  ✗ انقطع: {type(e).__name__}: {str(e)[:140]}", flush=True)
            if attempt < retries:
                print(f"    إعادة المحاولة بعد {wait}s (التنزيل بيكمّل من وين وقف) ...",
                      flush=True)
                time.sleep(wait)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="تنزيل أوزان النماذج مسبقاً")
    ap.add_argument("--retries", type=int, default=12)
    ap.add_argument("--wait", type=int, default=5, help="ثواني بين المحاولات")
    args = ap.parse_args()

    print(f"الكاش: {config.HF_HOME}")
    print("(التنزيل بيصير مرة وحدة — بعدها كل تشغيل بيقرأ من هون)\n")

    ok = True
    for repo in (config.EMBEDDING_MODEL, config.RERANKER_MODEL):
        print(f"» {repo}")
        if not fetch(repo, args.retries, args.wait):
            print(f"  ✗ فشل تنزيل {repo} بعد {args.retries} محاولة.")
            ok = False
        print()

    if ok:
        print("✓ كل النماذج جاهزة محلياً.")
        print("  الخطوة التالية:  python scripts/build_laws_index.py")
    else:
        print("✗ في نماذج ما نزلت. شغّلي السكربت مرة تانية — بيكمّل من وين وقف.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
