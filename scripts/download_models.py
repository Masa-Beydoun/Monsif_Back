"""
تنزيل أوزان النماذج مرة واحدة إلى الكاش المحلي (.hf_cache).

حجم التنزيل نحو 2.5GB وقد ينقطع، لذلك تُعاد المحاولة تلقائياً ويُستأنف التنزيل
من موضع الانقطاع لا من البداية.

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
                # ملفات غير مستخدمة (أوزان TF/Flax/ONNX)؛ استبعادها يوفّر مساحة كبيرة.
                ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "onnx/*", "*.ot"],
                max_workers=4,
            )
            print(f"  تم: {path}", flush=True)
            return True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  انقطع: {type(e).__name__}: {str(e)[:140]}", flush=True)
            if attempt < retries:
                print(f"    إعادة المحاولة بعد {wait}s (يُستأنف من موضع الانقطاع) ...",
                      flush=True)
                time.sleep(wait)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="تنزيل أوزان النماذج مسبقاً")
    ap.add_argument("--retries", type=int, default=12)
    ap.add_argument("--wait", type=int, default=5, help="ثواني بين المحاولات")
    args = ap.parse_args()

    print(f"الكاش: {config.HF_HOME}")
    print("(يجري التنزيل مرة واحدة، ثم يُقرأ من الكاش في كل تشغيل)\n")

    ok = True
    for repo in (config.EMBEDDING_MODEL, config.RERANKER_MODEL):
        print(f"- {repo}")
        if not fetch(repo, args.retries, args.wait):
            print(f"  فشل تنزيل {repo} بعد {args.retries} محاولة.")
            ok = False
        print()

    if ok:
        print("كل النماذج جاهزة محلياً.")
        print("  الخطوة التالية:  python scripts/build_laws_index.py")
    else:
        print("تعذّر تنزيل بعض النماذج. أعد تشغيل السكربت؛ يُستأنف من موضع الانقطاع.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
