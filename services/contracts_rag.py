"""
الاسترجاع الدلالي لنماذج العقود.

يحضّر نصوص البحث من ملف نماذج العقود، ويبني فهرس FAISS من نوع IndexFlatIP على
متجهات مُعيَّرة (أي تشابه cosine)، ثم تقترح طبقة النموذج اللغوي الأنسب من قائمة
المرشحين.

الميزة خطوتان: /contracts/search يعيد المرشحين، و/contracts/get يعيد النموذج
كاملاً بحسب doc_id.

نموذج الترميز مشترك عبر model_registry إن كان نموذج المشروع نفسه، وإلا حُمِّلت
نسخة خاصة به (انظر CONTRACTS_EMBEDDING_MODEL).
"""

import json
import os
import pickle
import re
import threading
import time
from typing import Dict, List, Optional

import numpy as np

import config
from services import model_registry

_TAG_RE = re.compile(r"\[\[\d+:[^\]]+\]\]")
_WS_RE = re.compile(r"\s+")


def _log(msg: str) -> None:
    print(f"[contracts] {msg}", flush=True)


# نموذج الترميز

_own_encoder = None
_encoder_lock = threading.Lock()


def _uses_shared_model() -> bool:
    return config.CONTRACTS_EMBEDDING_MODEL == config.EMBEDDING_MODEL


def _get_encoder():
    """نموذج المشروع من المستودع المشترك، أو نسخة خاصة إن طُلب نموذج آخر."""
    if _uses_shared_model():
        return model_registry.get_dense_encoder()

    global _own_encoder
    if _own_encoder is None:
        with _encoder_lock:
            if _own_encoder is None:
                from sentence_transformers import SentenceTransformer

                t0 = time.time()
                _log(f"تحميل نموذج خاص بالعقود: {config.CONTRACTS_EMBEDDING_MODEL} ...")
                _own_encoder = SentenceTransformer(
                    config.CONTRACTS_EMBEDDING_MODEL, device=config.resolve_device()
                )
                _log(f"جاهز ({time.time() - t0:.1f}s)")
    return _own_encoder


def _needs_e5_prefix(model_name: str) -> bool:
    """نماذج e5 تتطلب بادئة «query:» أو «passage:»، بخلاف غيرها مثل BGE-M3."""
    return "e5" in model_name.lower()


# تحضير البيانات

def _prepare_record(tpl: Dict) -> Dict:
    """تحويل نموذج خام إلى سجل جاهز للفهرسة."""
    doc_cat = tpl.get("DocCat", "")
    subject = tpl.get("Subject", "")
    index_val = tpl.get("Index", "") or ""
    body = tpl.get("Body", "") or ""

    # تنظيف نص الـ Body من الوسوم من نمط [[1:text]]
    clean_body = _TAG_RE.sub(" ", body)
    clean_body = _WS_RE.sub(" ", clean_body).strip()

    formatted_index = index_val.replace(";", ", ") if index_val else ""
    search_text = f"{doc_cat}. {subject}. {formatted_index}. {clean_body[:400]}"

    return {
        "doc_id": tpl.get("DocID", ""),
        "category": doc_cat,
        "subject": subject,
        "index_keywords": index_val,
        "search_text": search_text,
        "body": body,
        "raw_body": tpl.get("RawBody", ""),
        "placeholders": tpl.get("Placeholders", []) or [],
    }


def _load_templates(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("templates", [])
    return []


# المحرك

class ContractsRAG:
    """بحث دلالي بنماذج العقود + استرجاع النموذج كامل بالـ doc_id."""

    def __init__(self):
        self.index = None
        self.records: List[Dict] = []
        self._by_doc_id: Dict[str, int] = {}
        self._ready = False

    # التحميل والبناء

    def _reindex_lookup(self) -> None:
        # عند تكرار DocID يُعتمد أول ظهور ويُسجَّل تنبيه بالباقي.
        self._by_doc_id = {}
        duplicates = 0
        for i, rec in enumerate(self.records):
            key = str(rec.get("doc_id", ""))
            if not key:
                continue
            if key in self._by_doc_id:
                duplicates += 1
                continue
            self._by_doc_id[key] = i
        if duplicates:
            _log(f"{duplicates} معرّف مكرر؛ اعتُمد أول ظهور لكل منها.")

    def build_from_json(self, json_path: str, save: bool = True) -> None:
        import faiss

        templates = _load_templates(json_path)
        if not templates:
            raise ValueError(f"لا توجد نماذج عقود في الملف: {json_path}")

        self.records = [_prepare_record(t) for t in templates]
        _log(f"تجهيز {len(self.records)} نموذج عقد ...")

        model = _get_encoder()
        texts = [r["search_text"] for r in self.records]
        if _needs_e5_prefix(config.CONTRACTS_EMBEDDING_MODEL):
            texts = ["passage: " + t for t in texts]

        embeddings = model.encode(
            texts,
            batch_size=config.CONTRACTS_BUILD_BATCH,
            show_progress_bar=True,
            normalize_embeddings=True,   # يجعل IndexFlatIP مكافئاً لتشابه cosine
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self._reindex_lookup()
        self._ready = True
        _log(f"الفهرس جاهز: {self.index.ntotal} متجه")

        if save:
            self.save(config.CONTRACTS_INDEX_FILE, config.CONTRACTS_METADATA_FILE)

    def save(self, index_file: str, metadata_file: str) -> None:
        import faiss

        os.makedirs(os.path.dirname(index_file) or ".", exist_ok=True)
        faiss.write_index(self.index, index_file)
        with open(metadata_file, "wb") as f:
            # يُخزَّن اسم النموذج أيضاً: فهرس مبني بنموذج وبحث بنموذج آخر يعطي
            # نتائج عشوائية دون أي خطأ، فيُكتشف الأمر عند التحميل.
            pickle.dump(
                {"records": self.records, "model": config.CONTRACTS_EMBEDDING_MODEL}, f
            )
        _log(f"حُفظ → {index_file} + {metadata_file}")

    def load_index(self) -> None:
        import faiss

        index_file = config.CONTRACTS_INDEX_FILE
        metadata_file = config.CONTRACTS_METADATA_FILE
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"لم يُعثر على فهرس العقود: {index_file}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"لم يُعثر على بيانات العقود الوصفية: {metadata_file}")

        self.index = faiss.read_index(index_file)
        with open(metadata_file, "rb") as f:
            d = pickle.load(f)
        self.records = d["records"]
        built_with = d.get("model")
        if built_with and built_with != config.CONTRACTS_EMBEDDING_MODEL:
            _log(
                f"الفهرس مبني بنموذج «{built_with}» بينما تطلب الإعدادات "
                f"«{config.CONTRACTS_EMBEDDING_MODEL}»، والنتائج ستكون خاطئة.\n"
                f"    أعد البناء: python scripts/build_contracts_index.py --force"
            )
        self._reindex_lookup()
        self._ready = True
        _log(f"الفهرس محمّل: {self.index.ntotal} نموذج عقد")

    def auto_load(self) -> None:
        """يحمّل الفهرس المحفوظ إن وُجد، وإلا بناه من ملف JSON وحفظه."""
        if os.path.exists(config.CONTRACTS_INDEX_FILE) and os.path.exists(
            config.CONTRACTS_METADATA_FILE
        ):
            self.load_index()
        elif os.path.exists(config.CONTRACTS_SOURCE_JSON):
            _log("لا يوجد فهرس محفوظ؛ يجري البناء من ملف JSON ...")
            self.build_from_json(config.CONTRACTS_SOURCE_JSON)
        else:
            raise FileNotFoundError(
                f"لم يُعثر على فهرس نماذج العقود ولا على ملف JSON لبنائه.\n"
                f"  المتوقع: {config.CONTRACTS_INDEX_FILE}\n"
                f"  أو:      {config.CONTRACTS_SOURCE_JSON}"
            )

    # البحث

    def search(self, query_text: str, **kw) -> List[Dict]:
        """قائمة نماذج العقود المرشحة للاستعلام."""
        if not self._ready:
            raise RuntimeError("فهرس العقود غير محمّل.")

        p = dict(config.CONTRACTS_DEFAULTS)
        p.update({k: v for k, v in kw.items() if v is not None})
        top_k = max(1, min(int(p["top_k"]), len(self.records)))

        model = _get_encoder()
        text = query_text
        if _needs_e5_prefix(config.CONTRACTS_EMBEDDING_MODEL):
            text = "query: " + text
        q_vec = model.encode([text], normalize_embeddings=True).astype(np.float32)

        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < p["min_score"]:
                continue
            rec = self.records[int(idx)]
            results.append(
                {
                    "doc_id": rec["doc_id"],
                    "category": rec["category"],
                    "subject": rec["subject"],
                    "index_keywords": rec["index_keywords"],
                    "placeholders_count": len(rec["placeholders"]),
                    "score": round(float(score), 4),
                }
            )
        return results

    # استرجاع نموذج كامل

    def get_contract(self, doc_id: Optional[str] = None,
                     subject: Optional[str] = None) -> Optional[Dict]:
        """النموذج كاملاً بحسب doc_id، أو بحسب العنوان مطابقةً حرفية."""
        if not self._ready:
            raise RuntimeError("فهرس العقود غير محمّل.")

        rec = None
        if doc_id:
            i = self._by_doc_id.get(str(doc_id).strip())
            if i is not None:
                rec = self.records[i]
        if rec is None and subject:
            wanted = subject.strip()
            rec = next((r for r in self.records if r["subject"].strip() == wanted), None)
        if rec is None:
            return None

        return {
            "doc_id": rec["doc_id"],
            "category": rec["category"],
            "subject": rec["subject"],
            "index_keywords": rec["index_keywords"],
            "placeholders": rec["placeholders"],
            "placeholders_count": len(rec["placeholders"]),
            "body": rec["body"],
            "raw_body": rec["raw_body"],
        }

    def categories(self) -> Dict[str, int]:
        """عدد النماذج في كل فئة."""
        counts: Dict[str, int] = {}
        for rec in self.records:
            counts[rec["category"]] = counts.get(rec["category"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


# طبقة النموذج اللغوي: اقتراح الأنسب من المرشحين

class MissingAPIKey(RuntimeError):
    pass


def _groq_client():
    if not config.GROQ_API_KEY:
        raise MissingAPIKey(
            "GROQ_API_KEY غير مضبوط. أضيفيه لملف .env بجذر المشروع:\n"
            "    GROQ_API_KEY=gsk_..."
        )
    from groq import Groq

    return Groq(api_key=config.GROQ_API_KEY)


SUGGEST_SYSTEM = (
    "أنت مساعد قانوني متخصص بتصنيف نماذج العقود السورية. "
    "مهمتك الوحيدة: اقتراح رقم النموذج الأنسب من قائمة مرشحين بناءً على طلب المستخدم. "
    "جوابك يجب أن يكون حصراً بصيغة JSON صحيحة بدون أي نص إضافي قبلها أو بعدها."
)


def suggest_best_match(user_query: str, candidates: List[Dict], **kw) -> Dict:
    """اقتراح النموذج اللغوي: {"choice": رقم أو None, "reason": "..."}.

    لا يعيد العقد مباشرة؛ يقترح فحسب، ويبقى التأكيد النهائي للمستخدم.
    """
    if not candidates:
        return {"choice": None, "reason": None}

    p = dict(config.CONTRACTS_DEFAULTS)
    p.update({k: v for k, v in kw.items() if v is not None})

    candidates_text = "\n".join(
        f"{i + 1}. الفئة: {c['category']} | العنوان: {c['subject']}"
        for i, c in enumerate(candidates)
    )

    user_prompt = f"""طلب المستخدم (مكتوب بلغة بسيطة أو عامية):
"{user_query}"

قائمة المرشحين المتاحين:
{candidates_text}

المطلوب:
- اختر رقم المرشح (من 1 إلى {len(candidates)}) الأنسب فعلياً لطلب المستخدم من ناحية الغاية القانونية للعقد (نوع التصرف، الأطراف، طبيعة العلاقة).
- إذا كان أكثر من مرشح مناسب، اختر الأدق تطابقاً مع تفاصيل الطلب.
- إذا لم يوجد أي مرشح مناسب فعلياً، أعد choice: 0.

أجب بصيغة JSON فقط بهذا الشكل بالضبط:
{{"choice": <رقم>, "reason": "<سبب الاختيار بجملة وحدة قصيرة بالعربي>"}}"""

    messages = [
        {"role": "system", "content": SUGGEST_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    try:
        client = _groq_client()
        kwargs = dict(
            model=p["groq_model"],
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=p["suggest_max_tokens"],
            temperature=p["temperature"],
        )
        try:
            # يعطّل وضع التفكير في نماذج Qwen3؛ غير مدعوم في كل النماذج.
            response = client.chat.completions.create(reasoning_effort="none", **kwargs)
        except Exception:
            response = client.chat.completions.create(**kwargs)
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return {"choice": None, "reason": None,
                    "error": "أعاد النموذج اللغوي محتوى فارغاً."}
        parsed = json.loads(raw)
    except MissingAPIKey:
        raise
    except json.JSONDecodeError as e:
        return {"choice": None, "reason": None, "error": f"JSON غير صالح: {e}"}
    except Exception as e:
        return {"choice": None, "reason": None,
                "error": f"تعذّر الاتصال بخدمة النموذج اللغوي: {e}"}

    choice = parsed.get("choice")
    try:
        choice = int(choice)
    except (TypeError, ValueError):
        choice = None
    # القيمة 0 تعني «لا مرشح مناسب»، وأي رقم خارج المدى يُعدّ فشلاً.
    if not choice or not (1 <= choice <= len(candidates)):
        return {"choice": None, "reason": parsed.get("reason")}

    best = candidates[choice - 1]
    return {
        "choice": choice,
        "doc_id": best["doc_id"],
        "subject": best["subject"],
        "reason": parsed.get("reason"),
    }


# واجهة الاستخدام

_instance: Optional[ContractsRAG] = None
_init_lock = threading.Lock()


def warmup() -> None:
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                rag = ContractsRAG()
                rag.auto_load()
                _instance = rag


def get_rag() -> ContractsRAG:
    if _instance is None:
        warmup()
    return _instance


def search_contracts(query_text: str, **kw) -> Dict:
    """بحث مع اقتراح اختياري، بغلاف جاهز لاستجابة JSON."""
    if not query_text or not query_text.strip():
        return {"query": query_text, "error": "الاستعلام فارغ.", "results": []}

    t0 = time.time()
    p = dict(config.CONTRACTS_DEFAULTS)
    p.update({k: v for k, v in kw.items() if v is not None})

    rag = get_rag()
    results = rag.search(query_text, **kw)

    suggestion = None
    if p["suggest"] and results:
        suggestion = suggest_best_match(query_text, results, **kw)

    return {
        "query": query_text,
        "found": bool(results),
        "count": len(results),
        "results": results,
        "suggestion": suggestion,
        "message": None if results else "لا توجد نماذج عقود مطابقة للطلب.",
        "params_used": p,
        "took_ms": int((time.time() - t0) * 1000),
    }


def get_contract(doc_id: Optional[str] = None, subject: Optional[str] = None) -> Optional[Dict]:
    return get_rag().get_contract(doc_id=doc_id, subject=subject)


def is_loaded() -> bool:
    return _instance is not None
