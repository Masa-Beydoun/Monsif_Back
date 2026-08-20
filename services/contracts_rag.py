"""
Part D — RAG نماذج العقود (Contracts RAG).

مأخوذ من contract_RAG.ipynb: تحضير نصوص البحث من hammurabi_templates_flat.json،
فهرس FAISS (IndexFlatIP على متجهات مُعيَّرة = cosine)، ثم طبقة LLM بتقترح الأنسب
من قائمة المرشحين.

التغييرات عن النوتبوك:
  1. الحلقة التفاعلية (input/print) انحذفت — الـ route بيرجّع JSON.
     `interactive_search()` صارت خطوتين: /contracts/search ثم /contracts/get.
     الاختيار صار بـ doc_id (مفتاح ثابت) بدل مطابقة الاسم حرفياً.
  2. الاقتراح بينعمل عبر Groq متل باقي المشروع، مش بتحميل Qwen2.5-7B محلياً
     (النوتبوك كان على GPU كولاب؛ ~15GB أوزان ما بتنحمّل مع باقي النماذج).
     البرومبت منقول حرفياً.
  3. pandas انحذف — الـ DataFrame استُبدل بقائمة dict، وهيك ما منزيد اعتمادية
     جديدة على المشروع.
  4. نموذج الـ embedding بيجي من model_registry (مشترك) إذا كان نفس نموذج
     المشروع؛ وإلا بينحمّل نسخته الخاصة (شوف CONTRACTS_EMBEDDING_MODEL).

منطق التحضير والبحث (تنظيف [[1:text]]، بناء search_text، بادئات e5، cosine)
منقول حرفياً عن النوتبوك.
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


# ════════════════════════════ نموذج الـ Embedding ════════════════════════════

_own_encoder = None
_encoder_lock = threading.Lock()


def _uses_shared_model() -> bool:
    return config.CONTRACTS_EMBEDDING_MODEL == config.EMBEDDING_MODEL


def _get_encoder():
    """نفس نموذج المشروع من المستودع المشترك، أو نسخة خاصة إذا انطلب غيره."""
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
    """موديلات e5 لازمها بادئة «query:» / «passage:»، غيرها لأ (BGE-M3 مثلاً)."""
    return "e5" in model_name.lower()


# ═══════════════════════════════ تحضير البيانات ═══════════════════════════════

def _prepare_record(tpl: Dict) -> Dict:
    """تحويل تمبلت خام لسجل جاهز للفهرسة — منقول عن الخلية الثانية بالنوتبوك."""
    doc_cat = tpl.get("DocCat", "")
    subject = tpl.get("Subject", "")
    index_val = tpl.get("Index", "") or ""
    body = tpl.get("Body", "") or ""

    # تنظيف نص الـ Body من التاغات متل [[1:text]]
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


# ═══════════════════════════════════ المحرك ═══════════════════════════════════

class ContractsRAG:
    """بحث دلالي بنماذج العقود + استرجاع النموذج كامل بالـ doc_id."""

    def __init__(self):
        self.index = None
        self.records: List[Dict] = []
        self._by_doc_id: Dict[str, int] = {}
        self._ready = False

    # ─────────────────────────── تحميل / بناء ───────────────────────────

    def _reindex_lookup(self) -> None:
        # ⚠️ النوتبوك حذّر من تكرار DocID — أول ظهور بيربح، والباقي بينطبع كتحذير
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
            _log(f"⚠️ {duplicates} DocID مكرر — انعتمد أول ظهور لكل واحد.")

    def build_from_json(self, json_path: str, save: bool = True) -> None:
        import faiss

        templates = _load_templates(json_path)
        if not templates:
            raise ValueError(f"ما في نماذج عقود بالملف: {json_path}")

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
            normalize_embeddings=True,   # مهم عشان IndexFlatIP يساوي cosine
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
            # منخزّن اسم النموذج كمان: فهرس مبني بنموذج وبحث بنموذج تاني =
            # نتائج عشوائية بصمت، فأحسن نمسكها وقت التحميل.
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
                f"⚠️ الفهرس مبني بـ «{built_with}» بس الإعدادات عم تطلب "
                f"«{config.CONTRACTS_EMBEDDING_MODEL}» — النتائج رح تكون غلط.\n"
                f"    شغّلي: python scripts/build_contracts_index.py --force"
            )
        self._reindex_lookup()
        self._ready = True
        _log(f"الفهرس محمّل: {self.index.ntotal} نموذج عقد")

    def auto_load(self) -> None:
        """يحمّل الفهرس المحفوظ إن وُجد، وإلا بيبنيه من الـ JSON ويحفظه."""
        if os.path.exists(config.CONTRACTS_INDEX_FILE) and os.path.exists(
            config.CONTRACTS_METADATA_FILE
        ):
            self.load_index()
        elif os.path.exists(config.CONTRACTS_SOURCE_JSON):
            _log("ما في فهرس محفوظ — بناء من JSON ...")
            self.build_from_json(config.CONTRACTS_SOURCE_JSON)
        else:
            raise FileNotFoundError(
                f"لا فهرس ولا ملف JSON لنماذج العقود.\n"
                f"  متوقع: {config.CONTRACTS_INDEX_FILE}\n"
                f"  أو:    {config.CONTRACTS_SOURCE_JSON}"
            )

    # ─────────────────────────────── البحث ───────────────────────────────

    def search(self, query_text: str, **kw) -> List[Dict]:
        """قائمة المرشحين — نفس search_contracts يلي بالنوتبوك."""
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

    # ────────────────────────── استرجاع عقد كامل ──────────────────────────

    def get_contract(self, doc_id: Optional[str] = None,
                     subject: Optional[str] = None) -> Optional[Dict]:
        """النموذج كامل بالـ doc_id، أو بالعنوان الحرفي متل النوتبوك."""
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
        """توزيع الفئات — متل df['category'].value_counts() بالنوتبوك."""
        counts: Dict[str, int] = {}
        for rec in self.records:
            counts[rec["category"]] = counts.get(rec["category"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


# ══════════════════ طبقة الـ LLM: اقتراح الأنسب من المرشحين ══════════════════

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
    """اقتراح الموديل: {"choice": رقم أو None, "reason": "..."}.

    ما بيرجع العقد مباشرة — بس بيقترح، والمستخدم هو يلي بيأكد الاختيار.
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
- إذا ما في أي مرشح مناسب فعلياً، ارجع choice: 0.

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
            # يطفي وضع التفكير بموديلات Qwen3 على Groq — غير مدعوم بكل الموديلات
            response = client.chat.completions.create(reasoning_effort="none", **kwargs)
        except Exception:
            response = client.chat.completions.create(**kwargs)
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return {"choice": None, "reason": None, "error": "الموديل رجّع محتوى فارغ."}
        parsed = json.loads(raw)
    except MissingAPIKey:
        raise
    except json.JSONDecodeError as e:
        return {"choice": None, "reason": None, "error": f"JSON غير صالح: {e}"}
    except Exception as e:
        return {"choice": None, "reason": None, "error": f"خطأ أثناء الاتصال بـ Groq: {e}"}

    choice = parsed.get("choice")
    try:
        choice = int(choice)
    except (TypeError, ValueError):
        choice = None
    # 0 = «ما في مرشح مناسب» حسب البرومبت، وأي رقم برّا المدى منعتبره فشل
    if not choice or not (1 <= choice <= len(candidates)):
        return {"choice": None, "reason": parsed.get("reason")}

    best = candidates[choice - 1]
    return {
        "choice": choice,
        "doc_id": best["doc_id"],
        "subject": best["subject"],
        "reason": parsed.get("reason"),
    }


# ══════════════════════ واجهة الاستخدام من الـ Backend ══════════════════════

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
    """بحث + (اختيارياً) اقتراح LLM — بغلاف جاهز للـ JSON response."""
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
        "message": None if results else "ما في نماذج عقود مطابقة لطلبك.",
        "params_used": p,
        "took_ms": int((time.time() - t0) * 1000),
    }


def get_contract(doc_id: Optional[str] = None, subject: Optional[str] = None) -> Optional[Dict]:
    return get_rag().get_contract(doc_id=doc_id, subject=subject)


def is_loaded() -> bool:
    return _instance is not None
