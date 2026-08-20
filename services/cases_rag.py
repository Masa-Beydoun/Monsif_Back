"""
الاسترجاع الدلالي للسوابق القضائية.

مسار البحث: FAISS dense مع BM25 sparse ← دمج RRF ← إعادة ترتيب بالـ
cross-encoder ← تطبيق عتبة التشابه.

النماذج مشتركة عبر model_registry، وبارامترات top_k و threshold و hybrid_top_k
قابلة للتمرير مع كل استدعاء.
"""

import json
import os
import pickle
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

import config
from services import model_registry


class HybridVectorStore:
    """محرك بحث هجين (Dense FAISS + Sparse BM25 + RRF Fusion).

    محايد تجاه نوع العناصر؛ يعتمد على text_builder المُمرَّر إليه.
    """

    def __init__(self, text_builder: Callable[[Dict], str]):
        self.text_builder = text_builder
        self.index = None
        self.bm25 = None
        self.items: List[Dict] = []
        self.texts: List[str] = []

    @property
    def model(self):
        # تحميل كسول: يُحمَّل النموذج عند الحاجة الفعلية لا عند الإنشاء.
        return model_registry.get_dense_encoder()

    def build(self, items: List[Dict], batch_size: int = 32) -> None:
        import faiss
        from rank_bm25 import BM25Okapi

        self.items = items
        self.texts = [self.text_builder(it) for it in items]
        print(f"[cases] تجهيز {len(items)} عنصر ...", flush=True)

        embeddings = self.model.encode(
            self.texts, show_progress_bar=True,
            normalize_embeddings=True, batch_size=batch_size,
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        print("[cases] بناء فهرس BM25 ...", flush=True)
        self.bm25 = BM25Okapi([doc.split(" ") for doc in self.texts])
        print(f"[cases] الفهرس الهجين جاهز: {self.index.ntotal} عنصر", flush=True)

    def save(self, index_file: str, metadata_file: str) -> None:
        import faiss

        faiss.write_index(self.index, index_file)
        with open(metadata_file, "wb") as f:
            pickle.dump({"items": self.items, "texts": self.texts, "bm25": self.bm25}, f)
        print(f"[cases] حُفظ → {index_file} + {metadata_file}", flush=True)

    def load(self, index_file: str, metadata_file: str) -> None:
        import faiss

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"لم يُعثر على فهرس القضايا: {index_file}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"لم يُعثر على بيانات القضايا الوصفية: {metadata_file}")
        self.index = faiss.read_index(index_file)
        with open(metadata_file, "rb") as f:
            d = pickle.load(f)
        self.items, self.texts, self.bm25 = d["items"], d["texts"], d["bm25"]
        print(f"[cases] الفهرس الهجين محمّل: {self.index.ntotal} عنصر", flush=True)

    def search_hybrid(self, query: str, top_k: int) -> List[Dict]:
        """أفضل top_k عنصر بعد دمج Dense و Sparse عبر RRF.

        يجري الدمج على الفهرس الداخلي i لا على case_number، ضماناً لصحته حتى
        عند تكرار الأرقام أو غيابها.
        """
        q_vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        _, dense_indices = self.index.search(q_vec, top_k)
        dense_rank = {int(i): rank for rank, i in enumerate(dense_indices[0]) if i >= 0}

        bm25_scores = self.bm25.get_scores(query.split(" "))
        bm25_top_n = np.argsort(bm25_scores)[::-1][:top_k]
        sparse_rank = {int(i): rank for rank, i in enumerate(bm25_top_n)}

        all_indices = set(dense_rank) | set(sparse_rank)
        fused = {
            i: (1 / (60 + dense_rank.get(i, 1000))) + (1 / (60 + sparse_rank.get(i, 1000)))
            for i in all_indices
        }
        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [self.items[i] for i in best]


class CasesRAG:
    """نظام السوابق القضائية: بحث هجين ثم إعادة ترتيب بالـ cross-encoder."""

    def __init__(self):
        self.store = HybridVectorStore(text_builder=self._build_text)
        self._ready = False

    # تجهيز نص العنصر للترميز وإعادة الترتيب.
    @staticmethod
    def _build_text(item: Dict) -> str:
        facts = item.get("facts_text", "")
        return str(facts).strip() if facts else "لا توجد وقائع مسجلة"

    @staticmethod
    def _to_result(item: Dict, score: float) -> Dict:
        outcome = item.get("outcome")
        return {
            "case_number": item.get("case_number", "N/A"),
            "file_name": item.get("file_name", ""),
            "similarity_score": score,
            "decision_year": item.get("decision_year", ""),
            "outcome": " | ".join(outcome) if isinstance(outcome, list) else str(outcome or ""),
            "crimes": item.get("crimes", []),
            "penal_code_articles": item.get("penal_code_articles", []),
            "other_laws": item.get("other_laws", []),
            "facts_text": item.get("facts_text", ""),
            "claims_text": item.get("claims_text", ""),
            "reasoning_text": item.get("reasoning_text", ""),
            "judgment_text": item.get("judgment_text", ""),
        }

    # التحميل والبناء

    def load_index(self) -> None:
        self.store.load(config.CASES_INDEX_FILE, config.CASES_METADATA_FILE)
        self._ready = True

    def build_from_json(self, json_path: str, save: bool = True) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = raw if isinstance(raw, list) else [raw]
        self.store.build(items, batch_size=config.CASES_DEFAULTS["encode_batch_size"])
        if save:
            self.store.save(config.CASES_INDEX_FILE, config.CASES_METADATA_FILE)
        self._ready = True

    def auto_load(self) -> None:
        """يحمّل الفهرس المحفوظ إن وُجد، وإلا بناه من ملف JSON وحفظه."""
        if os.path.exists(config.CASES_INDEX_FILE) and os.path.exists(config.CASES_METADATA_FILE):
            self.load_index()
        elif os.path.exists(config.CASES_SOURCE_JSON):
            print("[cases] لا يوجد فهرس محفوظ؛ يجري البناء من ملف JSON ...", flush=True)
            self.build_from_json(config.CASES_SOURCE_JSON)
        else:
            raise FileNotFoundError(
                f"لم يُعثر على فهرس السوابق ولا على ملف JSON لبنائه.\n"
                f"  المتوقع: {config.CASES_INDEX_FILE}\n"
                f"  أو:      {config.CASES_SOURCE_JSON}"
            )

    # البحث

    def query_raw(self, query_text: str, **kw) -> List[Dict]:
        """يعيد قائمة النتائج مباشرة؛ يستعملها محرك الحكم الأولي."""
        if not self._ready:
            raise RuntimeError("فهرس القضايا غير محمّل.")

        p = dict(config.CASES_DEFAULTS)
        p.update({k: v for k, v in kw.items() if v is not None})

        candidates = self.store.search_hybrid(query_text, top_k=p["hybrid_top_k"])
        if not candidates:
            return []

        reranker = model_registry.get_reranker()
        pairs = [[query_text, self._build_text(it)] for it in candidates]
        scores = reranker.compute_score(pairs, max_length=p["rerank_max_length"])

        results = []
        for item, raw_score in zip(candidates, scores):
            normalized = round(float(1 / (1 + np.exp(-raw_score))) * 100, 1)
            if normalized < p["threshold"] * 100:
                continue
            results.append(self._to_result(item, normalized))

        results.sort(key=lambda r: r["similarity_score"], reverse=True)
        return results[:p["top_k"]]

    def query(self, query_text: str, **kw) -> Dict:
        """مثل query_raw مع غلاف جاهز لاستجابة JSON."""
        if not query_text or not query_text.strip():
            return {"query": query_text, "error": "الاستعلام فارغ.", "results": []}

        t0 = time.time()
        p = dict(config.CASES_DEFAULTS)
        p.update({k: v for k, v in kw.items() if v is not None})
        results = self.query_raw(query_text, **kw)
        return {
            "query": query_text,
            "found": bool(results),
            "count": len(results),
            "results": results,
            "message": None if results else
                       "لا توجد قضايا مشابهة تتجاوز عتبة التشابه المطلوبة.",
            "params_used": p,
            "took_ms": int((time.time() - t0) * 1000),
        }


# واجهة الاستخدام

_instance: Optional[CasesRAG] = None
_init_lock = threading.Lock()


def warmup() -> None:
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                rag = CasesRAG()
                rag.auto_load()
                _instance = rag


def get_rag() -> CasesRAG:
    if _instance is None:
        warmup()
    return _instance


def search_cases(query_text: str, **kw) -> Dict:
    return get_rag().query(query_text, **kw)


def is_loaded() -> bool:
    return _instance is not None
