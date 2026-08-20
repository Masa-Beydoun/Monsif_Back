"""البحث في المواد القانونية والاجتهادات القضائية (فهارس هجينة جاهزة)."""

import os
import pickle
from typing import Dict, List, Callable, Optional

import numpy as np

import config
from services import model_registry


# الإعدادات

EMBEDDING_MODEL = config.EMBEDDING_MODEL
RERANKER_MODEL  = config.RERANKER_MODEL
BASE_DIR = str(config.UTILS_DIR)


ARTICLE_INDEX_FILE    = os.path.join(BASE_DIR, "legal_articles_index.faiss")
ARTICLE_METADATA_FILE = os.path.join(BASE_DIR, "legal_articles_metadata.pkl")
JURIS_INDEX_FILE      = os.path.join(BASE_DIR, "legal_jurisprudence_index.faiss")
JURIS_METADATA_FILE   = os.path.join(BASE_DIR, "legal_jurisprudence_metadata.pkl")

TOP_K        = 3    # عدد النتائج المُعادة لكل نوع (مواد / اجتهادات)
HYBRID_TOP_K = 10    # عدد المرشحين الأوليين قبل إعادة الترتيب

ARTICLE_THRESHOLD = 0.60  # عتبة التشابه للمواد القانونية
JURIS_THRESHOLD   = 0.60  # عتبة التشابه للاجتهادات القضائية

# كلمات مفتاحية لتحديد نية المستخدم دون أي نموذج لغوي.
JURIS_KEYWORDS = [
    "اجتهاد", "الاجتهاد", "اجتهادات", "الاجتهادات",
    "سابقة قضائية", "سوابق قضائية",
    "حكم قضائي", "أحكام قضائية", "حكم محكمة", "أحكام المحاكم",
    "قرار قضائي", "قرارات قضائية", "قرار محكمة",
    "تطبيق قضائي", "تطبيقات قضائية",
]

ARTICLE_KEYWORDS = [
    "مادة", "المادة", "مواد", "المواد",
    "نص قانوني", "نصوص قانونية", "النص القانوني",
    "قانون", "القانون", "قوانين", "القوانين",
]


# دوال بناء النصوص

def build_article_text(article: Dict) -> str:
    """يحوّل قاموس المادة القانونية إلى نص غني بالسياق (لفهرس المواد)."""
    law_name     = (article.get("law_name") or "").strip()
    law_category = (article.get("law_category_raw") or article.get("law_category") or "").strip()
    article_num  = str(article.get("article_number") or "").strip()
    body         = (article.get("body_normalized") or article.get("body_raw")
                    or article.get("body") or "").strip()

    parts = []
    if law_name:
        parts.append(f"اسم القانون: {law_name}")
    if law_category:
        parts.append(f"التصنيف: {law_category}")
    if article_num:
        parts.append(f"المادة رقم: {article_num}")
    if body:
        parts.append(f"نص المادة: {body}")

    final_text = "\n".join(parts)
    return final_text if final_text.strip() else "لا يوجد نص"


def build_jtihad_text(record: Dict) -> str:
    """يحوّل سجل الاجتهاد القضائي إلى نص غني بالسياق (لفهرس الاجتهادات)."""
    parts = []
    if record.get("law_name"):
        parts.append(f"اسم القانون: {record['law_name']}")
    if record.get("article_number") not in (None, "", "N/A"):
        parts.append(f"المادة رقم: {record['article_number']}")
    if record.get("law_category"):
        parts.append(f"التصنيف: {record['law_category']}")

    jtihad_text = (record.get("jtihad_text") or "").strip()
    if jtihad_text:
        parts.append(f"نص الاجتهاد القضائي: {jtihad_text}")

    final_text = "\n".join(parts)
    return final_text if final_text.strip() else "لا يوجد نص"


def detect_query_mode(query_text: str) -> str:
    """
    يحدد نية المستخدم بدون أي LLM، بالاعتماد فقط على وجود كلمات مفتاحية:
    - 'articles'      → المستخدم يقصد نصوص المواد القانونية تحديداً
    - 'jurisprudence' → المستخدم يقصد الاجتهادات القضائية تحديداً
    - 'both'          → لم يُذكر أي منهما بوضوح، أو ذُكرا معاً → نبحث في الاثنين
    """
    q = query_text.strip()
    has_juris   = any(kw in q for kw in JURIS_KEYWORDS)
    has_article = any(kw in q for kw in ARTICLE_KEYWORDS)

    if has_juris and not has_article:
        return "jurisprudence"
    if has_article and not has_juris:
        return "articles"
    return "both"


# الفهرس الهجين

class LegalVectorStore:
    """فهرس هجين (Dense FAISS + Sparse BM25) عام، يُحمَّل من القرص فقط في وضع الـ Backend."""

    def __init__(self, text_builder: Callable[[Dict], str], model):
        self.text_builder = text_builder
        self.model = model
        self.index = None
        self.bm25 = None
        self.records: List[Dict] = []
        self.texts: List[str] = []

    def load(self, idx_path: str, meta_path: str) -> None:
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"لم يُعثر على ملف الفهرس: {idx_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"لم يُعثر على ملف البيانات الوصفية: {meta_path}")

        import faiss

        self.index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            d = pickle.load(f)
        self.records, self.texts, self.bm25 = d["records"], d["texts"], d["bm25"]
        print(f"[search] تم تحميل الفهرس: {self.index.ntotal} سجل ← {os.path.basename(idx_path)}")

    def search_hybrid(self, query: str, top_k: int = HYBRID_TOP_K) -> List[Dict]:
        """يدمج نتائج FAISS (دلالي) و BM25 (كلمات مفتاحية) عبر Reciprocal Rank Fusion."""
        if self.index is None or self.index.ntotal == 0:
            return []

        q_vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        _, dense_indices = self.index.search(q_vec, top_k)
        dense_results = {i: rank for rank, i in enumerate(dense_indices[0]) if i >= 0}

        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_n = np.argsort(bm25_scores)[::-1][:top_k]
        sparse_results = {i: rank for rank, i in enumerate(bm25_top_n)}

        fused_scores = {}
        for idx in set(dense_results.keys()).union(set(sparse_results.keys())):
            dense_rank = dense_results.get(idx, 1000)
            sparse_rank = sparse_results.get(idx, 1000)
            fused_scores[idx] = (1 / (60 + dense_rank)) + (1 / (60 + sparse_rank))

        sorted_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        return [self.records[i] for i in sorted_indices]


# محرك الاسترجاع

class LegalRAG:
    """
    يحمّل نماذج الـ Embedding/Re-ranker والفهارس المبنية مسبقاً،
    ويوفر query() الذي يوجّه الاستعلام تلقائياً إلى فهرس المواد و/أو فهرس الاجتهادات.
    """

    def __init__(self):
        # النماذج مشتركة مع باقي الميزات عبر model_registry؛ لا تُحمَّل نسخة ثانية.
        shared_model = model_registry.get_dense_encoder(max_seq_length=1024)

        self.article_store = LegalVectorStore(text_builder=build_article_text, model=shared_model)
        self.juris_store   = LegalVectorStore(text_builder=build_jtihad_text, model=shared_model)

        self.reranker = model_registry.get_reranker()
        self.rerank_max_length = 2048

        self._ready = False

    def load_index(self) -> None:
        self.article_store.load(ARTICLE_INDEX_FILE, ARTICLE_METADATA_FILE)
        self.juris_store.load(JURIS_INDEX_FILE, JURIS_METADATA_FILE)
        self._ready = True

    # البحث
    def _rank_candidates(self, query_text: str, candidates: List[Dict],
                          text_builder: Callable[[Dict], str], threshold: float,
                          top_k: int, mapper: Callable[[Dict, float], Dict]) -> List[Dict]:
        """إعادة ترتيب المرشحين بالـ CrossEncoder، واستبعاد ما دون العتبة، والاقتصار على top_k."""
        if not candidates:
            return []

        pairs = [[query_text, text_builder(c)] for c in candidates]
        scores = self.reranker.predict(pairs, max_length=self.rerank_max_length)
        if isinstance(scores, (float, np.float32)):
            scores = [scores]

        ranked = []
        for i, score in enumerate(scores):
            # تحويل الدرجة إلى احتمال عبر sigmoid
            prob = float(1 / (1 + np.exp(-score)))

            if prob <= threshold:
                continue

            normalized = round(float(prob * 100), 2)
            ranked.append(mapper(candidates[i], normalized))

        ranked.sort(key=lambda x: x["similarity_score"], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _article_mapper(art: Dict, score: float) -> Dict:
        return {
            "article_number": art.get("article_number", "N/A"),
            "law_name": art.get("law_name", ""),
            "law_category": art.get("law_category_raw") or art.get("law_category", ""),
            "status": art.get("status", ""),
            "body": art.get("body_normalized") or art.get("body_raw") or art.get("body", ""),
            "similarity_score": score,
        }

    @staticmethod
    def _jtihad_mapper(rec: Dict, score: float) -> Dict:
        return {
            "jtihad_text": rec.get("jtihad_text", ""),
            "article_number": rec.get("article_number", "N/A"),
            "law_name": rec.get("law_name", ""),
            "law_category": rec.get("law_category", ""),
            "status": rec.get("status", ""),
            "article_body": rec.get("article_body", ""),
            "similarity_score": score,
        }

    def query(self, query_text: str, top_k: int = TOP_K, mode: Optional[str] = None) -> Dict:
        """
        mode: 'articles' | 'jurisprudence' | 'both' | None (اكتشاف تلقائي من نص الاستعلام)
        يعيد قاموساً جاهزاً للتحويل إلى JSON.
        """
        if not self._ready:
            raise RuntimeError("الفهارس غير محمّلة بعد. استدعِ load_index() أولاً (أو استخدم search_legal()).")

        if not query_text or not query_text.strip():
            return {"query": query_text, "mode": "none", "error": "الاستعلام فارغ."}

        effective_mode = mode or detect_query_mode(query_text)
        response: Dict = {"query": query_text, "mode": effective_mode}

        if effective_mode in ("articles", "both"):
            candidates = self.article_store.search_hybrid(query_text, top_k=HYBRID_TOP_K)
            results = self._rank_candidates(
                query_text, candidates, build_article_text,
                ARTICLE_THRESHOLD, top_k, self._article_mapper
            )
            response["articles"] = (
                {"found": True, "results": results} if results else
                {"found": False, "message": "لا توجد مواد قانونية مطابقة تتجاوز عتبة التشابه المطلوبة."}
            )

        if effective_mode in ("jurisprudence", "both"):
            candidates = self.juris_store.search_hybrid(query_text, top_k=HYBRID_TOP_K)
            results = self._rank_candidates(
                query_text, candidates, build_jtihad_text,
                JURIS_THRESHOLD, top_k, self._jtihad_mapper
            )
            response["jurisprudence"] = (
                {"found": True, "results": results} if results else
                {"found": False, "message": "لا توجد اجتهادات قضائية مطابقة تتجاوز عتبة التشابه المطلوبة."}
            )

        return response


# واجهة الاستخدام

_rag_instance: Optional[LegalRAG] = None


def warmup() -> None:
    """تحميل النماذج والفهارس مرة واحدة.

    تُستدعى عند إقلاع الخادم كي لا يتحمّل أول مستخدم تأخير التحميل.
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LegalRAG()
        _rag_instance.load_index()


def get_rag() -> LegalRAG:
    """يعيد نسخة الـ RAG الجاهزة (Singleton)، ويحمّلها عند أول استخدام إن لم تُستدعَ warmup() مسبقاً."""
    global _rag_instance
    if _rag_instance is None:
        warmup()
    return _rag_instance


def search_legal(query_text: str, top_k: int = TOP_K, mode: Optional[str] = None) -> Dict:
    """الدالة الوحيدة التي يحتاجها الـ route.

    Args:
        query_text: نص استعلام المستخدم (اجتهاد / مادة / وقائع قضية).
        top_k: عدد النتائج المطلوبة لكل قسم (افتراضي 3).
        mode: فرض النمط يدوياً 'articles' | 'jurisprudence' | 'both'،
              أو تركه None للاكتشاف التلقائي من نص الاستعلام.

    Returns:
        dict جاهز للتحويل مباشرة إلى JSON في الـ API response.
    """
    rag = get_rag()
    return rag.query(query_text, top_k=top_k, mode=mode)


