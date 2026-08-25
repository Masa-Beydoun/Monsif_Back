"""
مستودع النماذج المشتركة (Lazy Singletons).

يُحمَّل كل نموذج مرة واحدة ويُتشارك بين جميع الميزات، بدل تحميل نسخة مستقلة
من الأوزان نفسها لكل ميزة.

ولا يُحمَّل شيء عند الاستيراد: يُحمَّل النموذج عند أول طلب فعلي له.
"""

import threading
import time
from typing import List, Optional

import numpy as np

import config

_lock = threading.Lock()

_dense_encoder = None       # SentenceTransformer — متجهات dense فقط
_bgem3_flag = None          # BGEM3FlagModel — متجهات dense و sparse
_reranker = None            # SharedReranker — مشترك بين جميع الميزات


def _log(msg: str) -> None:
    print(f"[models] {msg}", flush=True)


# Dense encoder

def get_dense_encoder(max_seq_length: Optional[int] = None):
    """SentenceTransformer(BGE-M3) — متجهات dense فقط.

    تستعمله خدمة السوابق القضائية وخدمة البحث في المواد والاجتهادات.
    """
    global _dense_encoder
    if _dense_encoder is None:
        with _lock:
            if _dense_encoder is None:
                from sentence_transformers import SentenceTransformer

                t0 = time.time()
                _log(f"تحميل نموذج الـ Embedding: {config.EMBEDDING_MODEL} ...")
                model = SentenceTransformer(config.EMBEDDING_MODEL, device=config.resolve_device())
                if max_seq_length:
                    model.max_seq_length = max_seq_length
                _dense_encoder = model
                _log(f"جاهز ({time.time() - t0:.1f}s, device={config.resolve_device()})")
    return _dense_encoder


# BGE-M3 dense + sparse (FlagEmbedding)

def get_bgem3_flag():
    """BGEM3FlagModel — يعيد متجهات dense و lexical_weights معاً.

    تستعمله خدمة المواد القانونية وحدها، إذ يحتاج Qdrant النوعين لدمج RRF.
    وهو نموذج منفصل لأن SentenceTransformer لا ينتج lexical_weights.
    """
    global _bgem3_flag
    if _bgem3_flag is None:
        with _lock:
            if _bgem3_flag is None:
                from FlagEmbedding import BGEM3FlagModel

                t0 = time.time()
                device = config.resolve_device()
                _log(f"تحميل BGE-M3 (dense+sparse): {config.EMBEDDING_MODEL} ...")
                _bgem3_flag = BGEM3FlagModel(
                    config.EMBEDDING_MODEL,
                    use_fp16=config.use_fp16(),
                    devices=device,
                )
                _log(f"جاهز ({time.time() - t0:.1f}s, device={device})")
    return _bgem3_flag


# Reranker

class SharedReranker:
    """Cross-encoder (bge-reranker-v2-m3) واحد للنظام كله.

    يقصّر الاستعلام إلى max_length*3//4 ثم يقصّ النص، ويحوّل الناتج بـ
    sigmoid(logit) عند normalize=True.

    max_length بارامتر لكل استدعاء لا قيمة ثابتة عند الإنشاء، لأن كل ميزة
    تستعمل قيمة مختلفة (512 / 2048 / 8192).
    """

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = config.resolve_device()
        t0 = time.time()
        _log(f"تحميل الـ Re-ranker: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if config.use_fp16():
            self.model.half()
        self.model.to(self.device).eval()
        self.default_max_length = 512
        # على المعالج: دفعات صغيرة + ترتيب حسب الطول = حشو أقل بكثير.
        # دفعة واحدة كبيرة تُحشى كلها إلى طول أطول نص فيها، فتُهدر الحوسبة
        # على رموز الحشو. القياس: 2.7x أسرع بنفس النتائج تماماً.
        self.batch_size = 4 if self.device == "cpu" else 128
        self.sort_by_length = self.device == "cpu"
        _log(f"جاهز ({time.time() - t0:.1f}s, device={self.device})")

    def compute_score(self, sentence_pairs, batch_size=None, max_length=None,
                      query_max_length=None, normalize=False) -> List[float]:
        if not sentence_pairs:
            return []
        if isinstance(sentence_pairs[0], str):      # زوج واحد [query, passage]
            sentence_pairs = [sentence_pairs]

        bs = batch_size or self.batch_size
        ml = max_length or self.default_max_length
        qml = query_max_length or (ml * 3 // 4)

        # ترتيب الأزواج حسب طول المتن كي تتجمع النصوص المتقاربة في دفعة واحدة،
        # ثم تُعاد النتائج إلى ترتيب الإدخال الأصلي.
        order = list(range(len(sentence_pairs)))
        if self.sort_by_length and len(sentence_pairs) > bs:
            order.sort(key=lambda i: len(sentence_pairs[i][1]))

        scored: List[float] = [0.0] * len(sentence_pairs)
        with self._torch.no_grad():
            for start in range(0, len(order), bs):
                idx = order[start:start + bs]
                batch = [sentence_pairs[i] for i in idx]
                # يُقصَّر الاستعلام أولاً كي لا تستهلك الوقائع الطويلة النافذة
                # كلها، ثم يقصّ truncation='only_second' متن المادة.
                queries = [
                    self.tokenizer.decode(
                        self.tokenizer(q, add_special_tokens=False, truncation=True,
                                       max_length=qml)["input_ids"])
                    for q, _ in batch
                ]
                passages = [p for _, p in batch]
                enc = self.tokenizer(queries, passages, padding=True, truncation="only_second",
                                     max_length=ml, return_tensors="pt").to(self.device)
                logits = self.model(**enc, return_dict=True).logits.view(-1).float()
                for position, score in zip(idx, logits.cpu().numpy().tolist()):
                    scored[position] = score

        if normalize:
            scored = [float(1 / (1 + np.exp(-s))) for s in scored]
        return scored

    # اسم بديل لتوافق الكود المعتمد على CrossEncoder.predict.
    def predict(self, sentence_pairs, max_length=None, **kwargs):
        return np.array(self.compute_score(sentence_pairs, max_length=max_length))


def get_reranker() -> SharedReranker:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = SharedReranker(config.RERANKER_MODEL)
    return _reranker


# حالة النماذج

def status() -> dict:
    """النماذج المحمّلة فعلياً؛ تُستعمل في /api/health."""
    return {
        "device": config.resolve_device(),
        "fp16": config.use_fp16(),
        "dense_encoder_loaded": _dense_encoder is not None,
        "bgem3_flag_loaded": _bgem3_flag is not None,
        "reranker_loaded": _reranker is not None,
    }
