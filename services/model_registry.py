"""
مستودع نماذج مشترك — Lazy Singletons.

ليش هالملف موجود:
  النوتبوك كان يحمّل نسخة BGE-M3 مستقلة لكل قسم، ونسخة reranker مستقلة كمان.
  ثلاث ميزات × نموذجين = ستة تحميلات لنفس الأوزان (~2.3GB للواحد).
  هون كل نموذج بينحمّل **مرة وحدة** وبينتشارك بين كل الميزات.

  وكمان: ما في شي بينحمّل عند الـ import. النموذج بينحمّل أول ما حدا يطلبه فعلياً،
  يعني `flask run` بيرجع فوراً، وميزة ما بتستعملها ما بتكلفك ولا بايت.
"""

import threading
import time
from typing import List, Optional

import numpy as np

import config

_lock = threading.Lock()

_dense_encoder = None       # SentenceTransformer  — dense فقط (القضايا + البحث القديم)
_bgem3_flag = None          # BGEM3FlagModel       — dense + sparse (المواد القانونية)
_reranker = None            # SharedReranker       — مشترك بين الكل


def _log(msg: str) -> None:
    print(f"[models] {msg}", flush=True)


# ══════════════════════════════ Dense encoder ══════════════════════════════

def get_dense_encoder(max_seq_length: Optional[int] = None):
    """SentenceTransformer(BGE-M3) — متجهات dense فقط.

    يستعمله: RAG السوابق (Part B) + خدمة البحث القديمة.
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


# ═══════════════════════ BGE-M3 dense + sparse (FlagEmbedding) ═══════════════════════

def get_bgem3_flag():
    """BGEM3FlagModel — بيرجّع dense **و** lexical_weights (sparse).

    يستعمله: RAG المواد القانونية فقط (Qdrant بده الاتنين للـ RRF).
    SentenceTransformer ما بيقدر ينتج lexical_weights، لهيك هاد نموذج منفصل.
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


# ═════════════════════════════════ Reranker ═════════════════════════════════

class SharedReranker:
    """Cross-encoder (bge-reranker-v2-m3) واحد لكل النظام.

    منسوخ سلوكياً عن TransformersReranker يلي بالنوتبوك: نفس الـ checkpoint،
    نفس التقطيع (تقصير الاستعلام لـ max_length*3//4 ثم تقصير النص)، ونفس
    التحويل sigmoid(logit) لما normalize=True.

    الفرق الوحيد: max_length صار بارامتر لكل استدعاء بدل ما يكون مثبّت بالبناء،
    لأن كل ميزة بالنوتبوك كانت تستعمل قيمة مختلفة (512 / 2048 / 8192).
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
        self.batch_size = 16 if self.device == "cpu" else 128
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

        scores: List[float] = []
        with self._torch.no_grad():
            for start in range(0, len(sentence_pairs), bs):
                batch = sentence_pairs[start:start + bs]
                # تقصير الاستعلام أولاً حتى وقائع طويلة ما تاكل كل النافذة،
                # وبعدها truncation='only_second' بتقصّ متن المادة.
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
                scores.extend(logits.cpu().numpy().tolist())

        if normalize:
            scores = [float(1 / (1 + np.exp(-s))) for s in scores]
        return scores

    # اسم بديل حتى الكود يلي متعوّد على CrossEncoder.predict يشتغل بدون تعديل
    def predict(self, sentence_pairs, max_length=None, **kwargs):
        return np.array(self.compute_score(sentence_pairs, max_length=max_length))


def get_reranker() -> SharedReranker:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = SharedReranker(config.RERANKER_MODEL)
    return _reranker


# ═════════════════════════════════ حالة النماذج ═════════════════════════════════

def status() -> dict:
    """شو محمّل هلق فعلياً — مفيد لـ /api/health."""
    return {
        "device": config.resolve_device(),
        "fp16": config.use_fp16(),
        "dense_encoder_loaded": _dense_encoder is not None,
        "bgem3_flag_loaded": _bgem3_flag is not None,
        "reranker_loaded": _reranker is not None,
    }
