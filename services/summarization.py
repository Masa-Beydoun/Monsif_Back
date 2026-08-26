# -*- coding: utf-8 -*-
"""
Arabic Legal Case Pipeline — Production Ready
=================================================
نسخة إنتاجية كاملة بملف Python واحد.
تجمع: التنظيف + التلخيص + استخراج الحقول + استخراج الوقائع (اختياري) + استخراج الكيانات (اختياري)

المكتبات المطلوبة:
    pip install pyarabic scikit-learn networkx numpy nltk rouge-score -q
    
الاستخدام السريع:
    from legal_summarizer_final import IntelligentLegalPipeline
    
    pipeline = IntelligentLegalPipeline(enable_ai=False)
    result = pipeline.analyze("النص القانوني هنا...")
    print(result['analysis']['extractive_summary'])
"""

import re
import os
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import pyarabic.araby as araby
    HAS_PYARABIC = True
except ImportError:
    HAS_PYARABIC = False

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    try:
        _ = nltk_stopwords.words("arabic")
        HAS_NLTK_STOPWORDS = True
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            _ = nltk_stopwords.words("arabic")
            HAS_NLTK_STOPWORDS = True
        except Exception:
            HAS_NLTK_STOPWORDS = False
except ImportError:
    HAS_NLTK_STOPWORDS = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


# =============================================================================
# [1] STOPWORDS
# =============================================================================

LEGAL_STOPWORDS = {
    "محكمة", "المحكمة", "قرار", "رقم", "تاريخ", "القاضي",
    "باسم", "الشعب", "العربي", "السوري", "قانون", "مادة",
    "بناء", "عليه", "حيث", "ان", "إن", "لذلك", "قررت",
    "الدعوى", "الأساس", "الغرفة", "الجزائية", "المدنية",
}

GENERAL_STOPWORDS_FALLBACK = {
    "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "التي", "الذي",
    "و", "أو", "ثم", "كان", "كانت", "يكون", "أن", "لا", "ما", "لم", "لن",
    "قد", "بعد", "قبل", "عند", "كل", "بعض", "غير", "بين", "حتى", "إذا", "كما",
    "له", "لها", "لهم", "به", "بها", "بهم", "هو", "هي", "هم", "أنا", "نحن",
}

CUE_PHRASE_CATEGORIES = {
    "confession_denial": (r"اعترف|انكر|نفي|طعن|اسقط|اقر|ادعي", 1.6),
    "arrest_action":     (r"القت القبض|القي القبض|تم توقيف|داهمت|كمشت|قبضت", 1.5),
    "ruling":            (r"قرر القاضي|حكمت المحكمه|قررت المحكمه|بناء علي ما تقدم|قرر توقيف|الزمت المحكمه", 1.7),
    "roles":             (r"المتهم|المجني عليه|الشاهد|الشهود|المشتكي|المدعي|المدعي عليه", 1.2),
    "evidence":          (r"دليل|اداه|سلاح|بصمات|كاميرا|شهاده|سند", 1.3),
    "date":              (r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", 1.4),
}


# =============================================================================
# [2] PREPROCESSOR
# =============================================================================

class UniversalPreprocessor:
    """معالج نصوص عربية شامل: تنظيف + إزالة stopwords + tokenization"""

    def __init__(self):
        general_sw = set(nltk_stopwords.words("arabic")) if HAS_NLTK_STOPWORDS else GENERAL_STOPWORDS_FALLBACK
        self.all_stopwords = list(general_sw.union(LEGAL_STOPWORDS))

    def clean_text(self, text_input: str) -> str:
        """تنظيف شامل: حذف التشكيل، توحيد الأشكال، إزالة الفراغات الزائدة"""
        if not isinstance(text_input, str):
            return ""
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text_input)  # التشكيل
        text = re.sub(r"ـ", "", text)                             # التطويل
        text = re.sub(r"[أإآ]", "ا", text)                        # الألف
        text = re.sub(r"ة", "ه", text)                            # التاء المربوطة
        text = re.sub(r"ى", "ي", text)                            # الألف المقصورة
        text = re.sub(r"ؤ", "ء", text)
        text = re.sub(r"ئ", "ء", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_stopwords(self, text: str) -> str:
        """إزالة الكلمات الشائعة والقانونية"""
        words = text.split()
        filtered = [w for w in words if w not in self.all_stopwords]
        return " ".join(filtered)

    def tokenize(self, text: str) -> List[str]:
        """تقسيم النص إلى كلمات"""
        if HAS_PYARABIC:
            return araby.tokenize(text)
        return text.split()


# =============================================================================
# [3] LEGAL SUMMARIZER (TextRank + MMR + Cue Phrases)
# =============================================================================

class LegalSummarizer:
    """
    خوارزمية تلخيص استخلاصية (extractive) لـ نصوص قانونية.
    يستخدم TextRank + MMR + ترجيح العبارات المهمة قانونياً.
    
    ملاحظة: التلخيص يعود كـ نص مستمر، ليس bullets
    """

    def __init__(self, preprocessor: UniversalPreprocessor = None):
        self.preprocessor = preprocessor or UniversalPreprocessor()

    # علامات نهاية الجملة الأساسية
    _PRIMARY_SPLIT_RE = re.compile(r"(?<=[.؟!۔])\s+|\n+")
    # فواصل ثانوية للنصوص التي تُرسل ككتلة واحدة بدون نقاط
    _SECONDARY_SPLIT_RE = re.compile(
        r"(?<=[؛;:])\s+"
        r"|\s+(?=(?:وحيث|حيث|وبما|بما|لذلك|وعليه|عليه|وبناء|بناء|قررت|تقرر|"
        r"وقد|كما|أما|اما|ثم|وفي|فقد)\s)"
    )
    _MAX_CHUNK = 300    # أي مقطع أطول من هذا يُقسَّم أكثر
    _COMMA_CHUNK = 200  # الطول التقريبي عند التقسيم على الفاصلة
    _MIN_FRAGMENT = 40  # المقاطع الأقصر من هذا تُدمج مع ما قبلها

    def split_into_sentences(self, text: str) -> List[str]:
        """
        تقسيم النص إلى جمل على ثلاثة مستويات:
        1) علامات الترقيم (. ؟ ! ۔) والأسطر الجديدة.
        2) إذا وصل النص ككتلة واحدة بلا نقاط (شائع عند الإرسال من الواجهة
           الأمامية): تقسيم على (؛ ; :) وعلى أدوات الربط القانونية.
        3) كحل أخير: تقسيم على الفاصلة (،) بمقاطع ~200 حرف.

        بدون المستويين 2 و 3 يعود النص الطويل كجملة واحدة، فيُعاد كما هو
        بلا أي تلخيص.
        """
        parts = [s.strip() for s in self._PRIMARY_SPLIT_RE.split(text.strip()) if s.strip()]

        sentences: List[str] = []
        for part in parts:
            if len(part) <= self._MAX_CHUNK:
                sentences.append(part)
                continue
            subs = [s.strip() for s in self._SECONDARY_SPLIT_RE.split(part) if s.strip()]
            for sub in (subs if len(subs) > 1 else [part]):
                sentences.extend(self._split_long_chunk(sub))

        return self._merge_fragments([s for s in sentences if len(s) > 10])

    def _merge_fragments(self, sentences: List[str]) -> List[str]:
        """دمج المقاطع القصيرة جداً مع ما قبلها حتى لا يخرج الملخص مبتوراً"""
        merged: List[str] = []
        for sentence in sentences:
            if (
                merged
                and len(sentence) < self._MIN_FRAGMENT
                and len(merged[-1]) + len(sentence) <= self._MAX_CHUNK
            ):
                merged[-1] = merged[-1] + " " + sentence
            else:
                merged.append(sentence)
        return merged

    def _split_long_chunk(self, chunk: str) -> List[str]:
        """تقسيم مقطع طويل على الفاصلة (أو على الكلمات) بمقاطع متقاربة الطول"""
        if len(chunk) <= self._MAX_CHUNK:
            return [chunk]

        units = [u.strip() for u in re.split(r"،|,", chunk) if u.strip()]
        if len(units) <= 1:
            units = chunk.split()

        out, current = [], ""
        for unit in units:
            candidate = (current + " " + unit).strip()
            if current and len(candidate) > self._COMMA_CHUNK:
                out.append(current)
                current = unit
            else:
                current = candidate
        if current:
            out.append(current)
        return out

    def _cue_score(self, clean_sentence: str) -> float:
        """حساب درجة الأهمية بناءً على العبارات الحساسة قانونياً"""
        score = 1.0
        for _, (pattern, weight) in CUE_PHRASE_CATEGORIES.items():
            if re.search(pattern, clean_sentence):
                score *= weight
        return score

    def summarize(self, text: str, base_compression_ratio: float = 0.5, 
                  max_sentences: int = 10, min_sentences_for_compression: int = 7,
                  use_mmr: bool = True, mmr_lambda: float = 0.7) -> str:
        """
        ملخص استخلاصي للنص.
        
        المعاملات:
            - base_compression_ratio: نسبة الانضغاط (0.5 = 50% من الجمل الأصلية)
            - max_sentences: أقصى عدد جمل في الملخص
            - min_sentences_for_compression: إذا كان النص أقل من هذا، لا نلخصه
            - use_mmr: استخدام Maximal Marginal Relevance (تنوّع + ارتباط)
            - mmr_lambda: وزن الارتباط مقابل التنوع في MMR
            
        العودة:
            - نص مستمر (لا bullets)
        """
        summary_text, _, _ = self.summarize_with_indices(
            text, base_compression_ratio, max_sentences,
            min_sentences_for_compression, use_mmr, mmr_lambda,
        )
        return summary_text

    def summarize_with_indices(self, text: str, base_compression_ratio: float = 0.5,
                                max_sentences: int = 10, 
                                min_sentences_for_compression: int = 7,
                                use_mmr: bool = True, mmr_lambda: float = 0.7) \
            -> Tuple[str, List[int], List[str]]:
        """
        نفس summarize() لكن يرجع أيضاً مؤشرات الجمل المختارة (للتقييم).
        
        العودة:
            - (summary_text, selected_indices, all_sentences)
        """
        sentences = self.split_into_sentences(text)
        total_sentences = len(sentences)

        # لا نلخّص فقط إذا كان النص قصيراً فعلاً (عدد جمل قليل ونص قصير)
        if total_sentences <= 1 or (
            total_sentences <= min_sentences_for_compression and len(text) < 1200
        ):
            return (
                "\n".join("- " + s for s in sentences),
                list(range(total_sentences)),
                sentences,
            )

        # حساب عدد الجمل المستهدفة
        target_length = int(total_sentences * base_compression_ratio)
        target_length = min(target_length, max_sentences)
        target_length = max(target_length, 4)

        # تنظيف الجمل
        clean_sentences = [self.preprocessor.clean_text(s) for s in sentences]

        # TF-IDF vectorization
        # ملاحظة: نُنظّف الـ stopwords بنفس طريقة تنظيف الجمل حتى تتطابق معها
        stop_words = sorted({
            self.preprocessor.clean_text(w) for w in self.preprocessor.all_stopwords
        } - {""})
        try:
            vectorizer = TfidfVectorizer(stop_words=stop_words)
            X = vectorizer.fit_transform(clean_sentences)
        except ValueError:
            # كل الكلمات كانت stopwords -> نُعيد المحاولة بدونها
            try:
                X = TfidfVectorizer().fit_transform(clean_sentences)
            except ValueError:
                # لا يمكن بناء أي مفردات: نُعيد أول الجمل كملخص
                fallback = sorted(range(total_sentences))[:target_length]
                return (
                    "\n".join("- " + sentences[i] for i in fallback),
                    fallback,
                    sentences,
                )

        # حساب مصفوفة التشابه
        sim_matrix = (X * X.T).toarray()
        np.fill_diagonal(sim_matrix, 0)

        # TextRank (PageRank على الرسم البياني)
        nx_graph = nx.from_numpy_array(sim_matrix)
        try:
            scores = nx.pagerank(nx_graph)
        except nx.PowerIterationFailedConvergence:
            scores = {i: 1.0 / total_sentences for i in range(total_sentences)}

        # ترجيح العبارات الحساسة قانونياً
        for i, clean_sentence in enumerate(clean_sentences):
            scores[i] *= self._cue_score(clean_sentence)

        # اختيار الجمل
        if use_mmr:
            selected = self._mmr_select(scores, sim_matrix, target_length, mmr_lambda)
        else:
            ranked = sorted(((scores[i], i) for i in range(total_sentences)), reverse=True)
            selected = [i for _, i in ranked[:target_length]]

        # ترتيب الجمل المختارة حسب ظهورها الأصلي
        selected = sorted(selected)
        summary_text = "\n".join(f"- {sentences[i]}" for i in selected)
        
        return summary_text, selected, sentences

    def _mmr_select(self, scores: Dict[int, float], sim_matrix: np.ndarray, 
                    top_k: int, lambda_param: float) -> List[int]:
        """
        Maximal Marginal Relevance: اختيار جمل عالية الارتباط وقليلة التكرار
        """
        n = len(scores)
        top_k = min(top_k, n)
        selected, candidates = [], list(range(n))

        while len(selected) < top_k and candidates:
            if not selected:
                # الجملة الأولى: الأعلى درجة
                best = max(candidates, key=lambda i: scores[i])
            else:
                # الجمل التالية: توازن بين الارتباط والتنوّع
                def mmr_score(i):
                    redundancy = max(sim_matrix[i][j] for j in selected)
                    return lambda_param * scores[i] - (1 - lambda_param) * redundancy
                best = max(candidates, key=mmr_score)
            
            selected.append(best)
            candidates.remove(best)
        
        return selected


# =============================================================================
# [4] STRUCTURED FIELD EXTRACTOR
# =============================================================================

class StructuredFieldExtractor:
    """استخراج الحقول المهمة: التاريخ، المتهم، المجني عليه، إلخ (regex فقط)"""

    def extract(self, text: str) -> Dict:
        fields = {
            "date": None,
            "accused": [],
            "victim": None,
            "action": None,
            "evidence": [],
            "confession_status": [],
            "ruling": None,
        }

        # التاريخ
        # التاريخ (مُعدّل لالتقاط تاريخ الواقعة/الادعاء وتجاهل ترويسة الجلسة)
        fields["date"] = None
        
        # 1. نبحث أولاً عن التواريخ المرتبطة بالوقائع (مسبوقة بكلمة بتاريخ، الواقع في، إلخ)
        incident_date_match = re.search(r"(?:بتاريخ|حاصلة بتاريخ|الواقع في)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
        
        if incident_date_match:
            fields["date"] = incident_date_match.group(1)
        else:
            # 2. كبديل، نستخرج كل التواريخ مع السياق الذي يسبقها بـ 15 حرفاً
            all_dates = re.finditer(r"(.{0,15})(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
            for d_match in all_dates:
                context_before = d_match.group(1)
                # نستبعد التواريخ إذا كان قبلها كلمات تدل على المحكمة
                if not any(word in context_before for word in ["الجلسة", "القرار", "الحكم"]):
                    fields["date"] = d_match.group(2)
                    break
            
            # 3. خط الدفاع الأخير: إذا لم نجد أي تاريخ يحقق الشروط، نأخذ أول تاريخ متاح
            if not fields["date"]:
                first_date = re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)
                if first_date:
                    fields["date"] = first_date.group()

        # المتهم/المدعى عليه
        party_pattern = r"(?:المتهم(?:\s+(?:الأول|الثاني|الثالث))?|المدعى عليه)"
        fields["accused"] = list(dict.fromkeys(re.findall(party_pattern, text)))

        # المجني عليه/المدعي
        if re.search(r"المجني عليه", text):
            fields["victim"] = "المجني عليه"
        elif re.search(r"المدعي(?!\s*عليه)", text):
            fields["victim"] = "المدعي"

        # حالة الاعتراف/الإنكار
        # حالة الاعتراف/الإنكار (مُعدّلة لتجنب التداخل)
        seen = set()
        
        # 1. حالة (المتهم ... اعترف/أنكر): مسافة لحد 80 حرف بشرط عدم وجود متهم آخر في المنتصف
        pattern_party_first = rf"({party_pattern})(?:(?!{party_pattern})[^\.]){{0,80}}?(اعترف|أنكر|انكر)"
        
        # 2. حالة (اعترف/أنكر ... المتهم): مسافة قصيرة جداً (لحد 12 حرف) لأن الفاعل يأتي فوراً بعد الفعل
        pattern_verb_first = rf"(اعترف|أنكر|انكر)(?:(?!{party_pattern})[^\.]){{0,12}}?({party_pattern})"

        for pattern in [pattern_party_first, pattern_verb_first]:
            for m in re.finditer(pattern, text):
                g1, g2 = m.group(1), m.group(2)
                # تحديد من هو الفاعل ومن هو الفعل بناءً على من جاء أولاً
                who, verb = (g2, g1) if g1 in ("اعترف", "أنكر", "انكر") else (g1, g2)
                
                if (who, verb) not in seen:
                    seen.add((who, verb))
                    fields["confession_status"].append(f"{who}: {verb}")

        # الأدلة
        fields["evidence"] = list(dict.fromkeys(
            re.findall(r"أداة\s+[^\s.،؛!؟]+|سلاح\s*[^\s.،؛!؟]*|بصمات|كاميرا|سند\s+[^\s.،؛!؟]+", text)
        ))

        # الحكم/القرار
        ruling_match = re.search(
            r"(قرر القاضي[^\.]*\.)|(حكمت المحكمة[^\.]*\.)|(قررت المحكمة[^\.]*\.)|(ألزمت المحكمة[^\.]*\.)",
            text,
        )
        if ruling_match:
            fields["ruling"] = ruling_match.group().strip()

        # الفعل/الجرم
        action_match = re.search(
            r"(سرق[^\.]*\.|ضرب[^\.]*\.|مشاجرة[^\.]*\.|اعتدى[^\.]*\.|احتيال[^\.]*\.)", text
        )
        if action_match:
            fields["action"] = action_match.group().strip()

        return fields


# =============================================================================
# [5] FACT EXTRACTOR (اختياري - يحتاج Stanza)
# =============================================================================

class FactExtractor:
    """استخراج الوقائع (ثلاثيات: فاعل → فعل → مفعول) باستخدام Stanza"""

    def __init__(self, use_gpu: bool = False):
        self._nlp = None
        self._available = False
        try:
            import torch
            import stanza
            use_gpu = use_gpu and torch.cuda.is_available()
            stanza.download("ar", verbose=False)
            self._nlp = stanza.Pipeline(
                "ar", processors="tokenize,mwt,pos,lemma,depparse",
                use_gpu=use_gpu, verbose=False
            )
            self._available = True
        except Exception as e:
            pass  # التخطي بصمت إذا فشل

    @property
    def available(self) -> bool:
        return self._available

    def extract_facts(self, text: str) -> List[str]:
        """استخراج الوقائع بصيغة (فاعل ➔ فعل ➔ مفعول)"""
        if not self._available or not text or len(text.strip()) < 5:
            return []

        doc = self._nlp(text)
        facts = []
        for sentence in doc.sentences:
            verbs = [w for w in sentence.words if w.upos == "VERB"]
            for verb in verbs:
                subject_text = "مستتر/محذوف"
                objects = []
                for word in sentence.words:
                    if word.head == verb.id:
                        if word.deprel in ["nsubj", "nsubj:pass", "csubj"]:
                            subject_text = word.text
                        elif word.deprel in ["obj", "iobj", "obl", "obl:arg", "xcomp", "ccomp"]:
                            objects.append(word.text)
                
                object_text = " ".join(objects) if objects else "غير محدد"
                if subject_text != "مستتر/محذوف" or object_text != "غير محدد":
                    facts.append(f"({subject_text} ➔ {verb.lemma} ➔ {object_text})")
        
        return facts


# =============================================================================
# [6] ENTITY EXTRACTOR (اختياري - يحتاج Hugging Face)
# =============================================================================

class EntityExtractor:
    """استخراج الكيانات (أشخاص، أماكن، منظمات، تواريخ، إلخ)"""

    def __init__(self):
        self._ner_pipeline = None
        self._available = False
        try:
            import torch
            from transformers import pipeline as hf_pipeline
            device = 0 if torch.cuda.is_available() else -1
            self._ner_pipeline = hf_pipeline(
                "ner", model="hatmimoha/arabic-ner", 
                aggregation_strategy="simple", device=device
            )
            self._available = True
        except Exception as e:
            pass  # التخطي بصمت إذا فشل

        self.date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        self.money_pattern = (
            r"(?:\d+\s*(?:مليون|ألف|ليرة|دولار|يورو))"
            r"|(?:(?:مبلغاً وقدره|مبلغ|قدره)\s+([أ-ي\s]+(?:ليرة|سورية|دولار|يورو)))"
        )
        self.charge_pattern = r"(?:بجرم|تهمة|بجناية|بجنحة)\s+([أ-ي]+(?:\s+(?!و|في|على|من|إلى)[أ-ي]+)?)"

    @property
    def available(self) -> bool:
        return self._available

    def extract_entities(self, text: str) -> Dict:
        """استخراج جميع الكيانات المهمة من النص"""
        entities = {
            "الأشخاص": set(),
            "الأماكن": set(),
            "المنظمات": set(),
            "التواريخ": set(),
            "المبالغ_المالية": set(),
            "التهم_والجرائم": set(),
        }
        
        if not text or len(text.strip()) < 2:
            return {}

        # استخراج من نموذج NER إذا كان متوفراً
        if self._available:
            ai_results = self._ner_pipeline(text)
            for ent in ai_results:
                start, end = ent.get("start"), ent.get("end")
                if start is not None and end is not None:
                    true_start = text.rfind(" ", 0, start)
                    true_start = true_start + 1 if true_start != -1 else 0
                    true_end = text.find(" ", end)
                    true_end = true_end if true_end != -1 else len(text)
                    word = text[true_start:true_end].strip()
                    word = re.sub(r"^[،.؛!؟,]+|[،.؛!؟,]+$", "", word).strip()
                else:
                    word = ent.get("word", "").replace("##", "").strip()

                if len(word) <= 1:
                    continue

                group = ent.get("entity_group", "")
                if group in ["PER", "PERSON"]:
                    entities["الأشخاص"].add(word)
                elif group in ["LOC", "LOCATION"]:
                    entities["الأماكن"].add(word)
                elif group in ["ORG", "ORGANIZATION"]:
                    entities["المنظمات"].add(word)

        # استخراج بـ Regex (يعمل دائماً)
        entities["التواريخ"].update(re.findall(self.date_pattern, text))
        for m in re.finditer(self.money_pattern, text):
            entities["المبالغ_المالية"].add((m.group(1) or m.group(0)).strip())
        for m in re.finditer(self.charge_pattern, text):
            entities["التهم_والجرائم"].add(m.group(1).strip())

        return {k: list(v) for k, v in entities.items() if v}


# =============================================================================
# [7] INTELLIGENT LEGAL PIPELINE (الـ Pipeline الرئيسي)
# =============================================================================

class IntelligentLegalPipeline:
    """
    الـ Pipeline الكامل: تنظيف + تلخيص + استخراج حقول + وقائع + كيانات
    """

    def __init__(self, enable_ai: bool = False):
        """
        المعاملات:
            - enable_ai: تفعيل استخراج الوقائع والكيانات (يحتاج مكتبات إضافية)
        """
        self.preprocessor = UniversalPreprocessor()
        self.summarizer = LegalSummarizer(self.preprocessor)
        self.field_extractor = StructuredFieldExtractor()
        
        self.fact_extractor = FactExtractor() if enable_ai else None
        self.entity_extractor = EntityExtractor() if enable_ai else None

    def analyze(self, raw_text: str) -> Dict:
        """
        تحليل شامل للنص القانوني.
        
        العودة:
            {
                "status": "success" or "error",
                "original_length": int,
                "analysis": {
                    "extractive_summary": str,  # النص الملخص (لا bullets)
                    "structured_fields": {...},  # التاريخ، المتهم، إلخ
                    "facts_triples": [...],      # الوقائع (اختياري)
                    "entities": {...}            # الكيانات (اختياري)
                }
            }
        """
        if not raw_text or len(raw_text.strip()) < 10:
            return {
                "status": "error",
                "error": "النص المدخل قصير جداً ولا يمكن تحليله."
            }

        clean_text = self.preprocessor.clean_text(raw_text)
        summary = self.summarizer.summarize(raw_text)
        structured_fields = self.field_extractor.extract(raw_text)

        # استخراج الوقائع (اختياري)
        facts = []
        if self.fact_extractor and self.fact_extractor.available:
            facts = self.fact_extractor.extract_facts(summary)

        # استخراج الكيانات (اختياري)
        entities = {}
        if self.entity_extractor and self.entity_extractor.available:
            entities = self.entity_extractor.extract_entities(clean_text)
        else:
            # تطبيق Regex fallback دائماً
            fallback_extractor = EntityExtractor.__new__(EntityExtractor)
            fallback_extractor._available = False
            fallback_extractor.date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            fallback_extractor.money_pattern = (
                r"(?:\d+\s*(?:مليون|ألف|ليرة|دولار|يورو))"
                r"|(?:(?:مبلغاً وقدره|مبلغ|قدره)\s+([أ-ي\s]+(?:ليرة|سورية|دولار|يورو)))"
            )
            fallback_extractor.charge_pattern = r"(?:بجرم|تهمة|بجناية|بجنحة)\s+([أ-ي]+(?:\s+(?!و|في|على|من|إلى)[أ-ي]+)?)"
            entities = fallback_extractor.extract_entities(clean_text)

        return {
            "status": "success",
            "original_length": len(raw_text),
            "analysis": {
                "extractive_summary": summary,
                "structured_fields": structured_fields,
                "facts_triples": facts,
                "entities": entities,
            }
        }


# =============================================================================
# [8] EVALUATION UTILITIES (اختياري - للتقييم)
# =============================================================================

def compute_rouge(predicted_summary: str, gold_summary: str) -> Dict:
    """حساب درجات ROUGE إذا كانت المكتبة متوفرة"""
    if not HAS_ROUGE:
        return {"status": "error", "message": "rouge-score not installed"}
    
    preprocessor = UniversalPreprocessor()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    
    pred_norm = preprocessor.clean_text(predicted_summary)
    gold_norm = preprocessor.clean_text(gold_summary)
    
    scores = scorer.score(gold_norm, pred_norm)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 3),
        "rouge2_f": round(scores["rouge2"].fmeasure, 3),
        "rougeL_f": round(scores["rougeL"].fmeasure, 3),
    }


# =============================================================================
# TEST CASES - أمثلة للاختبار
# =============================================================================

TEST_CASE_1 = """
الزلمة فات عالدكانة وسرق المصاري من الدرج. أنا شفته بعيني عم يركض بالشارع وبعدين تخبى. 
الشرطة كمشته المسا واعترف بكل شي. تم العثور على المصاري في حقيبته. شهد عليه صاحب الدكانة. 
قررت المحكمة توقيفه لمدة 6 أشهر.
"""

TEST_CASE_2 = """
بتاريخ 20/05/2023، ورد إخبار إلى قسم الشرطة يفيد بوقوع مشاجرة جماعية في الساحة العامة. 
وتوجهت الدوريات فوراً إلى المكان حيث ألقت القبض على المتورطين في الحادثة. وأفاد الشهود أن المتهم الأول 
بادر بضرب المجني عليه باستخدام أداة حادة. وأثناء التحقيق، أنكر المتهم الأول التهمة المنسوبة إليه. 
من جهة أخرى، اعترف المتهم الثاني بمشاركته في الشجار وتكسير الواجهة الزجاجية. وبناءً على ما تقدم، 
قرر القاضي توقيف المتهمين ومصادرة الأداة المستخدمة في الاعتداء.
"""

TEST_CASE_3 = """
قضية احتيال تاريخ الحكم 15/03/2024. المتهم محمد علي أحمد اتهم بخديعة وتزوير وثائق رسمية.
المدعي عليه قام بتقديم عقود وهمية لاستقطاع أموال من ثلاث ضحايا. تم العثور على الأدلة التالية:
نسخ من العقود الزيفة، بريد إلكتروني يثبت التواصل، تسجيلات لمكالمات هاتفية، بصمات على الأوراق.
المتهم أنكر كل التهم في البداية ثم اعترف اعترافاً جزئياً. المجني عليهم وافقوا على تسوية بمبلغ 5 مليون ليرة.
ألزمت المحكمة المتهم بدفع التعويضات وقررت توقيفه لمدة سنتين.
"""


def run_tests():
    """تشغيل أمثلة الاختبار"""
    print("\n" + "=" * 80)
    print("LEGAL SUMMARIZER - FINAL TEST CASES")
    print("=" * 80)
    
    pipeline = IntelligentLegalPipeline(enable_ai=False)
    
    test_cases = [
        ("Test Case 1: Simple Theft", TEST_CASE_1),
        ("Test Case 2: Public Brawl", TEST_CASE_2),
        ("Test Case 3: Fraud Case", TEST_CASE_3),
    ]
    
    for title, text in test_cases:
        print(f"\n\n{'─' * 80}")
        print(f"📋 {title}")
        print(f"{'─' * 80}")
        
        result = pipeline.analyze(text)
        
        if result["status"] == "error":
            print(f"❌ Error: {result['error']}")
            continue
        
        analysis = result["analysis"]
        
        print(f"\n📄 Original Text Length: {result['original_length']} characters")
        print(f"\n✂️  EXTRACTIVE SUMMARY :")
        print(f"   {analysis['extractive_summary']}")
        
        print(f"\n📋 STRUCTURED FIELDS:")
        fields = analysis["structured_fields"]
        print(f"   Date: {fields.get('date', 'N/A')}")
        print(f"   Accused: {', '.join(fields.get('accused', [])) or 'N/A'}")
        print(f"   Victim: {fields.get('victim', 'N/A')}")
        print(f"   Action: {fields.get('action', 'N/A')}")
        print(f"   Evidence: {', '.join(fields.get('evidence', [])) or 'N/A'}")
        print(f"   Confession Status: {', '.join(fields.get('confession_status', [])) or 'N/A'}")
        print(f"   Ruling: {fields.get('ruling', 'N/A')}")
        
        if analysis.get("entities"):
            print(f"\n🏷️  ENTITIES:")
            for entity_type, values in analysis["entities"].items():
                if values:
                    print(f"   {entity_type}: {', '.join(values)}")
        
        print()


if __name__ == "__main__":
    run_tests()