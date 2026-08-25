"""
تلخيص نص القضية واستخراج حقول بنيوية.

منقول عن summerization-3.ipynb — الأقسام [1] Preprocessor، [2] LegalSummarizer
(TextRank + MMR + ترجيح كلمات حساسة قانونياً)، [3] StructuredFieldExtractor
(regex بحت، صفر اختلاق). تكافؤ سلوكي مؤكَّد: نفس المُخرَج حرفياً لعشرات
التركيبات (نصوص × نِسَب ضغط × تفعيل/تعطيل MMR).

النوتبوك فيه كمان قسمان اختياريان لم يُنقَلا: FactExtractor (استخراج ثالوث
فاعل-فعل-مفعول عبر Stanza dependency parsing) وEntityExtractor (NER عبر
نموذج hatmimoha/arabic-ner). قرار مقصود لا نسيان: في كل استدعاء فعلي بالنوتبوك
نفسه (خلايا "Testing" و"MY TEST") يُنشأ الـ pipeline بـ
`enable_fact_extraction=False, enable_ner=False` — أي أن مؤلف النوتبوك لم
يُشغّلهما فعلياً قط. إضافتهما لاحقاً تستلزم: تبعية stanza جديدة (تُنزّل نموذجها
الخاص) للأولى، ونموذج NER إضافي (~500MB، عبر transformers الموجودة أصلاً)
للثانية، وتعديل شكل الاستجابة في IntelligentLegalPipeline.analyze() وفي
routes/summarization_routes.py لإخراج facts_triples وentities.
"""

import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

# المكتبات الاختيارية
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

# الثوابت والقوائم
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

class UniversalPreprocessor:
    def __init__(self):
        general_sw = set(nltk_stopwords.words("arabic")) if HAS_NLTK_STOPWORDS else GENERAL_STOPWORDS_FALLBACK
        self.all_stopwords = list(general_sw.union(LEGAL_STOPWORDS))

    def clean_text(self, text_input: str) -> str:
        if not isinstance(text_input, str):
            return ""
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text_input)
        text = re.sub(r"ـ", "", text)
        text = re.sub(r"[أإآ]", "ا", text)
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ؤ", "ء", text)
        text = re.sub(r"ئ", "ء", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

class LegalSummarizer:
    def __init__(self, preprocessor: UniversalPreprocessor = None):
        self.preprocessor = preprocessor or UniversalPreprocessor()

    def split_into_sentences(self, text: str):
        sentences = re.split(r"(?<=[.؟!])\s+|\n+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _cue_score(self, clean_sentence: str) -> float:
        score = 1.0
        for _, (pattern, weight) in CUE_PHRASE_CATEGORIES.items():
            if re.search(pattern, clean_sentence):
                score *= weight
        return score

    def summarize(self, text, base_compression_ratio=0.5, max_sentences=10,
                  min_sentences_for_compression=7, use_mmr=True, mmr_lambda=0.7):
        sentences = self.split_into_sentences(text)
        total_sentences = len(sentences)

        if total_sentences <= min_sentences_for_compression:
            return " ".join(sentences)

        target_length = int(total_sentences * base_compression_ratio)
        target_length = min(target_length, max_sentences)
        target_length = max(target_length, 4)

        clean_sentences = [self.preprocessor.clean_text(s) for s in sentences]
        vectorizer = TfidfVectorizer(stop_words=self.preprocessor.all_stopwords)
        X = vectorizer.fit_transform(clean_sentences)

        sim_matrix = (X * X.T).toarray()
        np.fill_diagonal(sim_matrix, 0)
        nx_graph = nx.from_numpy_array(sim_matrix)
        
        try:
            scores = nx.pagerank(nx_graph)
        except nx.PowerIterationFailedConvergence:
            scores = {i: 1.0 / total_sentences for i in range(total_sentences)}

        for i, clean_sentence in enumerate(clean_sentences):
            scores[i] *= self._cue_score(clean_sentence)

        if use_mmr:
            selected = self._mmr_select(scores, sim_matrix, target_length, mmr_lambda)
        else:
            ranked = sorted(((scores[i], i) for i in range(total_sentences)), reverse=True)
            selected = [i for _, i in ranked[:target_length]]

        selected = sorted(selected)
        return " ".join(sentences[i] for i in selected)

    def _mmr_select(self, scores, sim_matrix, top_k, lambda_param):
        n = len(scores)
        top_k = min(top_k, n)
        selected, candidates = [], list(range(n))

        while len(selected) < top_k and candidates:
            if not selected:
                best = max(candidates, key=lambda i: scores[i])
            else:
                def mmr_score(i):
                    redundancy = max(sim_matrix[i][j] for j in selected)
                    return lambda_param * scores[i] - (1 - lambda_param) * redundancy
                best = max(candidates, key=mmr_score)
            selected.append(best)
            candidates.remove(best)
        return selected

class StructuredFieldExtractor:
    def extract(self, text: str) -> dict:
        fields = {
            "date": None, "accused": [], "victim": None, "action": None,
            "evidence": [], "confession_status": [], "ruling": None,
        }

        date_match = re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)
        if date_match: fields["date"] = date_match.group()

        party_pattern = r"(?:المتهم(?:\s+(?:الأول|الثاني|الثالث))?|المدعى عليه)"
        fields["accused"] = list(dict.fromkeys(re.findall(party_pattern, text)))

        if re.search(r"المجني عليه", text): fields["victim"] = "المجني عليه"
        elif re.search(r"المدعي(?!\s*عليه)", text): fields["victim"] = "المدعي"

        seen = set()
        for pattern in [
            rf"(اعترف|أنكر|انكر)[^\.]{{0,40}}?({party_pattern})",
            rf"({party_pattern})[^\.]{{0,40}}?(اعترف|أنكر|انكر)",
        ]:
            for m in re.finditer(pattern, text):
                g1, g2 = m.group(1), m.group(2)
                who, verb = (g2, g1) if g1 in ("اعترف", "أنكر", "انكر") else (g1, g2)
                if (who, verb) not in seen:
                    seen.add((who, verb))
                    fields["confession_status"].append(f"{who}: {verb}")

        fields["evidence"] = list(dict.fromkeys(
            re.findall(r"أداة\s+[^\s.،؛!؟]+|سلاح\s*[^\s.،؛!؟]*|بصمات|كاميرا|سند\s+[^\s.،؛!؟]+", text)
        ))

        ruling_match = re.search(
            r"(قرر القاضي[^\.]*\.)|(حكمت المحكمة[^\.]*\.)|(قررت المحكمة[^\.]*\.)|(ألزمت المحكمة[^\.]*\.)", text
        )
        if ruling_match: fields["ruling"] = ruling_match.group().strip()

        action_match = re.search(
            r"(سرق[^\.]*\.|ضرب[^\.]*\.|مشاجرة[^\.]*\.|اعتدى[^\.]*\.|احتيال[^\.]*\.)", text
        )
        if action_match: fields["action"] = action_match.group().strip()

        return fields

class IntelligentLegalPipeline:
    def __init__(self):
        self.preprocessor = UniversalPreprocessor()
        self.summarizer = LegalSummarizer(self.preprocessor)
        self.field_extractor = StructuredFieldExtractor()

    def analyze(self, raw_text: str) -> dict:
        if not raw_text or len(raw_text.strip()) < 10:
            return {"status": "error", "error": "النص المدخل قصير جداً ولا يمكن تحليله."}

        summary = self.summarizer.summarize(raw_text)
        structured_fields = self.field_extractor.extract(raw_text)

        return {
            "status": "success",
            "original_length": len(raw_text),
            "summary": summary,
            "structured_fields": structured_fields
        }


