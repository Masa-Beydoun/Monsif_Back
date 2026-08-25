"""
إصدار الحكم الأولي (Legal Judgment Prediction).

يدمج ثلاث طبقات:
  1. PLJP (Wu et al., EMNLP 2023): إعادة تنظيم الوقائع لتحسين الاسترجاع.
     نص الاسترجاع دمج بين الوقائع الخام والثالوث لا استبدال لها، كي لا تضيع
     مفردات قانونية حاسمة (مثل «سند») أثناء التلخيص.
  2. ADAPT (Ask-Discriminate-Predict): كشف المواد المتشابهة لفظياً الحاضرة معاً
     في السياق، وطرح أسئلة تمييزية صريحة قبل قرار الحكم.
  3. تحقق ذاتي بأسلوب LegalReasoner: كشف التناقض بين المادة المُختارة وأرقام
     المواد المذكورة داخل تسبيب السوابق المرتبطة بالتهمة نفسها.

تعتمد هذه الميزة على:
    services.laws_rag   → المواد القانونية المرفقة بالسياق
    services.cases_rag  → السوابق القضائية المرفقة بالسياق

ويُعطَّل المصنّف الإحصائي تلقائياً عند غياب ملفات joblib بدل رمي خطأ.
"""

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

import config
from services import cases_rag, laws_rag, llm_client


# تطبيع النص العربي (لمقارنة الاقتباسات)

_TASHKEEL_RE = re.compile(r'[ً-ٰٟۖ-ۭ]')


def normalize_arabic(text: str) -> str:
    """تطبيع لأغراض المقارنة الحرفية للاقتباسات فقط."""
    if not text:
        return ""
    text = _TASHKEEL_RE.sub('', text)
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ٱ', 'ا')
    text = text.replace('ى', 'ي')
    text = text.replace('ة', 'ه')
    return re.sub(r'\s+', ' ', text).strip()


# تطبيع خاص بالمصنّف؛ يطابق حرفياً التطبيع المستخدم وقت التدريب.

_CLASSIFIER_DIACRITICS_RE = re.compile(
    "ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ"
)


def normalize_arabic_for_classifier(text: str) -> str:
    """يجب أن تطابق حرفياً دالة التطبيع المستخدمة في تدريب judicial_classifier.joblib."""
    text = str(text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub(_CLASSIFIER_DIACRITICS_RE, "", text)
    return re.sub(r"\s+", " ", text).strip()


# الإعدادات

@dataclass
class JudgmentConfig:
    top_k_laws: int = 5
    top_k_cases: int = 3
    exclude_repealed_laws: bool = True
    min_law_score: float = 0.0
    # عتبات جودة المواد المسترجَعة (انظر laws_rag.apply_score_cutoffs).
    # 0.0 = مطابقة النوتبوك: خليّة ADAPT تستدعي search() بلا عتبة نسبية ولا قصّ
    # ذيل، فتصل المواد الخمس كاملة إلى السياق. رفعها يُنظّف السياق لكنه يقلّص
    # قائمة المواد (قيست: 5 مواد → مادة واحدة على وقائع الأمانة).
    law_min_score_ratio: float = 0.0
    law_score_drop_ratio: float = 0.0
    case_threshold: float = 0.45
    max_output_tokens: int = 3072
    temperature: float = 0.1
    fact_reorg_max_tokens: int = 512
    # النموذج اللغوي: فارغ = الافتراضي العام في config (hf + Llama 3.1).
    llm_backend: str = ""
    llm_model: str = ""
    # مهمل: يُقرأ فقط عندما تكون الواجهة groq ولم يُحدَّد llm_model.
    groq_model: str = ""
    max_quote_words: int = 12

    # المصنّف الإحصائي المتخصص
    use_domain_classifier: bool = True
    top_k_classifier_labels: int = 5
    classifier_min_score: float = -1.0
    precedent_pool_size: int = 15

    # إعادة تنظيم الوقائع قبل الاسترجاع
    use_fact_reorganization: bool = True
    show_reorganized_facts_to_llm: bool = True

    # تمييز المواد المتشابهة لفظياً
    use_statute_discrimination: bool = True
    confusable_overlap_threshold: float = 0.15
    # طول نافذة المقارنة بالكلمات. 40 = مطابقة النوتبوك حرفياً.
    # تنبيه مقيس: على بيانات هذا المشروع تقطع نافذة الـ40 المادةَ 656 قبل شطر
    # العقوبة، فتهبط مشابهتها مع 657 إلى 0.067 (تحت العتبة 0.15) ولا يتشكّل
    # الزوج. القيمة 0 (المتن كامل) ترفعها إلى 0.275.
    confusable_window_words: int = 40
    max_confusable_pairs: int = 3
    discrimination_max_tokens: int = 400

    # مفتاحان يفصلان سلوك النوتبوك عن التصحيحات المقيسة. القيم الافتراضية
    # هي سلوك النوتبوك حرفياً؛ لا يُفعَّل أي منهما إلا بطلب صريح.
    #
    # False = النوتبوك: كشف الأزواج يمسح المواد المسترجَعة مباشرةً فقط.
    #   ملاحظة مقيسة: المادة 656 لا تظهر في الاسترجاع المباشر إطلاقاً (657 تأتي
    #   بنتيجة 0.746 و656 خارج أعلى النتائج)، وتصل عبر استشهادات السوابق وحدها.
    #   لذلك يبقى الكشف بلا أزواج على هذه البيانات ما دام هذا المفتاح False.
    confusable_scan_cited_articles: bool = False
    #
    # False = النوتبوك: مطابقة السوابق بـ case_number.
    #   ملاحظة مقيسة: الرقم ليس فريداً (48 تصادماً، 47 منها قضايا مختلفة فعلاً)،
    #   فقد تُنسب التسمية المرشحة إلى قضية أخرى تحمل الرقم نفسه.
    precedent_key_by_uid: bool = False
    #
    # False = النوتبوك: حلّ استشهادات السوابق بتلميح فضفاض وبلا مرشِّح.
    #   ملاحظة مقيسة على 301 قضية: 254 من 645 استشهاداً رقمياً (39%) تُحَل إلى
    #   القانون الخطأ — أشهرها المادة 129 التي تذهب إلى قانون العقوبات العسكري
    #   لأن اسمه يتضمّن كلمة «العقوبات». ومسار other_laws (163 مدخلاً) يلتقط أول
    #   رقم في النص، وهو غالباً رقم المرسوم لا رقم المادة («مرسوم العفو العام
    #   رقم 13 لعام 2021» ← يُقرأ كالمادة 13).
    #   True = تضييق المطابقة، وإسقاط ما يبقى ملتبساً بدل تخمينه.
    strict_citation_law_match: bool = False

    @classmethod
    def from_request(cls, overrides: Optional[Dict] = None) -> "JudgmentConfig":
        """القيم الافتراضية من config.py، وتعلوها أي قيمة أرسلها الطلب."""
        p = dict(config.JUDGMENT_DEFAULTS)
        if overrides:
            valid = {f for f in cls.__dataclass_fields__}
            p.update({k: v for k, v in overrides.items() if k in valid and v is not None})
        return cls(**{k: v for k, v in p.items() if k in cls.__dataclass_fields__})


# المصنّف الإحصائي المتخصص: SVM هجين (كلمات وحروف)

CLASSIFIER_FILES = ("judicial_classifier.joblib", "word_vectorizer.joblib",
                    "char_vectorizer.joblib", "label_binarizer.joblib")


def classifier_available() -> bool:
    return all((config.CLASSIFIER_DIR / f).exists() for f in CLASSIFIER_FILES)


class DomainCrimeClassifier:
    def __init__(self, classifier_dir):
        import joblib

        d = str(classifier_dir).rstrip("/\\") + os.sep
        self.model = joblib.load(d + "judicial_classifier.joblib")
        self.word_vectorizer = joblib.load(d + "word_vectorizer.joblib")
        self.char_vectorizer = joblib.load(d + "char_vectorizer.joblib")
        self.mlb = joblib.load(d + "label_binarizer.joblib")

    def _vectorize(self, text: str):
        from scipy.sparse import hstack

        norm = normalize_arabic_for_classifier(text)
        return hstack([self.word_vectorizer.transform([norm]),
                       self.char_vectorizer.transform([norm])])

    def predict_candidates(self, query_text: str, top_k: int = 5,
                           min_score: float = -1.0) -> List[Dict]:
        features = self._vectorize(query_text)
        scores = self.model.decision_function(features)[0]
        binary_pred = self.model.predict(features)[0]

        order = np.argsort(scores)[::-1]
        candidates = []
        for idx in order[:max(top_k, 1)]:
            score = float(scores[idx])
            if score < min_score and binary_pred[idx] == 0:
                continue
            candidates.append({
                "label": str(self.mlb.classes_[idx]),
                "score": round(score, 4),
                "predicted": bool(binary_pred[idx] == 1),
            })

        if not candidates:
            best_idx = int(np.argmax(scores))
            candidates.append({
                "label": str(self.mlb.classes_[best_idx]),
                "score": round(float(scores[best_idx]), 4),
                "predicted": False,
            })
        return candidates


_classifier_instance: Optional[DomainCrimeClassifier] = None
_classifier_lock = threading.Lock()


def get_domain_classifier(cfg: JudgmentConfig) -> Optional[DomainCrimeClassifier]:
    """يعيد None بهدوء إذا كان المصنّف معطَّلاً أو ملفاته ناقصة، ويكمل باقي المسار."""
    global _classifier_instance
    if not cfg.use_domain_classifier:
        return None
    if not classifier_available():
        print(f"[judgment] ملفات المصنّف غير موجودة في {config.CLASSIFIER_DIR}؛ "
              f"سيُتخطّى المصنّف الإحصائي.", flush=True)
        return None
    if _classifier_instance is None:
        with _classifier_lock:
            if _classifier_instance is None:
                _classifier_instance = DomainCrimeClassifier(config.CLASSIFIER_DIR)
                print("[judgment] المصنّف الإحصائي جاهز.", flush=True)
    return _classifier_instance


# عميل النموذج اللغوي

# يُعاد تصديرهما كي يبقى `except judgment.MissingAPIKey` في الـ route صالحاً.
MissingAPIKey = llm_client.MissingAPIKey
LLMError = llm_client.LLMError


def resolve_llm(cfg: JudgmentConfig):
    """(الواجهة، النموذج) الفعليان لهذا الطلب.

    groq_model حقل مهمل، فلا يُعتدّ به إلا مع الواجهة groq تحديداً — وإلا لسرّب
    اسم نموذج Groq إلى واجهة HuggingFace وأعاد 404.
    """
    backend = (cfg.llm_backend or config.LLM_BACKEND or "hf").strip().lower()
    model = (cfg.llm_model or "").strip()
    if not model and backend == "groq":
        model = (cfg.groq_model or "").strip()
    return llm_client.resolve(backend, model)


# سجلّ الاستدعاءات الفعلية لكل طلب.
#
# ‏resolve_llm يعطي الواجهة **المخطَّطة**؛ أما ما خدم الطلب فعلاً فقد يكون
# الاحتياطية عند نفاد الحصة. بدون هذا السجلّ يعرض meta.llm الواجهة المخطَّطة
# دائماً، فتظهر للواجهة الأمامية معلومة خاطئة كلما جرى تحويل.
#
# ‏thread-local لأن Flask يعمل بـ threaded=True: طلبان متزامنان لا يخلطان سجليهما.
_call_log = threading.local()


def reset_llm_call_log() -> None:
    _call_log.entries = []


def get_llm_call_log() -> List[Dict]:
    return list(getattr(_call_log, "entries", None) or [])


def _record_llm_call(stage: str, used: Dict) -> None:
    entries = getattr(_call_log, "entries", None)
    if entries is None:
        return
    entry = {"stage": stage}
    entry.update(used)
    entries.append(entry)


def _chat_json(messages: List[Dict], cfg: JudgmentConfig, max_tokens: int,
               temperature: float = 0.0, stage: str = "llm") -> Optional[dict]:
    """استدعاء النموذج اللغوي وإرجاع JSON مُحلَّل، أو None عند فشل الاستدعاء.

    خطأ المفتاح وحده يُرمى للأعلى؛ فهو خطأ إعداد يستحق 503 لا محاولة إكمال
    المسار بسياق ناقص.
    """
    backend, model = resolve_llm(cfg)
    used: Dict = {}
    try:
        return llm_client.chat_json(
            messages, backend=backend, model=model,
            max_tokens=max_tokens, temperature=temperature, used=used,
        )
    except MissingAPIKey:
        raise
    except llm_client.LLMError as e:
        print(f"[judgment] {e}", flush=True)
        return None
    finally:
        if used:
            _record_llm_call(stage, used)


# إعادة تنظيم الوقائع قبل الاسترجاع

FACT_REORG_SYSTEM = """
أنت مساعد قانوني. مهمتك إعادة صياغة وقائع قضية جزائية إلى ثلاثة عناصر بنيوية موحّدة
الشكل، تُستخدم لاحقاً للاسترجاع الدلالي والتصنيف الإحصائي فقط — لا تُستخدم كأساس
للحكم النهائي:

1. "subjective_motivation" (الدافع الذاتي): نية/قصد الفاعل الظاهرة من الوقائع، إن وُجدت.
2. "objective_behavior" (السلوك الموضوعي): الفعل المادي الذي قام به الفاعل، بأسلوب
   محايد ومختصر (من فعل ماذا، بأي وسيلة، تجاه من).
3. "ex_post_facto_circumstances" (الظروف اللاحقة): ما جرى بعد الفعل مباشرة.

قواعد صارمة:
- ممنوع إضافة أي معلومة أو تفصيل غير وارد صراحة بالوقائع الأصلية المرفقة.
- إذا كان أحد العناصر غير مذكور إطلاقاً بالوقائع، اتركيه نصاً فارغاً "" ولا تخترعيه.
- أعد الإجابة بصيغة JSON فقط بالحقول الثلاثة أعلاه بالضبط، بدون Markdown fences.
"""


def reorganize_facts(facts_input: str, cfg: JudgmentConfig) -> Dict[str, str]:
    if not cfg.use_fact_reorganization:
        return {}
    parsed = _chat_json(
        [{"role": "system", "content": FACT_REORG_SYSTEM},
         {"role": "user", "content": f"وقائع القضية:\n{facts_input}"}],
        cfg, max_tokens=cfg.fact_reorg_max_tokens, temperature=0.0,
        stage="fact_reorganization",
    )
    if not isinstance(parsed, dict):
        print("[judgment] تعذّرت إعادة تنظيم الوقائع؛ سيُستخدم النص الخام للاسترجاع.",
              flush=True)
        return {}
    return {
        "subjective_motivation": str(parsed.get("subjective_motivation", "") or "").strip(),
        "objective_behavior": str(parsed.get("objective_behavior", "") or "").strip(),
        "ex_post_facto_circumstances": str(parsed.get("ex_post_facto_circumstances", "") or "").strip(),
    }


def build_retrieval_text(facts_input: str, reorganized: Dict[str, str]) -> str:
    """دمج الوقائع الخام الكاملة مع ثالوث إعادة التنظيم، لا استبدالها.

    يضمن ذلك بقاء المفردات القانونية الحاسمة (مثل «سند») حاضرة في نص الاستعلام.
    """
    if not reorganized:
        return facts_input
    triplet = " ".join(p for p in (
        reorganized.get("subjective_motivation", ""),
        reorganized.get("objective_behavior", ""),
        reorganized.get("ex_post_facto_circumstances", ""),
    ) if p.strip())
    return f"{facts_input.strip()}\n\n{triplet}" if triplet.strip() else facts_input


# تمييز المواد المتشابهة لفظياً (ADAPT)

def _statute_view(item) -> Dict:
    """توحيد شكل المادة: كائن SearchResult من الاسترجاع، أو dict من السوابق."""
    if isinstance(item, dict):
        return {"article_id": item.get("article_id"),
                "law_name": item.get("law_name") or "",
                "body": item.get("body") or ""}
    return {"article_id": item.article_id,
            "law_name": item.law_name or "",
            "body": item.body or ""}


def _body_words(body: str, window: int) -> List[str]:
    """كلمات المتن، مقصوصة على نافذة إن طُلبت (0 = المتن كامل)."""
    words = body.split()
    return words[:window] if window else words


def detect_confusable_statutes(statutes: List, cfg: JudgmentConfig) -> List[Dict]:
    """كشف تقريبي لأزواج المواد من القانون نفسه المتقاربة لفظياً.

    يعتمد تقاطع مفردات المتن (معامل جاكار)، ويكفي لالتقاط حالات مثل 656/657 دون
    نموذج تشابه دلالي كامل.

    ما يُمرَّر إليها يحدّده cfg.confusable_scan_cited_articles في gather_context:
    قائمة الاسترجاع وحدها (سلوك النوتبوك) أو اتحادها مع مواد السوابق.
    """
    window = max(0, int(cfg.confusable_window_words or 0))

    unique, seen = [], set()
    for item in statutes:
        view = _statute_view(item)
        if view["article_id"] and view["article_id"] not in seen:
            seen.add(view["article_id"])
            unique.append(view)

    pairs = []
    for i, a in enumerate(unique):
        for b in unique[i + 1:]:
            if a["law_name"] != b["law_name"]:
                continue
            words_a = set(_body_words(a["body"], window))
            words_b = set(_body_words(b["body"], window))
            overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
            if overlap >= cfg.confusable_overlap_threshold:
                pairs.append({"article_a": a["article_id"], "article_b": b["article_id"],
                              "overlap": round(overlap, 2)})

    # لا ترتيب: النوتبوك يقتطع الأزواج بترتيب ورودها.
    return pairs[:cfg.max_confusable_pairs]


DISCRIMINATION_SYSTEM = """
أنت مساعد قانوني. أمامك مادتان قانونيتان متشابهتان لفظياً قد تُطبَّقان على نفس نوع
الوقائع، ووقائع قضية محددة. مهمتك فقط طرح أسئلة تمييزية والإجابة عليها اعتماداً
حصراً على الوقائع المرفقة — لا تقرري التهمة النهائية هون، هاي خطوة تحضيرية فقط.

أعد JSON فقط:
{
  "distinguishing_questions": [
    {"question": "...", "answer_from_facts": "...", "points_to_article": "article_id أو 'غير محسوم'"}
  ]
}
اطرحي 2-3 أسئلة تمس الفرق الجوهري بين نص المادتين حرفياً. ممنوع اختلاق تفصيل غير
وارد بالوقائع — إذا الوقائع غير كافية للحسم، اكتبي "غير محسوم".
"""


def discriminate_confusable_statutes(facts_input: str, pair: Dict, bodies: Dict[str, str],
                                     cfg: JudgmentConfig) -> Optional[Dict]:
    user_msg = (
        f"وقائع القضية:\n{facts_input}\n\n"
        f"المادة الأولى [{pair['article_a']}]:\n{bodies.get(pair['article_a'], '')}\n\n"
        f"المادة الثانية [{pair['article_b']}]:\n{bodies.get(pair['article_b'], '')}"
    )
    parsed = _chat_json(
        [{"role": "system", "content": DISCRIMINATION_SYSTEM},
         {"role": "user", "content": user_msg}],
        cfg, max_tokens=cfg.discrimination_max_tokens, temperature=0.0,
        stage="statute_discrimination",
    )
    if not isinstance(parsed, dict):
        return None
    parsed["article_a"] = pair["article_a"]
    parsed["article_b"] = pair["article_b"]
    return parsed


# جمع السياق

PENAL_CODE_HINTS = ["العقوبات", "قانون العقوبات"]

# التلميح أعلاه فحصُ تضمين نصي، و«قانون العقوبات واصول المحاكمات العسكرية» يتضمّن
# كلمة «العقوبات»، فيلتقطه أيضاً. هذا التلميح يحصر المطابقة بالقانون العام.
PENAL_CODE_STRICT_HINTS = ["قانون العقوبات العام"]

# «مادة 24 (القانون رقم 20 لعام 2022)» ← 24. أما «مرسوم العفو رقم 13 لعام 2021»
# فلا يذكر مادة إطلاقاً، ورقمه رقم المرسوم؛ في الوضع الصارم يُسقَط بدل تخمينه.
_ARTICLE_IN_ENTRY_RE = re.compile(r"(?:المادة|مادة)\s*/?(\d+)")


def _find_article_by_number(article_number: str,
                            law_name_contains: Optional[List[str]] = None,
                            require_unique: bool = False):
    """بحث عن مادة برقم معيّن في المجموعة الكاملة، لا في النتائج المسترجعة.

    require_unique=True يعيد None عند بقاء أكثر من مرشّح بعد الترشيح، بدل انتقاء
    الأول اعتباطاً: رقم المادة وحده لا يميّز بين 24 قانوناً تحمل «المادة 1».
    """
    rag = laws_rag.get_rag()
    article_number = str(article_number).strip()
    candidates = [a for a in rag.articles if a.article_number == article_number]
    if law_name_contains:
        filtered = [a for a in candidates
                    if any(h in (a.law_name or "") for h in law_name_contains)]
        if filtered:
            candidates = filtered
        elif require_unique:
            return None
    if require_unique:
        return candidates[0] if len(candidates) == 1 else None
    return candidates[0] if candidates else None


def resolve_case_citations(case: Dict, cfg: Optional[JudgmentConfig] = None) -> List[Dict]:
    """استخراج نصوص المواد المذكورة صراحةً داخل سابقة، من مجموعة المواد.

    السلوك الافتراضي هو سلوك النوتبوك؛ cfg.strict_citation_law_match يضيّقه.
    """
    strict = bool(cfg and cfg.strict_citation_law_match)
    penal_hints = PENAL_CODE_STRICT_HINTS if strict else PENAL_CODE_HINTS

    resolved = []
    for num in case.get("penal_code_articles") or []:
        art = _find_article_by_number(num, law_name_contains=penal_hints,
                                      require_unique=strict)
        if art:
            resolved.append({
                "article_id": art.article_id, "law_name": art.law_name,
                "article_number": art.article_number, "body": art.body_raw,
                "source": f"مذكورة بالقضية رقم {case.get('case_number')}",
            })
    for entry in case.get("other_laws") or []:
        text = str(entry)
        if strict:
            # لا يُقبل إلا ما ذُكرت فيه «مادة N» صراحةً، ويُرشَّح باسم القانون
            # الوارد في النص نفسه؛ ما يبقى ملتبساً يُسقَط.
            m = _ARTICLE_IN_ENTRY_RE.search(text)
            if not m:
                continue
            name_hint = text[m.end():].strip(" ()").strip()
            art = _find_article_by_number(
                m.group(1), law_name_contains=[name_hint] if name_hint else None,
                require_unique=True)
        else:
            m = re.search(r"(\d+)", text)
            if not m:
                continue
            art = _find_article_by_number(m.group(1))
        if art and art.article_id not in {r["article_id"] for r in resolved}:
            resolved.append({
                "article_id": art.article_id, "law_name": art.law_name,
                "article_number": art.article_number, "body": art.body_raw,
                "source": f"مذكورة بالقضية رقم {case.get('case_number')} ({entry})",
            })
    return resolved


def retrieve_precedents_per_candidate(retrieval_text: str, candidate_labels: List[str],
                                      cfg: JudgmentConfig) -> Dict[str, Dict]:
    """أفضل سابقة في المجموعة تحمل التهمة نفسها، لكل تسمية مرشّحة من المصنّف."""
    if not candidate_labels:
        return {}
    pool = cases_rag.get_rag().query_raw(
        retrieval_text, top_k=cfg.precedent_pool_size,
        hybrid_top_k=cfg.precedent_pool_size, threshold=0.0,
    )
    matched: Dict[str, Dict] = {}
    for label in candidate_labels:
        best_case, best_score = None, -1.0
        for case in pool:
            if label in (case.get("crimes") or []):
                score = case.get("similarity_score", 0) or 0
                if score > best_score:
                    best_case, best_score = case, score
        if best_case:
            matched[label] = best_case
    return matched


def gather_context(facts_input: str, cfg: JudgmentConfig) -> Dict:
    reorganized_facts = reorganize_facts(facts_input, cfg)
    retrieval_text = build_retrieval_text(facts_input, reorganized_facts)

    retrieved_laws = laws_rag.get_rag().search_raw(
        retrieval_text, top_n=cfg.top_k_laws,
        exclude_repealed=cfg.exclude_repealed_laws, min_score=cfg.min_law_score,
        min_score_ratio=cfg.law_min_score_ratio,
        score_drop_ratio=cfg.law_score_drop_ratio,
    )
    retrieved_cases = cases_rag.get_rag().query_raw(
        retrieval_text, top_k=cfg.top_k_cases, threshold=cfg.case_threshold,
    )

    classifier = get_domain_classifier(cfg)
    candidate_charges_from_classifier: List[Dict] = []
    if classifier is not None:
        candidate_charges_from_classifier = classifier.predict_candidates(
            retrieval_text, top_k=cfg.top_k_classifier_labels,
            min_score=cfg.classifier_min_score,
        )
        candidate_labels = [c["label"] for c in candidate_charges_from_classifier]
        matched_precedents = retrieve_precedents_per_candidate(
            retrieval_text, candidate_labels, cfg)

        # النوتبوك يطابق بـ case_number؛ المفتاح الفريد لا يُستعمل إلا بطلب صريح.
        def _uid(case: Dict) -> str:
            if cfg.precedent_key_by_uid:
                return str(case.get("case_uid") or case.get("file_name")
                           or case.get("case_number"))
            return str(case.get("case_number"))

        existing_ids = {_uid(c) for c in retrieved_cases}
        for label, matched_case in matched_precedents.items():
            cid = _uid(matched_case)
            if cid not in existing_ids:
                new_case = dict(matched_case)
                new_case["_matched_candidate_label"] = label
                retrieved_cases.append(new_case)
                existing_ids.add(cid)
            else:
                for c in retrieved_cases:
                    if _uid(c) == cid:
                        c["_matched_candidate_label"] = label

    cited_in_cases, seen_ids = [], {r.article_id for r in retrieved_laws}
    for case in retrieved_cases:
        for art in resolve_case_citations(case, cfg):
            if art["article_id"] not in seen_ids:
                seen_ids.add(art["article_id"])
                cited_in_cases.append(art)

    discrimination_results: List[Dict] = []
    confusable_pairs: List[Dict] = []
    if cfg.use_statute_discrimination:
        # متون المقارنة تشمل الاثنين دائماً، تماماً كالنوتبوك.
        all_bodies = {r.article_id: r.body for r in retrieved_laws}
        all_bodies.update({a["article_id"]: a["body"] for a in cited_in_cases})
        # أما مدخل الكشف فقائمة الاسترجاع وحدها، إلا إذا طُلب خلاف ذلك صراحةً.
        scan = list(retrieved_laws)
        if cfg.confusable_scan_cited_articles:
            scan += list(cited_in_cases)
        confusable_pairs = detect_confusable_statutes(scan, cfg)
        for pair in confusable_pairs:
            d = discriminate_confusable_statutes(facts_input, pair, all_bodies, cfg)
            if d:
                discrimination_results.append(d)

    return {
        "facts": facts_input,
        "retrieval_text": retrieval_text,
        "reorganized_facts": reorganized_facts,
        "retrieved_laws": retrieved_laws,
        "retrieved_cases": retrieved_cases,
        "cited_in_cases": cited_in_cases,
        "candidate_charges_from_classifier": candidate_charges_from_classifier,
        "confusable_pairs": confusable_pairs,
        "discrimination_results": discrimination_results,
    }


# بناء الـ Prompt

class LegalPromptBuilder:

    SYSTEM_INSTRUCTIONS = """
أنت مساعد قانوني متخصص بالقانون الجزائي السوري. مهمتك اقتراح حكم أولي (preliminary judgment)
اعتماداً حصراً على الوقائع، النصوص القانونية المرفقة أدناه، والسوابق القضائية المرفقة.

قواعد صارمة:
1. ممنوع الاستشهاد بأي مادة قانونية أو رقم قضية غير موجود صراحة ضمن "النصوص القانونية المرفقة"
   أو "السوابق المرفقة" أدناه. لا تستخدمي أي معرفة قانونية من ذاكرتك خارج هذا السياق.
2. اتبعي منهجية استدلال متسلسلة (IRAC/ADAPT) قبل الوصول لأي استنتاج:
   أ) وقائع: لخّصي الوقائع الجوهرية فقط، اعتماداً على نص "وقائع القضية الجديدة" الكامل
      أدناه — لا على أي تلخيص أو إعادة تنظيم مُرفَقة بالسياق (إن وُجدت).
   ب) التهم المحتملة: عدّدي كل التهم المرشحة (لا تكتفي بأول تهمة تخطر ببالك) وميّزي بينها.
   ج) تطبيق النص: لكل تهمة مرشحة اذكري المادة المطابقة من النصوص المرفقة وأركانها.
   د) الاستبعاد: وضّحي ليش استبعدتِ التهم غير المطابقة.
   هـ) الاستنتاج: التهمة/التهم الأقرب للوقائع.
3. النتيجة (outcome) يجب أن تكون واحدة من: "إدانة" | "براءة" | "إسقاط دعوى".
   لا تفترضي الإدانة كخيار افتراضي.
4. إذا كانت النصوص المرفقة غير كافية لإصدار حكم واثق، صرّحي بذلك ضمن "confidence_note".
5. **لكل تهمة مرشحة، يجب إرفاق "supporting_quote": اقتباس حرفي (كلمة بكلمة) لا يتجاوز
   {max_quote_words} كلمة من نص المادة المذكورة بالضبط، يثبت الركن الذي بنيتِ عليه
   المطابقة. إذا ما قدرتِ تجدي اقتباساً حرفياً يثبت الركن، هذا مؤشر إنو التهمة لا
   تنطبق فعلياً — لا تختلقي اقتباساً تقريبياً أو بالمعنى.**
6. أعد الإجابة بصيغة JSON فقط، بدون أي نص قبلها أو بعدها، وبدون Markdown fences.
7. إذا أُرفقت أدناه "تهم مرشحة من نموذج تصنيف إحصائي متخصص"، فاعتبريها إشارة إحصائية
   أولية مبنية على تشابه نصي سطحي فقط — لا حقيقة ملزمة.
8. إذا أُرفق أدناه "تنظيم أولي للوقائع"، فهو نتاج خطوة آلية منفصلة لتحسين الاسترجاع فقط
   — اعتمدي حصراً على نص "وقائع القضية الجديدة" الكامل كمصدر ملزم للحقائق.
9. **إذا أُرفق أدناه "تمييز مسبق بين مواد متشابهة لفظياً"، فهذا تحليل تحضيري يجيب على
   أسئلة تمييزية اعتماداً على الوقائع فقط. راجعيه بعناية قبل اختيار المادة النهائية —
   إذا كانت إحدى المادتين المتشابهتين مرفقة معك ضمن السياق ووجدتِ الوقائع تحقق ركنها
   الخاص (مثلاً وجود سند/مستند مكتوب لا مجرد مبلغ مالي عام)، يجب ترجيحها ولو كانت
   الأخرى أقرب لفظياً لعبارات الوقائع. **تحققي أيضاً من انسجام اختيارك مع رقم المادة
   المذكور صراحة داخل نص تسبيب السوابق المرفقة المرتبطة بنفس التهمة — أي تعارض بينهما
   مؤشر قوي إنك اخترتِ المادة الخطأ.**
"""

    OUTPUT_SCHEMA_HINT = """
أعد كائن JSON بالحقول التالية بالضبط:
{
  "facts_summary": "...",
  "candidate_charges": [
    {
      "charge": "اسم التهمة",
      "article_id": "article_id المطابق كما ورد بالنصوص المرفقة",
      "supporting_quote": "اقتباس حرفي قصير (≤ 12 كلمة) من نص المادة نفسها يثبت الركن",
      "elements_match": true
    }
  ],
  "reasoning": {
    "charges_analysis": "تحليل كل تهمة مرشحة وأركانها ومطابقتها للنص",
    "exclusion_notes": "ليش استُبعدت باقي التهم",
    "precedent_alignment": "كيف تتوافق/تختلف السوابق المرفقة مع هذه الوقائع، بما فيه مقارنة رقم المادة المُختارة برقم المادة المذكور داخل تسبيب السوابق"
  },
  "outcome": "إدانة | براءة | إسقاط دعوى",
  "final_charge": "التهمة النهائية أو null إذا براءة/إسقاط",
  "cited_statutes": ["article_id1", "article_id2"],
  "cited_precedents": ["case_number1", "case_number2"],
  "suggested_penalty_range": "كما وردت حرفياً بنص المادة المستشهد بها، أو null",
  "confidence_note": "أي تحفظ على كفاية السياق"
}
"""

    def _format_laws_block(self, laws: List, cited_in_cases: List[Dict]) -> str:
        lines = ["### النصوص القانونية المرفقة (من البحث المباشر عن الوقائع):"]
        for r in laws:
            badge = " (ملغاة)" if r.status == "ملغاة" else ""
            lines.append(f"- [article_id={r.article_id}] {r.law_name} — المادة {r.article_number}{badge}\n"
                         f"  النص: {r.body[:600]}")
        if cited_in_cases:
            lines.append("\n### نصوص قانونية إضافية (مذكورة داخل السوابق أدناه):")
            for a in cited_in_cases:
                lines.append(f"- [article_id={a['article_id']}] {a['law_name']} — المادة "
                             f"{a['article_number']} ({a['source']})\n  النص: {a['body'][:600]}")
        return "\n".join(lines)

    def _format_classifier_block(self, candidates: List[Dict]) -> str:
        if not candidates:
            return ""
        lines = ["### تهم مرشحة من النموذج التصنيفي المتخصص (إشارة إحصائية أولية غير ملزمة):"]
        for c in candidates:
            flag = "تجاوزت عتبة القرار" if c["predicted"] else "دون العتبة (أعلى تسجيل فقط)"
            lines.append(f"- {c['label']} — score={c['score']} ({flag})")
        return "\n".join(lines)

    def _format_reorganized_facts_block(self, reorganized: Dict[str, str]) -> str:
        if not reorganized:
            return ""
        return (
            "### تنظيم أولي للوقائع (مساعدة تنظيمية فقط — لأغراض الاسترجاع، غير ملزم):\n"
            f"- الدافع الذاتي: {reorganized.get('subjective_motivation') or '—'}\n"
            f"- السلوك الموضوعي: {reorganized.get('objective_behavior') or '—'}\n"
            f"- الظروف اللاحقة: {reorganized.get('ex_post_facto_circumstances') or '—'}"
        )

    def _format_discrimination_block(self, discrimination_results: List[Dict]) -> str:
        if not discrimination_results:
            return ""
        lines = ["### تمييز مسبق بين مواد متشابهة لفظياً (خطوة تحضيرية ADAPT-style، "
                 "إجابات مبنية على الوقائع فقط — راجعيها بعناية، قرارك النهائي مستقل):"]
        for d in discrimination_results:
            lines.append(f"\nبين [{d.get('article_a')}] و[{d.get('article_b')}]:")
            for q in d.get("distinguishing_questions", []):
                lines.append(f"- س: {q.get('question')}\n"
                             f"  ج (من الوقائع): {q.get('answer_from_facts')}\n"
                             f"  يرجّح: {q.get('points_to_article')}")
        return "\n".join(lines)

    def _format_cases_block(self, cases: List[Dict]) -> str:
        lines = ["### السوابق القضائية المشابهة:"]
        for c in cases:
            tag = (f" | مرتبطة بالتسمية المرشحة: {c['_matched_candidate_label']}"
                   if c.get("_matched_candidate_label") else "")
            lines.append(
                f"- [case_number={c.get('case_number')}] تشابه: {c.get('similarity_score')}% "
                f"| النتيجة: {c.get('outcome')} | التهم: {', '.join(c.get('crimes', []))}{tag}\n"
                f"  الوقائع: {str(c.get('facts_text', ''))[:400]}\n"
                f"  التسبيب: {str(c.get('reasoning_text', ''))[:400]}\n"
                f"  الحكم: {str(c.get('judgment_text', ''))[:300]}"
            )
        return "\n".join(lines)

    def build(self, context: Dict, cfg: JudgmentConfig) -> str:
        parts = [self.SYSTEM_INSTRUCTIONS.format(max_quote_words=cfg.max_quote_words),
                 f"\n### وقائع القضية الجديدة:\n{context['facts']}\n"]
        if cfg.show_reorganized_facts_to_llm:
            parts.append("\n" + self._format_reorganized_facts_block(
                context.get("reorganized_facts", {})))
        parts.append("\n" + self._format_discrimination_block(
            context.get("discrimination_results", [])))
        parts += [
            self._format_laws_block(context["retrieved_laws"], context["cited_in_cases"]),
            "\n" + self._format_classifier_block(
                context.get("candidate_charges_from_classifier", [])),
            "\n" + self._format_cases_block(context["retrieved_cases"]),
            "\n" + self.OUTPUT_SCHEMA_HINT,
        ]
        return "\n".join(parts)


# التحقق: الإسناد وتناقض السوابق

def _numbers(text: str) -> set:
    return set(re.findall(r"\d[\d,]*", text))


def _extract_article_numbers_from_text(text: str) -> set:
    """أرقام المواد المذكورة صراحة في نص حر (مثل «وفق أحكام المادة 656»)."""
    return set(re.findall(r"(?:المادة|مادة)\s*/?(\d+)/?", text or ""))


def _bodies_by_id(context: Dict) -> Dict[str, str]:
    bodies = {r.article_id: r.body for r in context["retrieved_laws"]}
    bodies.update({a["article_id"]: a["body"] for a in context["cited_in_cases"]})
    return bodies


def verify_grounding(result: dict, context: Dict) -> Dict:
    retrieved_statute_ids = {r.article_id for r in context["retrieved_laws"]}
    retrieved_statute_ids |= {a["article_id"] for a in context["cited_in_cases"]}
    retrieved_case_ids = {str(c.get("case_number")) for c in context["retrieved_cases"]}

    cited_statutes = {str(x) for x in result.get("cited_statutes", [])}

    def _extract_case_number(x) -> str:
        m = re.search(r"\d+", str(x))
        return m.group(0) if m else str(x).strip()

    cited_cases = {_extract_case_number(x) for x in result.get("cited_precedents", [])}

    unknown_statutes = cited_statutes - retrieved_statute_ids
    unknown_cases = cited_cases - retrieved_case_ids

    penalty_text = str(result.get("suggested_penalty_range") or "")
    content_mismatches = []
    if penalty_text:
        bodies = _bodies_by_id(context)
        penalty_numbers = _numbers(penalty_text)
        for aid in cited_statutes & retrieved_statute_ids:
            if penalty_numbers - _numbers(bodies.get(aid, "")):
                content_mismatches.append(aid)

    ok = not unknown_statutes and not unknown_cases and not content_mismatches
    return {
        "grounded": ok,
        "unknown_statute_citations": sorted(unknown_statutes),
        "unknown_case_citations": sorted(unknown_cases),
        "possible_penalty_hallucination_in": content_mismatches,
    }


def verify_charge_quotes(result: dict, context: Dict, cfg: JudgmentConfig) -> List[Dict]:
    """يتحقق من وجود كل supporting_quote فعلياً في متن المادة المذكورة معه.

    المقارنة تجري بعد التطبيع العربي، لا بمجرد التأكد من صحة شكل article_id.
    """
    bodies = _bodies_by_id(context)
    flagged = []
    for charge in result.get("candidate_charges", []):
        if not isinstance(charge, dict):
            continue
        charge_name = charge.get("charge", "؟")
        aid = charge.get("article_id")
        quote_raw = str(charge.get("supporting_quote") or "").strip()

        if not quote_raw:
            flagged.append({"charge": charge_name, "article_id": aid,
                            "reason": "لا يوجد اقتباس داعم (supporting_quote فارغ)"})
            continue
        if aid not in bodies:
            flagged.append({"charge": charge_name, "article_id": aid,
                            "reason": "article_id غير موجود أصلاً بالسياق المسترجع"})
            continue

        word_count = len(quote_raw.split())
        if word_count > cfg.max_quote_words * 2:
            flagged.append({"charge": charge_name, "article_id": aid,
                            "reason": f"الاقتباس طويل جداً ({word_count} كلمة) — قد يكون "
                                      f"تحايلاً على الفحص بدل اقتباس ركن محدد"})
            continue

        if normalize_arabic(quote_raw) not in normalize_arabic(bodies[aid]):
            flagged.append({"charge": charge_name, "article_id": aid,
                            "reason": "الاقتباس غير موجود حرفياً بنص المادة — احتمال هلوسة "
                                      "بمضمون الركن القانوني",
                            "quoted": quote_raw})
    return flagged


def verify_precedent_consistency(result: dict, context: Dict) -> Dict:
    """كشف التناقض بين المادة المُختارة ورقم المادة المذكور صراحة في تسبيب
    السوابق المرتبطة بالتهمة المرشحة نفسها."""
    mismatches = []
    id_to_number = {r.article_id: r.article_number for r in context["retrieved_laws"]}
    id_to_number.update({a["article_id"]: a["article_number"] for a in context["cited_in_cases"]})

    chosen_article_numbers = set()
    for charge in result.get("candidate_charges", []):
        if isinstance(charge, dict):
            num = id_to_number.get(charge.get("article_id"))
            if num:
                chosen_article_numbers.add(num)

    for case in context["retrieved_cases"]:
        precedent_numbers = _extract_article_numbers_from_text(str(case.get("reasoning_text", "")))
        if not precedent_numbers:
            continue
        if (case.get("_matched_candidate_label") and chosen_article_numbers
                and not (chosen_article_numbers & precedent_numbers)):
            mismatches.append({
                "case_number": case.get("case_number"),
                "precedent_articles": sorted(precedent_numbers),
                "chosen_articles": sorted(chosen_article_numbers),
            })
    return {"precedent_article_mismatches": mismatches}


def verify_all(result: dict, context: Dict, cfg: JudgmentConfig) -> Dict:
    id_check = verify_grounding(result, context)
    quote_flags = verify_charge_quotes(result, context, cfg)
    consistency_check = verify_precedent_consistency(result, context)

    id_check["unsupported_charge_quotes"] = quote_flags
    id_check.update(consistency_check)
    id_check["fully_grounded"] = (
        id_check["grounded"] and not quote_flags
        and not consistency_check["precedent_article_mismatches"]
    )
    return id_check


# المسار الكامل

class LegalJudgmentPredictor:
    """يجمع السياق، ويبني الـ prompt، ويستدعي النموذج اللغوي، ثم يتحقق من الناتج."""

    def __init__(self, cfg: Optional[JudgmentConfig] = None):
        self.config = cfg or JudgmentConfig()
        self.prompt_builder = LegalPromptBuilder()

    def predict(self, facts_input: str) -> Dict:
        cfg = self.config
        reset_llm_call_log()
        context = gather_context(facts_input, cfg)
        prompt = self.prompt_builder.build(context, cfg)
        raw_result = _chat_json([{"role": "user", "content": prompt}], cfg,
                                max_tokens=cfg.max_output_tokens,
                                temperature=cfg.temperature, stage="judgment")

        if not isinstance(raw_result, dict):
            return {
                "error": "parsing_error",
                "message": "لم يُعِد النموذج اللغوي مخرجاً صالحاً. يرجى إعادة المحاولة.",
                "llm_calls": get_llm_call_log(),
                "retrieved_statutes": [r.to_dict() for r in context["retrieved_laws"]],
                "retrieved_cases": context["retrieved_cases"],
            }

        raw_result["_verification"] = verify_all(raw_result, context, cfg)
        raw_result["_retrieved_statutes"] = [
            {"article_id": r.article_id, "law_name": r.law_name,
             "article_number": r.article_number, "status": r.status,
             "score": round(float(r.score), 4), "body": r.body}
            for r in context["retrieved_laws"]
        ] + context["cited_in_cases"]
        raw_result["_retrieved_cases"] = context["retrieved_cases"]
        raw_result["_classifier_candidates"] = context.get("candidate_charges_from_classifier", [])
        raw_result["_reorganized_facts"] = context.get("reorganized_facts", {})
        raw_result["_discrimination_results"] = context.get("discrimination_results", [])
        raw_result["_confusable_pairs"] = context.get("confusable_pairs", [])
        raw_result["_llm_calls"] = get_llm_call_log()
        raw_result["_retrieval_text"] = context.get("retrieval_text", "")
        return raw_result


# تشكيل الاستجابة للواجهة الأمامية

OUTCOMES = ["إدانة", "براءة", "إسقاط دعوى"]

VERIFICATION_LABELS = {
    True: "مطابق للسياق (هوية + مضمون الأركان + انسجام السوابق)",
    False: "استشهادات أو أركان غير موثّقة — تحتاج مراجعة يدوية",
}


def _enrich_charges(raw_charges, statutes: List[Dict], quote_flags: List[Dict]) -> List[Dict]:
    """يضيف لكل تهمة بيانات المادة وسبب الوسم، كي لا تربط الواجهة الجداول يدوياً."""
    by_id = {s["article_id"]: s for s in statutes}
    flag_by_key = {(f.get("charge"), f.get("article_id")): f for f in quote_flags}

    charges = []
    for ch in raw_charges or []:
        if not isinstance(ch, dict):
            charges.append({"charge": str(ch), "article_id": None, "flagged": True,
                            "flag_reason": "صيغة غير متوقعة من النموذج اللغوي"})
            continue
        aid = ch.get("article_id")
        statute = by_id.get(aid, {})
        flag = flag_by_key.get((ch.get("charge"), aid))
        charges.append({
            "charge": ch.get("charge"),
            "article_id": aid,
            "article_number": statute.get("article_number"),
            "law_name": statute.get("law_name"),
            "article_status": statute.get("status"),
            "supporting_quote": ch.get("supporting_quote"),
            "elements_match": ch.get("elements_match"),
            # الواجهة تلوّن بهذين الحقلين مباشرة دون قراءة كتلة التحقق.
            "flagged": flag is not None,
            "flag_reason": flag.get("reason") if flag else None,
        })
    return charges


def build_api_payload(result: Dict, cfg: JudgmentConfig, took_ms: int) -> Dict:
    """يحوّل مخرَج المتنبّئ إلى بنية ثابتة موثّقة تستهلكها الواجهة الأمامية.

    مخرَج النموذج اللغوي الخام يبقى تحت raw_model_output، فأي تغيير في الـ prompt
    لا يكسر عقد الواجهة.
    """
    backend, model = resolve_llm(cfg)
    calls = result.get("_llm_calls") or result.get("llm_calls") or []
    served = [{"stage": c.get("stage"), "backend": c.get("backend"),
               "model": c.get("model")} for c in calls]
    meta = {
        "took_ms": took_ms,
        "llm": {
            # المخطَّط مقابل ما خدم فعلاً: يفترقان عند التحويل الاحتياطي.
            "backend": backend,
            "model": model,
            "fallback_used": any(c.get("fallback_used") for c in calls),
            "calls": len(calls),
            "served_by": served,
        },
        "pipeline": {
            "fact_reorganization": bool(cfg.use_fact_reorganization),
            "domain_classifier": bool(cfg.use_domain_classifier) and classifier_available(),
            "statute_discrimination": bool(cfg.use_statute_discrimination),
        },
        "config_used": asdict(cfg),
    }

    if result.get("error"):
        return {
            "ok": False,
            "error": result["error"],
            "message": result.get("message"),
            "evidence": {
                "statutes": result.get("retrieved_statutes", []),
                "cases": result.get("retrieved_cases", []),
            },
            "meta": meta,
        }

    verification = dict(result.get("_verification", {}))
    fully = bool(verification.get("fully_grounded"))
    verification["label"] = VERIFICATION_LABELS[fully]

    # القائمة الكاملة (مسترجَعة دلالياً + مستخرجة من تسبيب السوابق) — تلزم كاملةً
    # لتعبئة article_number/law_name لأي تهمة، حتى لو استشهدت بمادة لم تُسترجَع
    # دلالياً وإنما وصلت فقط عبر سابقة (حالة 656 الموثّقة في README).
    statutes = result.get("_retrieved_statutes", [])
    charges = _enrich_charges(result.get("candidate_charges"), statutes,
                              verification.get("unsupported_charge_quotes", []))

    # أما قائمة «النصوص المسترجَعة» المعروضة للتصفّح فتقتصر على المسترجَعة
    # دلالياً وحدها (تحمل مفتاح score). المستخرجة من تسبيب السوابق ليس لها نسبة
    # تشابه أصلاً — لم تُقارَن بوقائع القضية قط — فعرضها بجانب نتائج مرتّبة
    # بالتشابه يوهم أنها منها. تبقى متاحة داخلياً للنموذج والتحقق، فلا شيء
    # يُفقد سوى ظهورها بهذه القائمة تحديداً.
    scored_statutes = [s for s in statutes if "score" in s]

    outcome = str(result.get("outcome") or "").strip()
    reasoning = result.get("reasoning") or {}

    return {
        "ok": True,
        "facts_summary": result.get("facts_summary"),
        "outcome": outcome,
        # قيمة خارج القائمة تعني أن النموذج خرج عن التعليمات؛ تعرضها الواجهة كتحذير.
        "outcome_is_standard": outcome in OUTCOMES,
        "final_charge": result.get("final_charge"),
        "candidate_charges": charges,
        "reasoning": {
            "charges_analysis": reasoning.get("charges_analysis"),
            "exclusion_notes": reasoning.get("exclusion_notes"),
            "precedent_alignment": reasoning.get("precedent_alignment"),
        },
        "cited_statutes": result.get("cited_statutes", []),
        "cited_precedents": result.get("cited_precedents", []),
        "suggested_penalty_range": result.get("suggested_penalty_range"),
        "confidence_note": result.get("confidence_note"),
        "verification": verification,
        "evidence": {
            "statutes": scored_statutes,
            "cases": result.get("_retrieved_cases", []),
            "classifier_candidates": result.get("_classifier_candidates", []),
            "reorganized_facts": result.get("_reorganized_facts", {}),
            "confusable_pairs": result.get("_confusable_pairs", []),
            "discrimination": result.get("_discrimination_results", []),
            "retrieval_text": result.get("_retrieval_text", ""),
        },
        "raw_model_output": {k: v for k, v in result.items() if not k.startswith("_")},
        "meta": meta,
    }


# واجهة الاستخدام

def warmup() -> None:
    """تحميل مسبق للميزتين اللتين تعتمد عليهما. يُحمَّل المصنّف كسولاً عند أول تنبؤ."""
    laws_rag.warmup()
    cases_rag.warmup()


def predict_judgment(facts_input: str, overrides: Optional[Dict] = None) -> Dict:
    """الدالة الوحيدة التي يحتاجها الـ route."""
    cfg = JudgmentConfig.from_request(overrides)
    t0 = time.time()
    result = LegalJudgmentPredictor(cfg).predict(facts_input)
    return build_api_payload(result, cfg, int((time.time() - t0) * 1000))
