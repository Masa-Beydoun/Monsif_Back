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
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

import config
from services import cases_rag, laws_rag


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
    case_threshold: float = 0.45
    max_output_tokens: int = 3072
    temperature: float = 0.1
    fact_reorg_max_tokens: int = 512
    groq_model: str = "qwen/qwen3.6-27b"
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
    max_confusable_pairs: int = 3
    discrimination_max_tokens: int = 400

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


# عميل النموذج اللغوي (Groq)

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


def _chat_json(messages: List[Dict], cfg: JudgmentConfig, max_tokens: int,
               temperature: float = 0.0) -> Optional[dict]:
    """استدعاء Groq وإرجاع JSON مُحلَّل، أو None عند الفشل."""
    try:
        client = _groq_client()
        kwargs = dict(
            model=cfg.groq_model, messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens, temperature=temperature,
        )
        try:
            # يعطّل وضع التفكير في نماذج Qwen3؛ غير مدعوم في كل النماذج.
            response = client.chat.completions.create(reasoning_effort="none", **kwargs)
        except Exception:
            response = client.chat.completions.create(**kwargs)

        raw = response.choices[0].message.content
        if not raw or not raw.strip():
            print("[judgment] أعاد النموذج اللغوي محتوى فارغاً.", flush=True)
            return None
        return json.loads(raw)
    except MissingAPIKey:
        raise
    except json.JSONDecodeError as e:
        print(f"[judgment] JSON غير صالح: {e}", flush=True)
        return None
    except Exception as e:
        print(f"[judgment] تعذّر الاتصال بخدمة النموذج اللغوي: {e}", flush=True)
        return None


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

def detect_confusable_statutes(retrieved_laws: List, cfg: JudgmentConfig) -> List[Dict]:
    """كشف تقريبي لأزواج المواد من القانون نفسه المتقاربة لفظياً.

    يعتمد تقاطع مفردات أول 40 كلمة، ويكفي لالتقاط حالات مثل 656/657 دون
    نموذج تشابه دلالي كامل.
    """
    pairs = []
    laws = list(retrieved_laws)
    for i, a in enumerate(laws):
        for b in laws[i + 1:]:
            if a.law_name != b.law_name:
                continue
            words_a = set(a.body.split()[:40])
            words_b = set(b.body.split()[:40])
            overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
            if overlap >= cfg.confusable_overlap_threshold:
                pairs.append({"article_a": a.article_id, "article_b": b.article_id,
                              "overlap": round(overlap, 2)})
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
    )
    if not isinstance(parsed, dict):
        return None
    parsed["article_a"] = pair["article_a"]
    parsed["article_b"] = pair["article_b"]
    return parsed


# جمع السياق

PENAL_CODE_HINTS = ["العقوبات", "قانون العقوبات"]


def _find_article_by_number(article_number: str,
                            law_name_contains: Optional[List[str]] = None):
    """بحث عن مادة برقم معيّن في المجموعة الكاملة، لا في النتائج المسترجعة."""
    rag = laws_rag.get_rag()
    article_number = str(article_number).strip()
    candidates = [a for a in rag.articles if a.article_number == article_number]
    if law_name_contains:
        filtered = [a for a in candidates
                    if any(h in (a.law_name or "") for h in law_name_contains)]
        if filtered:
            candidates = filtered
    return candidates[0] if candidates else None


def resolve_case_citations(case: Dict) -> List[Dict]:
    """استخراج نصوص المواد المذكورة صراحةً داخل سابقة، من مجموعة المواد."""
    resolved = []
    for num in case.get("penal_code_articles") or []:
        art = _find_article_by_number(num, law_name_contains=PENAL_CODE_HINTS)
        if art:
            resolved.append({
                "article_id": art.article_id, "law_name": art.law_name,
                "article_number": art.article_number, "body": art.body_raw,
                "source": f"مذكورة بالقضية رقم {case.get('case_number')}",
            })
    for entry in case.get("other_laws") or []:
        m = re.search(r"(\d+)", str(entry))
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

        existing_ids = {str(c.get("case_number")) for c in retrieved_cases}
        for label, matched_case in matched_precedents.items():
            cid = str(matched_case.get("case_number"))
            if cid not in existing_ids:
                new_case = dict(matched_case)
                new_case["_matched_candidate_label"] = label
                retrieved_cases.append(new_case)
                existing_ids.add(cid)
            else:
                for c in retrieved_cases:
                    if str(c.get("case_number")) == cid:
                        c["_matched_candidate_label"] = label

    cited_in_cases, seen_ids = [], {r.article_id for r in retrieved_laws}
    for case in retrieved_cases:
        for art in resolve_case_citations(case):
            if art["article_id"] not in seen_ids:
                seen_ids.add(art["article_id"])
                cited_in_cases.append(art)

    discrimination_results: List[Dict] = []
    if cfg.use_statute_discrimination:
        all_bodies = {r.article_id: r.body for r in retrieved_laws}
        all_bodies.update({a["article_id"]: a["body"] for a in cited_in_cases})
        for pair in detect_confusable_statutes(retrieved_laws, cfg):
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
        context = gather_context(facts_input, cfg)
        prompt = self.prompt_builder.build(context, cfg)
        raw_result = _chat_json([{"role": "user", "content": prompt}], cfg,
                                max_tokens=cfg.max_output_tokens,
                                temperature=cfg.temperature)

        if not isinstance(raw_result, dict):
            return {
                "error": "parsing_error",
                "message": "لم يُعِد النموذج اللغوي مخرجاً صالحاً. يرجى إعادة المحاولة.",
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
        raw_result["_retrieval_text"] = context.get("retrieval_text", "")
        return raw_result


# واجهة الاستخدام

def warmup() -> None:
    """تحميل مسبق للميزتين اللتين تعتمد عليهما. يُحمَّل المصنّف كسولاً عند أول تنبؤ."""
    laws_rag.warmup()
    cases_rag.warmup()


def predict_judgment(facts_input: str, overrides: Optional[Dict] = None) -> Dict:
    """الدالة الوحيدة التي يحتاجها الـ route."""
    cfg = JudgmentConfig.from_request(overrides)
    result = LegalJudgmentPredictor(cfg).predict(facts_input)
    result["_config_used"] = asdict(cfg)
    return result
