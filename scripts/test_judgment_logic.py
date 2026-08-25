"""
اختبار منطق التحقق في ميزة الحكم الأولي، دون نماذج ودون استدعاء نموذج لغوي.

ينتهي خلال ثانية، فيصلح للتشغيل بعد كل تعديل على services/judgment.py.

    python scripts/test_judgment_logic.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import llm_client  # noqa: E402
from services.cases_rag import CasesRAG  # noqa: E402
from services.judgment import (  # noqa: E402
    JudgmentConfig,
    LegalPromptBuilder,
    build_api_payload,
    detect_confusable_statutes,
    normalize_arabic,
    resolve_llm,
    verify_all,
    verify_charge_quotes,
    verify_grounding,
    verify_precedent_consistency,
)

PASS, FAIL = 0, 0


def check(name: str, got, expected):
    global PASS, FAIL
    if got == expected:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}\n      got={got!r}\n      expected={expected!r}")


@dataclass
class FakeArt:
    """يحاكي SearchResult من laws_rag، بالحقول التي يستعملها التحقق فقط."""
    article_id: str
    body: str
    law_name: str = "قانون العقوبات العام"
    article_number: str = "656"
    status: str = "عادية"
    score: float = 0.9


BODY_656 = ("كل من أقدم قصدا على كتم أو اختلاس أو إتلاف سند يتضمن تعهدا أو إبراء "
            "يعاقب بالحبس من ستة أشهر إلى سنتين")
BODY_657 = ("كل من سلم إليه مبلغ من المال على سبيل الأمانة فامتنع عن رده "
            "يعاقب بالحبس من ثلاثة أشهر إلى سنة")

CTX = {
    "facts": "سلّم المدعى عليه مبلغاً بموجب سند موقع ورفض ردّه.",
    "retrieved_laws": [FakeArt("law:656", BODY_656, article_number="656")],
    "cited_in_cases": [],
    "retrieved_cases": [],
}


# 1. تحقق الاقتباسات

def test_charge_quotes():
    print("\n[1] verify_charge_quotes — الاقتباس لازم يكون حرفياً بنص المادة")
    cfg = JudgmentConfig()
    cases = [
        ("اقتباس صحيح", "كل من أقدم قصدا على كتم", False),
        ("اقتباس مختلق", "يشترط وجود سوء نية مثبت بالتحقيق", True),
        ("إعادة صياغة", "من يقوم عمداً بإخفاء سند تعهد", True),
        ("مسافات زايدة", "كل من   أقدم قصدا\nعلى كتم", False),
        ("اختلاف همزة", "كل من اقدم قصدا على كتم", False),
        ("اقتباس فارغ", "", True),
    ]
    for name, quote, expect_flag in cases:
        result = {"candidate_charges": [
            {"charge": "إساءة أمانة", "article_id": "law:656", "supporting_quote": quote}]}
        flags = verify_charge_quotes(result, CTX, cfg)
        check(name, bool(flags), expect_flag)

    # article_id غير موجود في السياق
    result = {"candidate_charges": [
        {"charge": "إساءة أمانة", "article_id": "law:999", "supporting_quote": "أي كلام"}]}
    check("article_id غير موجود", bool(verify_charge_quotes(result, CTX, cfg)), True)

    # اقتباس طويل جداً (تحايل على الفحص: نسخ نص المادة كامل بدل ركن محدد).
    # الحد = max_quote_words * 2 = 24 كلمة، فيُستعمل متن أطول لتجاوزه فعلاً.
    long_body = BODY_656 + " " + " ".join(f"كلمة{i}" for i in range(20))
    ctx_long = dict(CTX, retrieved_laws=[FakeArt("law:656", long_body)])
    long_quote = " ".join(long_body.split()[:30])          # 30 كلمة > 24
    result = {"candidate_charges": [
        {"charge": "إساءة أمانة", "article_id": "law:656", "supporting_quote": long_quote}]}
    flags = verify_charge_quotes(result, ctx_long, cfg)
    check("اقتباس أطول من 2× الحد", bool(flags), True)
    check("سبب الرفض هو الطول", "طويل جداً" in flags[0]["reason"], True)


# 2. تحقق هوية الاستشهادات

def test_grounding():
    print("\n[2] verify_grounding — ممنوع الاستشهاد بشي مو بالسياق")
    v = verify_grounding({"cited_statutes": ["law:656"], "cited_precedents": []}, CTX)
    check("استشهاد موجود بالسياق", v["grounded"], True)

    v = verify_grounding({"cited_statutes": ["law:999"], "cited_precedents": []}, CTX)
    check("استشهاد مختلق بمادة", v["unknown_statute_citations"], ["law:999"])

    v = verify_grounding({"cited_statutes": [], "cited_precedents": ["12345"]}, CTX)
    check("استشهاد مختلق بسابقة", v["unknown_case_citations"], ["12345"])

    # عقوبة تتضمن أرقاماً غير واردة في نص المادة؛ هلوسة محتملة.
    v = verify_grounding({"cited_statutes": ["law:656"], "cited_precedents": [],
                          "suggested_penalty_range": "الحبس من خمس إلى عشر سنوات و 5000 ليرة"}, CTX)
    check("أرقام عقوبة غير مسنودة", v["possible_penalty_hallucination_in"], ["law:656"])

    # عقوبة منقولة حرفياً من نص المادة؛ سليمة.
    v = verify_grounding({"cited_statutes": ["law:656"], "cited_precedents": [],
                          "suggested_penalty_range": "الحبس من ستة أشهر إلى سنتين"}, CTX)
    check("عقوبة منقولة حرفياً", v["possible_penalty_hallucination_in"], [])


# 3. كشف المواد المتشابهة لفظياً (ADAPT)

def test_confusable():
    print("\n[3] detect_confusable_statutes — كشف 656/657")
    cfg = JudgmentConfig(confusable_overlap_threshold=0.15)
    laws = [FakeArt("law:656", BODY_656, article_number="656"),
            FakeArt("law:657", BODY_657, article_number="657")]
    pairs = detect_confusable_statutes(laws, cfg)
    check("زوج متشابه مكتشف", len(pairs) >= 1, True)

    # مواد من قوانين مختلفة لا تُقارن.
    laws2 = [FakeArt("a:1", BODY_656, law_name="قانون العقوبات"),
             FakeArt("b:1", BODY_656, law_name="قانون التجارة")]
    check("قوانين مختلفة لا تُقارن", detect_confusable_statutes(laws2, cfg), [])

    # المادة الثانية من الزوج تصل عبر استشهادات السوابق كـ dict لا ككائن؛ لو لم
    # يقبل الكاشف الشكلين معاً لبقيت طبقة ADAPT معطّلة صامتة على بيانات المشروع.
    mixed = [FakeArt("law:657", BODY_657, article_number="657"),
             {"article_id": "law:656", "body": BODY_656,
              "law_name": "قانون العقوبات العام", "article_number": "656"}]
    pairs_mixed = detect_confusable_statutes(mixed, cfg)
    check("زوج مختلط (كائن + dict) مكتشف", len(pairs_mixed), 1)
    check("الزوج المختلط يحمل الطرفين",
          {pairs_mixed[0]["article_a"], pairs_mixed[0]["article_b"]},
          {"law:656", "law:657"})

    # المفردات المشتركة بين 656 و657 الحقيقيتين تقع في شطر العقوبة بذيل المادة،
    # فنافذة الـ40 كلمة الواردة في النوتبوك تقطعها وتُسقط الزوج (0.067 مقابل
    # 0.275 على المتن الكامل). المتنان هنا يحاكيان ذلك: مقدّمتان متباينتان
    # تماماً وذيل مشترك.
    head_a = "ألف باء تاء ثاء جيم حاء خاء دال ذال راء"
    head_b = "زاي سين شين صاد ضاد طاء ظاء عين غين فاء"
    tail = "يعاقب بالحبس من ثلاثة أشهر إلى سنتين وبالغرامة حتى ربع القيمة"
    tailed = [FakeArt("law:1", head_a + " " + tail),
              FakeArt("law:2", head_b + " " + tail)]
    check("المتن الكامل يكشف الزوج",
          len(detect_confusable_statutes(tailed, JudgmentConfig(
              confusable_overlap_threshold=0.15, confusable_window_words=0))), 1)
    check("النافذة الضيقة تقطع الذيل المشترك",
          detect_confusable_statutes(tailed, JudgmentConfig(
              confusable_overlap_threshold=0.15, confusable_window_words=10)), [])
    check("النافذة الافتراضية 40 = النوتبوك", JudgmentConfig().confusable_window_words, 40)

    # التكرار في الاتحاد لا يولّد زوجاً من المادة مع نفسها.
    dup = [FakeArt("law:656", BODY_656),
           {"article_id": "law:656", "body": BODY_656,
            "law_name": "قانون العقوبات العام"}]
    check("مادة مكرّرة لا تُقارن بنفسها", detect_confusable_statutes(dup, cfg), [])

    # النوتبوك يقتطع بترتيب الورود لا بالأقوى تشابهاً؛ نثبّت ذلك كي لا يعود
    # الترتيب التنازلي خلسةً ويغيّر أي زوج يصل إلى طبقة التمييز.
    many = [FakeArt("law:1", "ألف باء تاء ثاء"), FakeArt("law:2", "ألف باء تاء ثاء"),
            FakeArt("law:3", "ألف باء تاء ثاء جيم حاء خاء دال")]
    ordered = detect_confusable_statutes(many, JudgmentConfig(
        confusable_overlap_threshold=0.0, max_confusable_pairs=2))
    check("الاقتطاع بترتيب الورود (بلا فرز)",
          [(p["article_a"], p["article_b"]) for p in ordered],
          [("law:1", "law:2"), ("law:1", "law:3")])

    # عتبة عالية؛ لا أزواج متوقعة.
    cfg_high = JudgmentConfig(confusable_overlap_threshold=0.99)
    check("عتبة عالية تمنع الكشف", detect_confusable_statutes(laws, cfg_high), [])


# 4. تناقض السوابق

def test_precedent_consistency():
    print("\n[4] verify_precedent_consistency — تعارض رقم المادة مع تسبيب السابقة")
    ctx = dict(CTX)
    ctx["retrieved_laws"] = [FakeArt("law:656", BODY_656, article_number="656")]
    ctx["retrieved_cases"] = [{
        "case_number": "100", "crimes": ["إساءة أمانة"],
        "reasoning_text": "وحيث أن الفعل ينطبق على أحكام المادة 657 من قانون العقوبات",
        "_matched_candidate_label": "إساءة أمانة",
    }]
    result = {"candidate_charges": [{"charge": "إساءة أمانة", "article_id": "law:656"}]}
    out = verify_precedent_consistency(result, ctx)
    check("تعارض مكتشف (اخترنا 656 والسابقة بتقول 657)",
          len(out["precedent_article_mismatches"]), 1)

    # الرقم نفسه؛ لا تعارض.
    ctx["retrieved_cases"][0]["reasoning_text"] = "تنطبق أحكام المادة 656 من قانون العقوبات"
    out = verify_precedent_consistency(result, ctx)
    check("لا تعارض لما الرقمين متطابقين", out["precedent_article_mismatches"], [])

    # سابقة غير مرتبطة بتسمية مرشحة؛ تُتجاهل.
    ctx["retrieved_cases"][0]["reasoning_text"] = "أحكام المادة 999"
    ctx["retrieved_cases"][0].pop("_matched_candidate_label")
    out = verify_precedent_consistency(result, ctx)
    check("سابقة غير مرتبطة تُتجاهل", out["precedent_article_mismatches"], [])


# 5. verify_all المجمّع

def test_verify_all():
    print("\n[5] verify_all — fully_grounded بس لما كل الفحوص تنجح")
    cfg = JudgmentConfig()
    good = {
        "cited_statutes": ["law:656"], "cited_precedents": [],
        "candidate_charges": [{"charge": "إساءة أمانة", "article_id": "law:656",
                               "supporting_quote": "كل من أقدم قصدا على كتم"}],
    }
    check("نتيجة سليمة", verify_all(good, CTX, cfg)["fully_grounded"], True)

    bad = dict(good)
    bad["candidate_charges"] = [{"charge": "إساءة أمانة", "article_id": "law:656",
                                 "supporting_quote": "اقتباس مختلق تماماً"}]
    check("اقتباس مختلق يسقط fully_grounded",
          verify_all(bad, CTX, cfg)["fully_grounded"], False)


# 6. بناء الـ Prompt

def test_prompt_builder():
    print("\n[6] LegalPromptBuilder — كل الأقسام موجودة بالـ prompt")
    cfg = JudgmentConfig()
    ctx = {
        "facts": "وقائع تجريبية للاختبار",
        "reorganized_facts": {"subjective_motivation": "قصد الاحتيال",
                              "objective_behavior": "امتنع عن الرد",
                              "ex_post_facto_circumstances": "أنكر بالتحقيق"},
        "retrieved_laws": [FakeArt("law:656", BODY_656)],
        "cited_in_cases": [],
        "retrieved_cases": [{"case_number": "100", "similarity_score": 88.0,
                             "outcome": "إدانة", "crimes": ["إساءة أمانة"],
                             "facts_text": "وقائع السابقة", "reasoning_text": "التسبيب",
                             "judgment_text": "الحكم"}],
        "candidate_charges_from_classifier": [
            {"label": "إساءة أمانة", "score": 1.23, "predicted": True}],
        "discrimination_results": [{
            "article_a": "law:656", "article_b": "law:657",
            "distinguishing_questions": [
                {"question": "هل يوجد سند مكتوب؟", "answer_from_facts": "نعم",
                 "points_to_article": "law:656"}]}],
    }
    prompt = LegalPromptBuilder().build(ctx, cfg)
    for label, needle in [
        ("الوقائع الأصلية", "وقائع تجريبية للاختبار"),
        ("كتلة النصوص القانونية", "[article_id=law:656]"),
        ("كتلة السوابق", "[case_number=100]"),
        ("كتلة المصنّف", "إساءة أمانة — score=1.23"),
        ("كتلة إعادة التنظيم", "الدافع الذاتي: قصد الاحتيال"),
        ("كتلة التمييز", "هل يوجد سند مكتوب؟"),
        ("مخطط الإخراج", '"outcome"'),
        ("حد الاقتباس مُعبّأ", "12 كلمة"),
    ]:
        check(label, needle in prompt, True)

    # لما نطفي إعادة التنظيم، ما بتنعرض للـ LLM
    cfg_off = JudgmentConfig(show_reorganized_facts_to_llm=False)
    check("كتلة إعادة التنظيم تختفي عند تعطيلها",
          "الدافع الذاتي" in LegalPromptBuilder().build(ctx, cfg_off), False)


# 7. تمرير الإعدادات من الطلب

def test_config_overrides():
    print("\n[7] JudgmentConfig.from_request — الأولوية للطلب")
    cfg = JudgmentConfig.from_request(None)
    check("يقرأ الافتراضيات", cfg.top_k_laws, 5)

    cfg = JudgmentConfig.from_request({"top_k_laws": 9, "case_threshold": 0.7})
    check("تجاوز top_k_laws", cfg.top_k_laws, 9)
    check("تجاوز case_threshold", cfg.case_threshold, 0.7)

    cfg = JudgmentConfig.from_request({"حقل_غير_موجود": 1, "top_k_cases": None})
    check("حقول غير معروفة تُتجاهل", cfg.top_k_cases, 3)


def test_notebook_parity_defaults():
    print("\n[12] الافتراضيات مطابِقة لخلية ## 3-ADAPT layer")
    cfg = JudgmentConfig()
    # كل قيمة هنا منسوخة من خلية النوتبوك؛ أي انحراف يجب أن يكون قراراً صريحاً.
    for name, expected in [
        ("top_k_laws", 5), ("top_k_cases", 3), ("case_threshold", 0.45),
        ("min_law_score", 0.0), ("exclude_repealed_laws", True),
        ("max_output_tokens", 3072), ("fact_reorg_max_tokens", 512),
        ("max_quote_words", 12), ("top_k_classifier_labels", 5),
        ("classifier_min_score", -1.0), ("precedent_pool_size", 15),
        ("use_domain_classifier", True), ("use_fact_reorganization", True),
        ("show_reorganized_facts_to_llm", True), ("use_statute_discrimination", True),
        ("confusable_overlap_threshold", 0.15), ("max_confusable_pairs", 3),
        ("discrimination_max_tokens", 400),
        # النافذة 40 والمفتاحان أدناه هي نقاط الاختلاف الوحيدة الممكنة؛
        # افتراضها سلوك النوتبوك حرفياً.
        ("confusable_window_words", 40),
        ("confusable_scan_cited_articles", False),
        ("precedent_key_by_uid", False),
        ("strict_citation_law_match", False),
        # النوتبوك يستدعي search() بلا عتبة نسبية ولا قصّ ذيل.
        ("law_min_score_ratio", 0.0), ("law_score_drop_ratio", 0.0),
    ]:
        check(f"{name} = {expected!r}", getattr(cfg, name), expected)


def test_evidence_statutes_scored_only():
    print("\n[15] evidence.statutes يستبعد المواد بلا نسبة تشابه")
    result = {
        "facts_summary": "ملخص",
        "outcome": "إدانة",
        "final_charge": "إساءة أمانة",
        "candidate_charges": [
            {"charge": "إساءة أمانة", "article_id": "law:656",
             "supporting_quote": "كل من أقدم قصدا", "elements_match": True},
        ],
        "reasoning": {}, "cited_statutes": ["law:656"], "cited_precedents": [],
        "_verification": {
            "grounded": True, "fully_grounded": True,
            "unknown_statute_citations": [], "unknown_case_citations": [],
            "possible_penalty_hallucination_in": [],
            "unsupported_charge_quotes": [], "precedent_article_mismatches": [],
        },
        "_retrieved_statutes": [
            # مسترجَعة دلالياً: تحمل score.
            {"article_id": "law:657", "law_name": "قانون العقوبات العام",
             "article_number": "657", "status": "عادية", "score": 0.746,
             "body": BODY_657},
            # مستخرجة من تسبيب سابقة: بلا score إطلاقاً (شكل resolve_case_citations).
            {"article_id": "law:656", "law_name": "قانون العقوبات العام",
             "article_number": "656", "body": BODY_656,
             "source": "مذكورة بالقضية رقم 512"},
        ],
        "_retrieved_cases": [], "_classifier_candidates": [],
        "_reorganized_facts": {}, "_discrimination_results": [], "_confusable_pairs": [],
    }
    payload = build_api_payload(result, JudgmentConfig(), took_ms=1)

    check("evidence.statutes يستبعد غير المُسترجَعة",
          [s["article_id"] for s in payload["evidence"]["statutes"]], ["law:657"])
    check("evidence.statutes يبقي المُسترجَعة", len(payload["evidence"]["statutes"]), 1)
    # التهمة استشهدت بـ656 (غير المُسترجَعة) ومع ذلك بياناتها مكتملة: enrich_charges
    # يستعمل القائمة الكاملة داخلياً، لا evidence.statutes المُصفّاة.
    check("بيانات التهمة مكتملة رغم الاستبعاد من evidence",
          payload["candidate_charges"][0]["article_number"], "656")
    check("التهمة غير موسومة رغم عدم ظهور مادتها بالأدلة",
          payload["candidate_charges"][0]["flagged"], False)


def test_llm_fallback():
    print("\n[14] التحويل إلى الواجهة الاحتياطية")
    import config as C

    calls = []

    def fake_attempt(messages, *, backend=None, model=None, **kw):
        calls.append((backend, model))
        if backend == "hf":
            raise llm_client.LLMError("نفدت الحصة", status_code=402, backend="hf")
        return '{"ok": true}'

    original_attempt = llm_client._attempt
    original = (C.LLM_ENABLE_FALLBACK, C.LLM_FALLBACK_BACKEND,
                C.LLM_FALLBACK_MODEL, C.OPENROUTER_API_KEY)
    llm_client._attempt = fake_attempt
    C.LLM_ENABLE_FALLBACK = True
    C.LLM_FALLBACK_BACKEND = "openrouter"
    C.LLM_FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct"
    C.OPENROUTER_API_KEY = "sk-or-test"
    try:
        check("هدف الاحتياطي", llm_client.fallback_target(),
              ("openrouter", "meta-llama/llama-3.3-70b-instruct"))

        used = {}
        out = llm_client.chat_json([{"role": "user", "content": "x"}],
                                   backend="hf", model="m", used=used)
        check("النتيجة تصل رغم فشل الأساسية", out, {"ok": True})
        check("جُرّبت الاثنتان بالترتيب", calls,
              [("hf", "m"), ("openrouter", "meta-llama/llama-3.3-70b-instruct")])
        check("الواجهة الفعلية مسجّلة", used["backend"], "openrouter")
        check("علم التحويل مرفوع", used["fallback_used"], True)
        check("سبب فشل الأساسية محفوظ", "نفدت الحصة" in used["primary_error"], True)

        # 400 خطأ في الطلب نفسه: سيتكرر عند أي مزوّد، فلا يُهدَر استدعاء ثانٍ.
        calls.clear()

        def bad_request(messages, *, backend=None, model=None, **kw):
            calls.append((backend, model))
            raise llm_client.LLMError("طلب معطوب", status_code=400, backend=backend)

        llm_client._attempt = bad_request
        try:
            llm_client.chat([{"role": "user", "content": "x"}], backend="hf", model="m")
            check("400 يجب أن يرمي", False, True)
        except llm_client.LLMError:
            check("400 لا يُحوَّل", len(calls), 1)

        # بلا مفتاح للاحتياطية لا تحويل، ويظهر خطأ الأساسية كما هو.
        llm_client._attempt = fake_attempt
        C.OPENROUTER_API_KEY = ""
        check("بلا مفتاح احتياطي لا هدف", llm_client.fallback_target(), None)
        calls.clear()
        try:
            llm_client.chat([{"role": "user", "content": "x"}], backend="hf", model="m")
            check("يجب أن يرمي بلا احتياطي", False, True)
        except llm_client.LLMError as e:
            check("خطأ الأساسية يظهر كما هو", "نفدت الحصة" in str(e), True)
            check("بلا محاولة ثانية", len(calls), 1)

        # التعطيل الصريح يمنع التحويل حتى مع وجود مفتاح.
        C.OPENROUTER_API_KEY = "sk-or-test"
        C.LLM_ENABLE_FALLBACK = False
        check("التعطيل يلغي الهدف", llm_client.fallback_target(), None)
    finally:
        llm_client._attempt = original_attempt
        (C.LLM_ENABLE_FALLBACK, C.LLM_FALLBACK_BACKEND,
         C.LLM_FALLBACK_MODEL, C.OPENROUTER_API_KEY) = original


def test_citation_resolution():
    print("\n[13] حلّ استشهادات السوابق — الفضفاض مقابل الصارم")
    import services.judgment as J

    # مجموعة مصغّرة تحاكي التصادم الحقيقي: المادة 129 موجودة في ثلاثة قوانين،
    # واسم القانون العسكري يتضمّن كلمة «العقوبات» فيلتقطه التلميح الفضفاض.
    @dataclass
    class Art:
        article_id: str
        article_number: str
        law_name: str
        body_raw: str = "نص"

    MILITARY = "مرسوم تشريعي رقم 61 عام 1950 - قانون العقوبات واصول المحاكمات العسكرية"
    GENERAL = "مرسوم تشريعي رقم 148 لعام 1949 - قانون العقوبات العام"
    TRAFFIC = "مرسوم تشريعي رقم 11 عام 2008 - تعديل قانون السير والمركبات"

    class FakeRag:
        # ترتيب متعمَّد: العسكري أولاً، لأن الكود الفضفاض يأخذ candidates[0].
        articles = [
            Art("mil:129", "129", MILITARY),
            Art("gen:129", "129", GENERAL),
            Art("traffic:1", "1", TRAFFIC),
            Art("gen:1", "1", GENERAL),
        ]

    original = J.laws_rag.get_rag
    J.laws_rag.get_rag = lambda: FakeRag()
    try:
        case = {"case_number": "401", "penal_code_articles": ["129"],
                "other_laws": ["مرسوم العفو العام رقم 13 لعام 2021"]}

        loose = J.resolve_case_citations(case, JudgmentConfig())
        strict = J.resolve_case_citations(
            case, JudgmentConfig(strict_citation_law_match=True))

        # الفضفاض: يلتقط العسكري لأن اسمه يتضمّن «العقوبات» وهو الأول ترتيباً.
        check("الفضفاض يصيب القانون العسكري",
              [a["article_id"] for a in loose if a["article_number"] == "129"],
              ["mil:129"])
        check("الصارم يصيب قانون العقوبات العام",
              [a["article_id"] for a in strict if a["article_number"] == "129"],
              ["gen:129"])

        # ‏«مرسوم العفو رقم 13» رقمُ مرسوم لا رقم مادة؛ الفضفاض يقرأه كمادة 13
        # (لا وجود لها هنا فيُسقَط)، والصارم يرفضه لغياب كلمة «مادة».
        check("الصارم لا يحلّ مرجع قانون بلا كلمة مادة",
              [a for a in strict if a["article_number"] == "13"], [])

        # رقم مادة موجود في قانونين وبلا تلميح ⇒ الصارم يُسقطه بدل تخمينه.
        case2 = {"case_number": "1", "penal_code_articles": [],
                 "other_laws": ["مادة 1"]}
        check("الصارم يُسقط الملتبس",
              J.resolve_case_citations(case2, JudgmentConfig(
                  strict_citation_law_match=True)), [])
        check("الفضفاض يخمّن الأول",
              [a["article_id"] for a in J.resolve_case_citations(case2, JudgmentConfig())],
              ["traffic:1"])

        # اسم القانون داخل القوسين يحسم الالتباس في الوضع الصارم.
        case3 = {"case_number": "2", "penal_code_articles": [],
                 "other_laws": ["مادة 1 (قانون العقوبات العام)"]}
        check("اسم القانون في النص يحسم الاختيار",
              [a["article_id"] for a in J.resolve_case_citations(
                  case3, JudgmentConfig(strict_citation_law_match=True))],
              ["gen:1"])
    finally:
        J.laws_rag.get_rag = original


def test_llm_resolution():
    print("\n[8] اختيار الواجهة والنموذج")
    cfg = JudgmentConfig(llm_backend="hf", llm_model="meta-llama/Llama-3.1-8B-Instruct")
    check("hf صريح", resolve_llm(cfg), ("hf", "meta-llama/Llama-3.1-8B-Instruct"))

    # groq_model حقل مهمل: لو سُرّب لواجهة hf لأعاد المزوّد 404.
    cfg = JudgmentConfig(llm_backend="hf", groq_model="llama-3.3-70b-versatile")
    check("groq_model لا يسرّب إلى hf",
          resolve_llm(cfg)[1] == "llama-3.3-70b-versatile", False)

    cfg = JudgmentConfig(llm_backend="groq", groq_model="llama-3.3-70b-versatile")
    check("groq_model يُقرأ مع groq", resolve_llm(cfg),
          ("groq", "llama-3.3-70b-versatile"))

    cfg = JudgmentConfig(llm_backend="groq", llm_model="x/y", groq_model="a/b")
    check("llm_model يعلو groq_model", resolve_llm(cfg)[1], "x/y")


def test_json_extraction():
    print("\n[9] استخراج JSON من مخرَج النموذج")
    fenced = "```json\n{\"a\": 1}\n```"
    check("JSON نظيف", llm_client.extract_json('{"a": 1}'), {"a": 1})
    check("مغلّف بـ fences", llm_client.extract_json(fenced), {"a": 1})
    check("جملة تمهيدية قبله",
          llm_client.extract_json('طبعاً، إليك النتيجة:\n{"a": 1}'), {"a": 1})
    check("نص بعده", llm_client.extract_json('{"a": 1}\nأتمنى أن يفيدك'), {"a": 1})
    # قوس داخل سلسلة نصية يجب ألّا يُنهي الكائن باكراً.
    check("قوس داخل سلسلة", llm_client.extract_json('{"a": "x}y"}'), {"a": "x}y"})
    check("بلا JSON", llm_client.extract_json("ما في جواب"), None)
    check("فارغ", llm_client.extract_json(""), None)
    check("مصفوفة لا كائن", llm_client.extract_json("[1, 2]"), None)


def test_api_payload():
    print("\n[10] بنية الاستجابة للواجهة الأمامية")
    result = {
        "facts_summary": "ملخص",
        "outcome": "إدانة",
        "final_charge": "إساءة أمانة",
        "candidate_charges": [
            {"charge": "إساءة أمانة", "article_id": "law:656",
             "supporting_quote": "كل من أقدم قصدا", "elements_match": True},
            {"charge": "مختلقة", "article_id": "law:999", "supporting_quote": "لا شيء"},
        ],
        "reasoning": {"charges_analysis": "أ", "exclusion_notes": "ب",
                      "precedent_alignment": "ج"},
        "cited_statutes": ["law:656"],
        "cited_precedents": ["512"],
        "_verification": {
            "grounded": False, "fully_grounded": False,
            "unknown_statute_citations": [], "unknown_case_citations": [],
            "possible_penalty_hallucination_in": [],
            "unsupported_charge_quotes": [
                {"charge": "مختلقة", "article_id": "law:999", "reason": "غير موجود"}],
            "precedent_article_mismatches": [],
        },
        "_retrieved_statutes": [
            {"article_id": "law:656", "law_name": "قانون العقوبات العام",
             "article_number": "656", "body": BODY_656}],
        "_retrieved_cases": [], "_classifier_candidates": [],
        "_reorganized_facts": {}, "_discrimination_results": [], "_confusable_pairs": [],
    }
    payload = build_api_payload(result, JudgmentConfig(), took_ms=42)

    check("ok", payload["ok"], True)
    check("outcome قياسي", payload["outcome_is_standard"], True)
    check("عدد التهم", len(payload["candidate_charges"]), 2)

    ok_charge, bad_charge = payload["candidate_charges"]
    check("التهمة السليمة غير موسومة", ok_charge["flagged"], False)
    # الواجهة تلوّن من هذين الحقلين مباشرة دون قراءة كتلة التحقق.
    check("التهمة المختلقة موسومة", bad_charge["flagged"], True)
    check("سبب الوسم مرفق", bad_charge["flag_reason"], "غير موجود")
    check("بيانات المادة مُثرّاة", ok_charge["article_number"], "656")
    check("مادة مجهولة بلا بيانات", bad_charge["law_name"], None)

    check("عنوان التحقق جاهز للعرض",
          payload["verification"]["label"].startswith("استشهادات أو أركان"), True)
    check("الأدلة مجمّعة", sorted(payload["evidence"]),
          ["cases", "classifier_candidates", "confusable_pairs", "discrimination",
           "reorganized_facts", "retrieval_text", "statutes"])
    # الحقول الداخلية لا تتسرّب إلى المخرَج الخام المعروض.
    check("raw_model_output بلا حقول داخلية",
          any(k.startswith("_") for k in payload["raw_model_output"]), False)
    check("meta.took_ms", payload["meta"]["took_ms"], 42)

    # outcome خارج القائمة الثلاثية = خروج عن التعليمات، تعرضه الواجهة كتحذير.
    result["outcome"] = "غرامة مالية"
    check("outcome غير قياسي",
          build_api_payload(result, JudgmentConfig(), 1)["outcome_is_standard"], False)

    # مسار الفشل لازم يبقى قابلاً للعرض لا أن يرمي KeyError.
    failed = build_api_payload(
        {"error": "parsing_error", "message": "فشل", "retrieved_statutes": [],
         "retrieved_cases": []}, JudgmentConfig(), 5)
    check("مسار الفشل ok=False", failed["ok"], False)
    check("مسار الفشل يحمل رسالة", failed["message"], "فشل")


def test_cases_dedupe():
    print("\n[11] إزالة تكرار السوابق — بالمحتوى لا برقم القضية")
    # ‏case_number ليس فريداً: في المجموعة 48 رقماً مكرراً، 47 منها قضايا
    # مختلفة فعلاً. الإزالة على أساس الرقم كانت ستحذف قضايا حقيقية.
    same_a = {"case_number": "579", "file_name": "579_2022_verified_fraud.docx",
              "facts_text": "وقائع", "judgment_text": "حكم", "claims_text": "ادعاء",
              "reasoning_text": "تسبيب فيه فرق OCR"}
    same_b = {"case_number": "579", "file_name": "579_verified_fraud.docx",
              "facts_text": "وقائع", "judgment_text": "حكم", "claims_text": "ادعاء",
              "reasoning_text": "تسبيب فيه فرق OCR اخر"}
    # نفس الرقم لكن قضية مختلفة تماماً — يجب أن تبقى.
    other = {"case_number": "579", "file_name": "579_other.docx",
             "facts_text": "وقائع مختلفة", "judgment_text": "حكم اخر",
             "claims_text": "ادعاء اخر"}

    check("المكرر الحقيقي يُطوى", len(CasesRAG._dedupe([same_a, same_b])), 1)
    check("يُبقى أول ظهور",
          CasesRAG._dedupe([same_a, same_b])[0]["file_name"],
          "579_2022_verified_fraud.docx")
    # الفرق في reasoning_text وحده لا يمنع الطيّ: هو أثر OCR لا قضية أخرى.
    check("اختلاف التسبيب وحده لا يمنع الطيّ",
          CasesRAG._content_key(same_a), CasesRAG._content_key(same_b))
    check("قضية مختلفة بنفس الرقم تبقى",
          len(CasesRAG._dedupe([same_a, same_b, other])), 2)
    check("الترتيب محفوظ",
          [c["file_name"] for c in CasesRAG._dedupe([same_a, other, same_b])],
          ["579_2022_verified_fraud.docx", "579_other.docx"])
    check("قائمة فارغة", CasesRAG._dedupe([]), [])


def main() -> int:
    print("اختبار منطق الحكم الأولي (دون نماذج ودون نموذج لغوي)")
    print("=" * 60)
    test_charge_quotes()
    test_grounding()
    test_confusable()
    test_precedent_consistency()
    test_verify_all()
    test_prompt_builder()
    test_config_overrides()
    test_llm_resolution()
    test_json_extraction()
    test_api_payload()
    test_cases_dedupe()
    test_notebook_parity_defaults()
    test_citation_resolution()
    test_llm_fallback()
    test_evidence_statutes_scored_only()
    print("=" * 60)
    print(f"نجح {PASS} | فشل {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
