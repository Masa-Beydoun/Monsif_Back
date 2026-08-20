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

from services.judgment import (  # noqa: E402
    JudgmentConfig,
    LegalPromptBuilder,
    detect_confusable_statutes,
    normalize_arabic,
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
    print("=" * 60)
    print(f"نجح {PASS} | فشل {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
