"""
طبقة موحّدة لاستدعاء النموذج اللغوي عبر واجهة HTTP متوافقة مع OpenAI.

تدعم واجهتين، وكلتاهما استدعاء شبكة فقط — لا تُنزَّل أي أوزان محلياً:

  hf    → HuggingFace Inference Providers router (الافتراضي)
          https://router.huggingface.co/v1/chat/completions
          يحتاج HF_TOKEN، ويخدم نماذج Meta Llama المقيّدة الوصول متى مُنحت
          للحساب موافقة على مستودعها.
  groq  → https://api.groq.com/openai/v1/chat/completions، يحتاج GROQ_API_KEY.

سبب توحيدهما في ملف واحد: الميزتان اللتان تستدعيان نموذجاً لغوياً (الحكم الأولي
ونماذج العقود) كانتا تكرران عميل Groq ومعالجة الأخطاء. أي واجهة جديدة تُضاف هنا
مرة واحدة.

ملاحظتان عمليتان عن مزوّدي HuggingFace:
  1. ليس كل مزوّد يدعم response_format=json_object؛ عند رفضه تُعاد المحاولة
     بدونه، ويُستخرَج الـ JSON من النص باستخراج متسامح.
  2. المزوّد قد يرجع 503 أثناء إقلاع النموذج و429 عند تجاوز الحصة؛ كلاهما
     يُعاد معه المحاولة بتراجع أسّي.
"""

import json
import re
import time
from typing import Dict, List, Optional, Tuple

import config


class LLMError(RuntimeError):
    """فشل استدعاء النموذج اللغوي.

    يحمل رمز حالة HTTP (أو None عند عطل شبكة) كي تقرّر طبقة التحويل الاحتياطي
    هل يستحق الخطأ تجربة واجهة أخرى أم أنه خطأ في الطلب نفسه سيتكرر عند الجميع.
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None,
                 backend: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.backend = backend


class MissingAPIKey(LLMError):
    """المفتاح المطلوب للواجهة المختارة غير مضبوط في .env."""


# وصف الواجهات المدعومة

_BACKENDS = {
    "hf": {
        "label": "HuggingFace Inference Providers",
        "base_url": lambda: config.HF_BASE_URL,
        "api_key": lambda: config.HF_TOKEN,
        "api_keys": lambda: _non_empty(config.HF_TOKEN),
        "default_model": lambda: config.HF_MODEL,
        "key_env": "HF_TOKEN",
        "key_hint": "HF_TOKEN=hf_...    (من https://huggingface.co/settings/tokens)",
    },
    "groq": {
        "label": "Groq",
        "base_url": lambda: config.GROQ_BASE_URL,
        "api_key": lambda: config.GROQ_API_KEY,
        "api_keys": lambda: _non_empty(config.GROQ_API_KEY),
        "default_model": lambda: config.GROQ_MODEL,
        "key_env": "GROQ_API_KEY",
        "key_hint": "GROQ_API_KEY=gsk_...  (من https://console.groq.com/keys)",
    },
    # قيس من شبكة المشروع: Groq وحده يردّ 403 «Access denied» قبل قراءة المفتاح
    # أصلاً (حجب على مستوى الشبكة)، بينما الواجهتان أدناه تستجيبان طبيعياً ولا
    # ينقصهما إلا مفتاح. كلتاهما متوافقة مع OpenAI فتعملان بنفس المسار تماماً.
    #
    # OpenRouter تحديداً يدعم حتى ثلاثة مفاتيح (ثلاثة حسابات منفصلة): إن نفدت
    # حصة أحدها (401/402/403/429) يُجرَّب التالي تلقائياً قبل اللجوء لواجهة
    # مزوّد أخرى بالكامل. للتعديل الفوري: غيّري القيم في .env وأعيدي تشغيل
    # الخادم فقط — لا حاجة لمسّ هذا الملف.
    "openrouter": {
        "label": "OpenRouter",
        "base_url": lambda: config.OPENROUTER_BASE_URL,
        "api_key": lambda: config.OPENROUTER_API_KEY,
        "api_keys": lambda: _non_empty(
            config.OPENROUTER_API_KEY, config.OPENROUTER_API_KEY_2, config.OPENROUTER_API_KEY_3),
        "default_model": lambda: config.OPENROUTER_MODEL,
        "key_env": "OPENROUTER_API_KEY",
        "key_hint": ("OPENROUTER_API_KEY=sk-or-...            (من https://openrouter.ai/keys)\n"
                     "    OPENROUTER_API_KEY_2=sk-or-...          (حساب ثانٍ، اختياري)\n"
                     "    OPENROUTER_API_KEY_3=sk-or-...          (حساب ثالث، اختياري)"),
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": lambda: config.GEMINI_BASE_URL,
        "api_key": lambda: config.GOOGLE_API_KEY,
        "api_keys": lambda: _non_empty(config.GOOGLE_API_KEY),
        "default_model": lambda: config.GEMINI_MODEL,
        "key_env": "GOOGLE_API_KEY",
        "key_hint": "GOOGLE_API_KEY=...  (من https://aistudio.google.com/apikey)",
    },
}


def _non_empty(*keys: str) -> List[str]:
    return [k.strip() for k in keys if k and k.strip()]


def available_backends() -> List[str]:
    return sorted(_BACKENDS)


def resolve(backend: Optional[str] = None,
            model: Optional[str] = None) -> Tuple[str, str]:
    """يعيد (اسم الواجهة، اسم النموذج) بعد تطبيق الافتراضيات."""
    name = (backend or config.LLM_BACKEND or "hf").strip().lower()
    if name not in _BACKENDS:
        raise LLMError(
            f"واجهة نموذج لغوي غير معروفة: «{name}». "
            f"المدعوم: {', '.join(available_backends())}."
        )
    return name, (model or "").strip() or _BACKENDS[name]["default_model"]()


def is_configured(backend: Optional[str] = None) -> bool:
    """هل مفتاح الواجهة موجود؟ يستعملها مسار الحالة دون رمي خطأ."""
    try:
        name, _ = resolve(backend)
    except LLMError:
        return False
    return bool(_BACKENDS[name]["api_keys"]())


def status() -> Dict:
    """ملخّص جاهز للعرض في مسارات الحالة والتشخيص."""
    active, active_model = resolve()
    return {
        "active_backend": active,
        "active_model": active_model,
        "configured": is_configured(active),
        "backends": {
            name: {
                "label": spec["label"],
                "base_url": spec["base_url"](),
                "default_model": spec["default_model"](),
                "key_env": spec["key_env"],
                "key_set": bool(spec["api_keys"]()),
                "keys_configured": len(spec["api_keys"]()),
            }
            for name, spec in _BACKENDS.items()
        },
    }


def _require_keys(name: str) -> List[str]:
    spec = _BACKENDS[name]
    keys = spec["api_keys"]()
    if not keys:
        raise MissingAPIKey(
            f"{spec['key_env']} غير مضبوط، وهو مطلوب لواجهة «{spec['label']}». "
            f"أضيفيه لملف .env بجذر المشروع:\n    {spec['key_hint']}"
        )
    return keys


# استخراج JSON متسامح

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.+?)\s*```", re.S)


def _first_json_object(text: str) -> Optional[str]:
    """أول كائن JSON متوازن الأقواس في النص، مع تجاهل الأقواس داخل السلاسل."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json(raw: str) -> Optional[dict]:
    """يحوّل مخرَج النموذج إلى dict.

    نماذج Llama تميل لتغليف الـ JSON بـ```json أو إضافة جملة تمهيدية حتى مع
    التعليمات الصريحة، لذا لا يكفي json.loads المباشر.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    for candidate in (text, _first_json_object(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# الاستدعاء

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# ‏HuggingFace يردّ 400 لا 404 على نموذج لا يخدمه أي مزوّد مفعَّل
# (code=model_not_supported). هذا عطل توفّر يستحق التحويل، لا خطأ في صياغة
# الطلب — وبدون التمييز يُعامَل كخطأ نهائي ولا تُجرَّب الواجهة الاحتياطية.
_MODEL_UNAVAILABLE_RE = re.compile(
    r"model_not_supported|model_not_found|does not exist|is not supported"
    r"|is not a chat model|no provider", re.I)


# أعطال تخص المفتاح نفسه لا الطلب: نفاد الحصة (402/429) أو مفتاح تالف/ناقص
# الصلاحيات (401/403). حين تتوفر عدة مفاتيح لنفس الواجهة (OpenRouter) يُجرَّب
# التالي منها فوراً بلا انتظار، إذ لا فائدة من إعادة محاولة مفتاح مستنفَد.
_KEY_EXHAUSTED_STATUS = {401, 402, 403, 429}


def _attempt(messages: List[Dict], *, backend: Optional[str] = None,
             model: Optional[str] = None, max_tokens: int = 1024,
             temperature: float = 0.0, json_mode: bool = True,
             timeout: Optional[int] = None,
             max_retries: Optional[int] = None) -> str:
    """استدعاء واجهة واحدة وإرجاع النص الخام. يرمي LLMError عند الفشل.

    عند توفّر أكثر من مفتاح لنفس الواجهة (OpenRouter بثلاثة حسابات مثلاً)،
    يُجرَّب كل مفتاح على حدة: 5xx/انقطاع الشبكة تُعاد محاولتها على نفس المفتاح
    بتراجع أسّي كالسابق، أما استنفاد المفتاح (401/402/403/429) فيبدّل فوراً
    إلى المفتاح التالي دون انتظار.
    """
    import requests

    name, model_id = resolve(backend, model)
    spec = _BACKENDS[name]
    keys = _require_keys(name)

    url = spec["base_url"]().rstrip("/") + "/chat/completions"
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    # نماذج Qwen3 تُخرِج سلسلة تفكير قبل الجواب ما لم يُطفأ صراحةً، فتلتهم سقف
    # التوكنات وقد تقطع الـ JSON. المعامل غير مدعوم عند كل مزوّد، ويُسقَط تلقائياً
    # عند رفضه (نفس مسار response_format أدناه).
    if "qwen" in model_id.lower():
        payload["reasoning_effort"] = "none"

    timeout = timeout or config.LLM_TIMEOUT
    attempts = max(1, max_retries or config.LLM_MAX_RETRIES)
    last_error = ""
    last_status: Optional[int] = None

    for key_index, key in enumerate(keys):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        key_label = f"{spec['label']}" if len(keys) == 1 else f"{spec['label']} (مفتاح {key_index + 1}/{len(keys)})"

        for attempt in range(attempts):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            except Exception as e:                      # شبكة أو مهلة
                last_error = f"تعذّر الاتصال بـ{key_label}: {e}"
                last_status = None
                if attempt + 1 < attempts:
                    time.sleep(2 ** attempt)
                    continue
                break  # جرّبي المفتاح التالي إن وُجد

            if response.status_code == 200:
                try:
                    data = response.json()
                    return data["choices"][0]["message"]["content"] or ""
                except Exception as e:
                    raise LLMError(f"استجابة غير متوقعة من {spec['label']}: {e}",
                                   status_code=200, backend=name)

            body = response.text[:400]

            # يُفحص قبل إسقاط المعاملات: وإلا استُهلكت محاولة على إسقاط
            # response_format بينما السبب الحقيقي أن النموذج غير مخدوم أصلاً.
            # هذا عطل نموذج لا عطل مفتاح، فلا فائدة من تبديل المفتاح.
            if response.status_code in (400, 404) and _MODEL_UNAVAILABLE_RE.search(body):
                raise LLMError(
                    f"النموذج «{model_id}» غير متاح على {spec['label']} "
                    f"(HTTP {response.status_code}). جرّبي اسماً آخر أو واجهة أخرى.\n{body}",
                    status_code=404, backend=name)

            # مزوّدون كثر لا يدعمون هذين المعاملين؛ نسقطهما ونعيد المحاولة فوراً.
            if response.status_code in (400, 422) and "reasoning_effort" in payload:
                print(f"[llm] {name}: المزوّد لا يدعم reasoning_effort؛ سيُعاد الطلب بدونه.",
                      flush=True)
                payload.pop("reasoning_effort", None)
                continue
            if json_mode and response.status_code in (400, 422) and "response_format" in payload:
                print(f"[llm] {name}: المزوّد لا يدعم response_format؛ "
                      f"سيُعاد الطلب بدونه ويُستخرَج الـ JSON من النص.", flush=True)
                payload.pop("response_format", None)
                continue

            if response.status_code == 404:
                raise LLMError(
                    f"النموذج «{model_id}» غير متاح على {spec['label']} (HTTP 404). "
                    f"جرّبي اسماً آخر عبر متغيّر البيئة، أو غيّري الواجهة.\n{body}",
                    status_code=404, backend=name)

            last_status = response.status_code
            if response.status_code in _KEY_EXHAUSTED_STATUS:
                last_error = f"{key_label} أعاد HTTP {response.status_code}: {body}"
                if key_index + 1 < len(keys):
                    print(f"[llm] {last_error}\n"
                          f"      سيُجرَّب المفتاح التالي ({key_index + 2}/{len(keys)}) فوراً.",
                          flush=True)
                break  # التالي: مفتاح جديد، لا فائدة من إعادة محاولة هذا

            last_error = f"{spec['label']} أعاد HTTP {response.status_code}: {body}"
            if response.status_code in _RETRYABLE_STATUS and attempt + 1 < attempts:
                time.sleep(2 ** attempt)
                continue
            raise LLMError(last_error, status_code=response.status_code, backend=name)
        # استُنفدت محاولات هذا المفتاح (أو استُنفد هو نفسه) -> جرّبي التالي

    # استُنفدت كل المفاتيح المتاحة لهذه الواجهة
    if last_status in (401, 403):
        raise MissingAPIKey(
            status_code=last_status, backend=name, message=(
            f"{spec['key_env']}"
            f"{' وما يليه' if len(keys) > 1 else ''} مرفوض من {spec['label']} "
            f"(HTTP {last_status}). إما مفتاح غير صالح (401) أو صلاحياته لا "
            f"تكفي هذا النداء (403). ولّدي مفتاحاً جديداً في .env ثم أعيدي "
            f"تشغيل الخادم — الملف يُقرأ عند الإقلاع مرة واحدة.\n{last_error}")
        )
    raise LLMError(
        last_error or f"فشل استدعاء {spec['label']} بكل المفاتيح المتاحة.",
        status_code=last_status, backend=name)


# أخطاء تستحق تجربة الواجهة الاحتياطية: نفاد الحصة، تعطّل المزوّد، مفتاح تالف،
# نموذج غير متاح، انقطاع الشبكة. أما 400/422 فخلل في الطلب نفسه وسيتكرر عند
# الجميع، فلا معنى لإهدار استدعاء ثانٍ عليه.
_FALLBACK_WORTHY = {401, 402, 403, 404, 408, 409, 425, 429, 500, 502, 503, 504, None}


def fallback_target() -> Optional[tuple]:
    """(الواجهة، النموذج) الاحتياطية، أو None إذا كانت معطّلة أو بلا مفتاح."""
    if not config.LLM_ENABLE_FALLBACK:
        return None
    name = (config.LLM_FALLBACK_BACKEND or "").strip().lower()
    if name not in _BACKENDS or not _BACKENDS[name]["api_key"]():
        return None
    model = (config.LLM_FALLBACK_MODEL or "").strip() or _BACKENDS[name]["default_model"]()
    return name, model


def chat(messages: List[Dict], *, backend: Optional[str] = None,
         model: Optional[str] = None, allow_fallback: bool = True,
         used: Optional[Dict] = None, **kw) -> str:
    """كـ _attempt، لكن ينتقل إلى الواجهة الاحتياطية عند فشل يستحق ذلك.

    ‏used قاموس اختياري يُملأ بما خدم الطلب فعلاً: الواجهة والنموذج وهل جرى
    تحويل. بدونه لا سبيل لمعرفة من ردّ، فيصبح ما تعرضه الواجهة الأمامية تخميناً.
    """
    primary_name, primary_model = resolve(backend, model)
    try:
        text = _attempt(messages, backend=primary_name, model=primary_model, **kw)
        if used is not None:
            used.update({"backend": primary_name, "model": primary_model,
                         "fallback_used": False})
        return text
    except LLMError as primary_error:
        target = fallback_target() if allow_fallback else None
        if target is None or primary_error.status_code not in _FALLBACK_WORTHY:
            raise
        fb_name, fb_model = target
        if (fb_name, fb_model) == (primary_name, primary_model):
            raise
        print(f"[llm] فشلت الواجهة الأساسية {primary_name} "
              f"(HTTP {primary_error.status_code})؛ التحويل إلى "
              f"{fb_name}:{fb_model}.", flush=True)
        try:
            text = _attempt(messages, backend=fb_name, model=fb_model, **kw)
        except LLMError as fb_error:
            # رسالة واحدة تحمل السببين: بدونها يظهر عطل الاحتياطي وحده ويضيع
            # السبب الأصلي (نفاد حصة النموذج الأساسي مثلاً).
            raise LLMError(
                f"فشلت الواجهتان.\n"
                f"  الأساسية ({primary_name}): {primary_error}\n"
                f"  الاحتياطية ({fb_name}): {fb_error}",
                status_code=fb_error.status_code, backend=fb_name) from fb_error
        if used is not None:
            used.update({"backend": fb_name, "model": fb_model, "fallback_used": True,
                         "primary_error": str(primary_error)[:200],
                         "primary_backend": primary_name})
        return text


def chat_json(messages: List[Dict], **kw) -> Optional[dict]:
    """مثل chat لكن يعيد dict مُحلَّلاً، أو None إذا لم يُعد النموذج JSON صالحاً.

    تُمرَّر أخطاء المفاتيح كما هي (MissingAPIKey) لأنها خطأ إعداد لا خطأ نموذج،
    وتستحق رمز حالة مختلفاً في الـ route.
    """
    raw = chat(messages, **kw)
    parsed = extract_json(raw)
    if parsed is None:
        preview = (raw or "").strip().replace("\n", " ")[:200]
        print(f"[llm] لم يُعِد النموذج JSON صالحاً. مقتطف: {preview!r}", flush=True)
    return parsed


def ping(backend: Optional[str] = None, model: Optional[str] = None) -> Dict:
    """استدعاء تجريبي قصير للتأكد من صحة المفتاح والنموذج قبل تشغيل الميزة.

    allow_fallback=False عمداً: الهدف اختبار الواجهة المطلوبة تحديداً (هل
    OpenRouter نفسه يعمل؟) لا معرفة أن *واجهة ما* تردّ. بدون هذا، فشل
    OpenRouter الصامت ثم نجاح التحويل التلقائي لـGroq كان يُعاد كـ
    "ok: true, backend: openrouter" رغم أن OpenRouter لم يخدم الطلب فعلياً —
    نتيجة مضلِّلة تماماً لمن يفحص هل مفاتيحه تعمل.
    """
    name, model_id = resolve(backend, model)
    t0 = time.time()
    used: Dict = {}
    try:
        parsed = chat_json(
            [{"role": "user", "content": 'أعد هذا الكائن حرفياً بصيغة JSON فقط: {"ok": true}'}],
            backend=name, model=model_id, max_tokens=32, temperature=0.0, max_retries=2,
            allow_fallback=False, used=used,
        )
        return {"ok": True, "backend": used.get("backend", name),
                "model": used.get("model", model_id),
                "json_parsed": isinstance(parsed, dict), "reply": parsed,
                "took_ms": int((time.time() - t0) * 1000)}
    except LLMError as e:
        return {"ok": False, "backend": name, "model": model_id, "error": str(e),
                "took_ms": int((time.time() - t0) * 1000)}
