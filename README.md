# Monsif_Back — الواجهة الخلفية للنظام القضائي الذكي

Flask API فيه ميزات مستقلة، كل وحدة قابلة للاستدعاء لحالها:

| الميزة | المسار | المصدر |
|---|---|---|
| تلخيص نص قضية واستخراج حقول | `POST /api/legal/summarize` | موجودة أصلاً |
| بحث المواد + الاجتهادات (النظام القديم) | `POST /api/legal/search` | موجودة أصلاً |
| **RAG المواد القانونية** | `POST /api/legal/laws/search` | `predection.ipynb` — Part A |
| **RAG السوابق القضائية** | `POST /api/legal/cases/search` | `predection.ipynb` — Part B |
| **إصدار حكم أولي** | `POST /api/legal/judgment/predict` | `issuing_preliminary_ruling.ipynb` — الخلايا الثلاث الأخيرة |
| **RAG نماذج العقود** | `POST /api/legal/contracts/search` | `contract_RAG.ipynb` — Part D |

ميزة الحكم الأولي **بتستدعي الميزتين التانتين داخلياً** — بتجمع المواد + السوابق
كسياق، وبتمرّرهن للـ LLM، وبعدين بتتحقق إنو كل استشهاد موجود فعلاً بالسياق.

### الواجهة الاحتياطية للنموذج اللغوي

في واجهة أساسية وواجهة احتياطية. لما تفشل الأساسية فشلاً يستحق التحويل — نفاد
حصة (402)، تجاوز معدل (429)، تعطّل مزوّد (5xx)، مفتاح تالف (401/403)، نموذج غير
مخدوم، أو انقطاع شبكة — بينتقل الطلب تلقائياً للاحتياطية **مرة وحدة**. أخطاء
الطلب نفسه (400/422) ما بتتحوّل لأنها رح تتكرر عند أي مزوّد.

```
LLM_BACKEND=hf                 # الأساسية
LLM_FALLBACK_BACKEND=openrouter
LLM_FALLBACK_MODEL=meta-llama/llama-3.3-70b-instruct
LLM_ENABLE_FALLBACK=true       # false = تعطيل التحويل
```

الواجهات المدعومة: `hf` · `openrouter` · `gemini` · `groq` — كلهن متوافقات مع
OpenAI وبينضافوا من قاموس `_BACKENDS` بـ[llm_client.py](services/llm_client.py).

**⚠️ Groq محجوب من شبكة المشروع.** `api.groq.com` بيرجع 403 «Access denied»
لأي طلب — بمفتاح صحيح، بمفتاح مزيّف، وبلا مفتاح إطلاقاً. يعني حجب على مستوى
الشبكة قبل ما يقرأ المفتاح، مش مشكلة صلاحيات. مقيس كمان: HuggingFace و
OpenRouter و Gemini و Together و Cohere و Mistral كلهن بيوصلوا طبيعي.

**معرفة مين خدم الطلب فعلاً.** `meta.llm` بيفرّق بين المخطَّط والفعلي:

```jsonc
"llm": {
  "backend": "hf",                  // المخطَّط
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "fallback_used": true,            // صار تحويل
  "calls": 2,
  "served_by": [                    // مين خدم كل مرحلة فعلاً
    {"stage": "statute_discrimination", "backend": "openrouter", "model": "..."},
    {"stage": "judgment", "backend": "openrouter", "model": "..."}
  ]
}
```

النموذج اللغوي بينستدعى كـ **API** عبر HuggingFace Inference Providers
(`meta-llama/Llama-3.3-70B-Instruct` افتراضياً) — ما في تنزيل أوزان ولا GPU.
كل يلي بده الأمر: `HF_TOKEN` بملف `.env` (نوع **Read**).

**ليش 70B مش 8B.** جُرّبا الاتنين على نفس القضية: 70B بيجتاز التحقق كاملاً
(`fully_grounded=true`) وبيختار المادة **656** الصحيحة خلافاً للـ657 الأقرب لفظياً،
وبياخد ~60 ثانية. أما 8B فبيقتبس المادة كاملة (58 كلمة بدل 12)، وبيكتب `article_id`
ناقص، وبيختار 657 رغم إنو كل السوابق المرفقة بتقول 656 — فبيسقط `fully_grounded`
وبياخد ~140 ثانية. `Llama-3.1-70B` مش مخدوم على مزوّدي HuggingFace.

---

## بنية الملفات

```
app.py                    تسجيل الـ blueprints + /api/health + منطق التسخين
config.py                 ★ كل البارامترات القابلة للتعديل بمكان واحد

routes/                   طبقة HTTP رفيعة — تحقق من المدخلات ثم استدعاء الخدمة
  summarization_routes.py
  law_and_jurisprudence_search_routes.py
  laws_rag_routes.py          POST /laws/search   · GET /laws/config
  cases_rag_routes.py         POST /cases/search  · GET /cases/config
  judgment_routes.py          POST /judgment/predict · GET /judgment/config
                              · GET /judgment/llm/ping
  contracts_routes.py         POST /contracts/search · POST /contracts/get

services/                 كل المنطق — بدون أي استيراد لـ Flask
  model_registry.py       ★ نماذج مشتركة، تحميل كسول (أهم ملف للسرعة)
  summarization.py
  law_and_jurisprudence_search.py
  laws_rag.py             Part A — hybrid → RRF → rerank → دمج المواد المرتبطة
  cases_rag.py            Part B — FAISS + BM25 → RRF → rerank → عتبة
  judgment.py             Part C — بيعتمد على laws_rag و cases_rag
  llm_client.py           ★ استدعاء النموذج اللغوي (hf | groq) بواجهة موحّدة
  contracts_rag.py        Part D — FAISS cosine → مرشحين → اقتراح LLM

scripts/
  download_models.py      تنزيل أوزان النماذج مسبقاً (مع إعادة محاولة تلقائية)
  build_laws_index.py     ★ بناء فهرس Qdrant — مرة وحدة بس
  build_cases_index.py    بناء فهرس القضايا — مرة وحدة (فيه فهرس جاهز أصلاً)
  build_contracts_index.py  بناء فهرس نماذج العقود — مرة وحدة
  smoke_test.py           اختبار كل المسارات (بده السيرفر شغّال)
  test_judgment_logic.py  اختبار منطق التحقق — بثانية، بدون نماذج ولا LLM

data/
  laws/articles_unified.jsonl      2278 مادة + رسم الارتباطات
  laws/qdrant_db/                  ← بينبنى بالسكربت (مستثنى من git)
  cases/legal_facts_hybrid.faiss   فهرس القضايا الجاهز
  cases/legal_metadata_hybrid.pkl
  cases/standard_cases.json        المصدر (للبناء من الصفر)
  classifier/                      ← انسخي ملفات joblib هون (اختياري)
  contracts/hammurabi_templates_flat.json   ← انسخي ملف النماذج هون
  contracts/contracts.faiss        ← بينبنى بالسكربت
  contracts/contracts_meta.pkl

utils/                    فهارس النظام القديم (.faiss / .pkl)
```

---

## التشغيل أول مرة

```bash
cd r:/5th_year/GP/back/Monsif_Back

# 1) البيئة الافتراضية (Python 3.11 — أفضل توافق مع faiss/torch)
"C:/Users/Rony/.pyenv/pyenv-win/versions/3.11.9/python.exe" -m venv venv
./venv/Scripts/python.exe -m pip install -r requirements.txt

# 2) المفاتيح
cp .env.example .env
#    وحطّي HF_TOKEN جوّاته (مطلوب لميزة الحكم الأولي بس)
#    الرمز من https://huggingface.co/settings/tokens
#    ولازم حسابك يكون موافق على شروط مستودع النموذج:
#    https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# 3) تنزيل أوزان النماذج (~4.6GB) — مرة وحدة
./venv/Scripts/python.exe scripts/download_models.py

# 4) بناء فهرس المواد القانونية — مرة وحدة، وبعدها ما بتعيديه أبداً
./venv/Scripts/python.exe scripts/build_laws_index.py

# 5) (لميزة العقود) حطّي hammurabi_templates_flat.json بـ data/contracts/ وابني الفهرس
./venv/Scripts/python.exe scripts/build_contracts_index.py
```

**الخطوة 3** بتنزّل BGE-M3 (2.27GB) والـ reranker (2.27GB) للكاش المحلي
`.hf_cache/`. حسب سرعة نتك ممكن تاخد ساعة أو أكتر. إذا انقطع النت بنص
التنزيل، شغّلي السكربت مرة تانية — **بيكمّل من وين وقف مش من الصفر**.

**الخطوة 4** هي الشي الغالي الوحيد: بترمّز 2278 مادة لمتجهات dense + sparse.
على CPU بتاخد **20–60 دقيقة**. بعدها بينحفظ الفهرس على القرص وما بينبنى مرة تانية.

> جرّبي أول شي على عيّنة صغيرة حتى تتأكدي إنو كل شي ماشي:
> ```bash
> ./venv/Scripts/python.exe scripts/build_laws_index.py --limit 50
> ```
> وبعدين للبناء الكامل:
> ```bash
> ./venv/Scripts/python.exe scripts/build_laws_index.py --force
> ```

---

## التشغيل اليومي

```bash
./venv/Scripts/python.exe app.py
```

السيرفر بيقلع **بثانية وحدة**. أول طلب لكل ميزة بيحمّل نماذجها (30–90 ثانية على CPU)،
وبعدها كل الطلبات سريعة.

---

## ليش ما عاد يستنى كل مرة

| قبل | بعد |
|---|---|
| كل ميزة تحمّل نسخة BGE-M3 خاصة فيها + reranker خاص = 6 تحميلات لنفس الأوزان | `model_registry` بيحمّل نموذجين بس، مشتركين بين كل الميزات |
| `warmup()` بينستدعى وقت الـ import → السيرفر ما بيقلع قبل ما يخلص | تحميل كسول: أول طلب لكل ميزة بيحمّل نماذجها. ميزة ما بتستعمليها ما بتكلفك شي |
| `debug=True` مع الـ reloader → كل شي بينحمّل **مرتين** | `use_reloader=False` |
| بناء الفهرس بينحصل بالنوتبوك بنفس الجلسة | انفصل لـ `scripts/build_laws_index.py` — بينشتغل مرة وبينحفظ على القرص |
| أوزان HuggingFace بتنزّل كل جلسة Colab جديدة | `HF_HOME=.hf_cache` — بتنزل مرة وحدة محلياً |
| `hybrid_top_k=30` (مضبوطة لـ GPU) | 15 افتراضياً — نص عدد تمريرات الـ cross-encoder على CPU |

### إذا بدك أول طلب يكون سريع كمان

حطّي بملف `.env`:
```
WARMUP=laws,cases
```
عندها الإقلاع بياخد دقيقة تقريباً، بس أول طلب بيرجع فوراً. مفيد قبل عرض/مناقشة.

---

## البارامترات المكشوفة

كل بارامتر بيتغيّر بثلاث طرق، الأقوى بتغلب:

1. القيمة الافتراضية بـ [config.py](config.py)
2. متغير بيئة بملف `.env` (مثال `LAWS_TOP_N=7`)
3. حقل بجسم الطلب نفسه — بيأثر على هالطلب بس

لعرض القيم الحالية:
```bash
curl http://127.0.0.1:5000/api/legal/laws/config
curl http://127.0.0.1:5000/api/legal/cases/config
curl http://127.0.0.1:5000/api/legal/judgment/config
curl http://127.0.0.1:5000/api/legal/contracts/config
```

### RAG المواد القانونية — `POST /api/legal/laws/search`

| بارامتر | افتراضي | شو بيعمل |
|---|---|---|
| `top_n` | 7 | كم مادة ترجع بالنتيجة النهائية |
| `hybrid_top_k` | 15 | **أهم بارامتر للسرعة** — كم مرشّح يدخل لإعادة الترتيب. النوتبوك كان 30 (على GPU) |
| `exclude_repealed` | false | استبعاد المواد الملغاة |
| `min_score` | 0.0 | عتبة الامتناع — نتيجة تحتها بتنرمى |
| `with_dependencies` | true | دمج المواد المرتبطة بكل نتيجة |
| `dep_depth` | 2 | 1 = المواد المرتبطة مباشرة، 2 = + المرتبطة فيهن |
| `dep_max` | 12 | سقف المواد المدموجة لكل نتيجة |
| `decompose` | false | تفكيك الوقائع الطويلة لمسائل منفصلة (بده `GOOGLE_API_KEY`) |
| `rerank_max_length` | 512 | نافذة الـ cross-encoder — أصغر = أسرع |

### RAG السوابق القضائية — `POST /api/legal/cases/search`

| بارامتر | افتراضي | شو بيعمل |
|---|---|---|
| `top_k` | 5 | كم سابقة ترجع |
| `hybrid_top_k` | 15 | عدد المرشحين قبل إعادة الترتيب (السرعة). النوتبوك كان 20 |
| `threshold` | 0.50 | عتبة التشابه 0.0–1.0 |
| `rerank_max_length` | 2048 | نافذة الـ cross-encoder. **النوتبوك كان 8192** — نزّلتها للسرعة على CPU. ارفعيها لـ 8192 إذا وقائعك طويلة وبدك سلوك النوتبوك بالضبط |

### الحكم الأولي — `POST /api/legal/judgment/predict`

بتنبعت جوّا كائن `config`. أهمهن:

| بارامتر | افتراضي | شو بيعمل |
|---|---|---|
| `top_k_laws` / `top_k_cases` | 5 / 3 | حجم السياق المُمرَّر للـ LLM |
| `case_threshold` | 0.45 | عتبة السوابق |
| `use_fact_reorganization` | true | إعادة تنظيم الوقائع قبل الاسترجاع (PLJP) — **استدعاء LLM إضافي** |
| `use_statute_discrimination` | true | تمييز المواد المتشابهة (ADAPT) — **استدعاء LLM لكل زوج** |
| `use_domain_classifier` | true | المصنّف الإحصائي — بينطفي لحاله إذا ملفاته مش موجودة |
| `max_quote_words` | 12 | سقف الاقتباس الداعم لكل تهمة |
| `confusable_overlap_threshold` | 0.15 | حد تشابه المفردات لاعتبار مادتين «متشابهتين لفظياً» |
| `confusable_window_words` | 0 | طول نافذة المقارنة بالكلمات. **0 = المتن كامل** — شوفي الملاحظة تحت |
| `max_confusable_pairs` | 3 | سقف الأزواج المُمرَّرة لطبقة التمييز (استدعاء LLM لكل زوج) |
| `llm_backend` | `hf` | `hf` = HuggingFace Inference Providers · `groq` = Groq |
| `llm_model` | `meta-llama/Llama-3.3-70B-Instruct` | اسم النموذج عند المزوّد |

طفّي `use_fact_reorganization` و `use_statute_discrimination` إذا بدك ردّ أسرع —
بيوفّروا حتى 4 استدعاءات LLM.

### مطابقة النوتبوك

الافتراضيات كلها منسوخة حرفياً من خلية `## 3-ADAPT layer` بنوتبوك
[predection.ipynb](predection.ipynb)، ومثبّتة باختبار في
`scripts/test_judgment_logic.py` (القسم 12) حتى ما تنزاح بالسهو. في تلات مفاتيح
بتفصل سلوك النوتبوك عن تصحيحات مقيسة، وكلهن **مطفيين افتراضياً**:

| المفتاح | الافتراضي (= النوتبوك) | شو بيصير لما تشغّليه |
|---|---|---|
| `confusable_window_words` | `40` | `0` = مقارنة المتن كامل |
| `confusable_scan_cited_articles` | `false` | كشف الأزواج بيمسح كمان مواد السوابق |
| `precedent_key_by_uid` | `false` | مطابقة السوابق بـ `case_uid` بدل `case_number` |
| `strict_citation_law_match` | `false` | تضييق حلّ استشهادات السوابق (تحت) |

**⚠️ نتيجة مقيسة على بيانات هذا المشروع:** بالافتراضيات (سلوك النوتبوك) طبقة
ADAPT **ما بتنتج ولا زوج**، يعني ما في أي استدعاء تمييز والطبقة التالتة بتضل بلا
أثر — بدون ما يظهر أي خطأ. السبب سببين متراكبين:

1. المادة **656 ما بتطلع أبداً** بالاسترجاع الدلالي المباشر (657 بتجي بنتيجة
   0.746 و656 خارج أعلى النتائج)، وبتوصل للسياق حصراً عبر استشهادات السوابق —
   فلما الكشف يمسح قائمة الاسترجاع وحدها ما بيلاقي الطرف التاني.
2. نافذة الـ40 كلمة بتقطع 656 قبل شطر العقوبة، والمفردات المشتركة مع 657 كلها
   بذيل المادة. المقاس: **0.067** بالنافذة مقابل **0.275** على المتن الكامل،
   والعتبة 0.15.

لتشغيل الطبقة فعلياً (مجرَّب: بيختار 656 الصحيحة و`fully_grounded=true`):

```json
{"config": {"confusable_window_words": 0, "confusable_scan_cited_articles": true}}
```

### ⚠️ استشهادات السوابق بتنحلّ لقوانين غلط

المواد يلي بتوصل للسياق عبر `resolve_case_citations` (يعني المذكورة جوّا تسبيب
السوابق) **ما إلها نتيجة تشابه** — لأنها ما انسترجعت دلالياً أصلاً. لهيك بتظهر
بالواجهة بلا نسبة «تطابق». وهاد سليم؛ المشكلة إنو كتير منها **مواد خطأ**:

1. `PENAL_CODE_HINTS = ["العقوبات", ...]` فحص تضمين نصي، و«قانون العقوبات واصول
   المحاكمات العسكرية» بيتضمّن كلمة «العقوبات» — فالمادة 129 بتنحلّ للقانون
   **العسكري** بدل العام. المقاس على 301 قضية: **254 من 645 استشهاد رقمي (39%)
   بتصيب القانون الخطأ**.
2. مسار `other_laws` بلا أي مرشِّح: بياخد أول رقم بالنص، وغالباً هو رقم المرسوم
   لا رقم المادة — «مرسوم العفو العام رقم 13 لعام 2021» بينقرأ كالمادة 13. و163
   مدخل بيمرقوا من هون.

هالمنطق **منقول حرفياً من النوتبوك** (`resolve_case_citations` وحدة من تلات دوال
مطابقة بايت-ببايت)، لهيك تصحيحه ورا مفتاح مطفي افتراضياً:

```json
{"config": {"strict_citation_law_match": true}}
```

المقاس بعد التشغيل: 659 مادة محلولة، **كلها** من قانون العقوبات العام، وصفر من
العسكري أو السير. الوضع الصارم بيُسقط الملتبس بدل ما يخمّنه.

**ليش هاد مهم مش بس شكلياً:** هالمواد بتدخل سياق النموذج كنصوص قانونية شرعية،
وبتنضاف لخريطة `bodies` يلي بتتحقق فيها `verify_charge_quotes` من الاقتباسات —
يعني اقتباس من مادة عسكرية بيعدي التحقق بنجاح.

---

### RAG نماذج العقود — `POST /api/legal/contracts/search`

| بارامتر | افتراضي | شو بيعمل |
|---|---|---|
| `top_k` | 5 | كم نموذج عقد يرجع بقائمة المرشحين |
| `min_score` | 0.0 | عتبة cosine 0.0–1.0. النوتبوك ما كان فيه عتبة |
| `suggest` | true | طبقة الـ LLM يلي بتقترح الأنسب — **استدعاء Groq واحد**. طفّيها لبحث دلالي صرف |
| `groq_model` | `qwen/qwen3.6-27b` | موديل Groq للاقتراح |

الميزة خطوتين، متل الحلقة التفاعلية بالنوتبوك:

```bash
# 1) البحث — بيرجع مرشحين + اقتراح الأنسب
curl -X POST http://127.0.0.1:5000/api/legal/contracts/search   -H "Content-Type: application/json"   -d '{"text": "بدي عقد ايجار محل تجاري", "top_k": 5}'

# 2) عرض العقد كامل بعد ما يختار المستخدم
curl -X POST http://127.0.0.1:5000/api/legal/contracts/get   -H "Content-Type: application/json"   -d '{"doc_id": "..."}'
```

الاقتراح **ما بيختار عن المستخدم** — بيرجع `suggestion.choice` و `reason` بس،
والاختيار النهائي بيصير بالخطوة التانية. إذا الموديل ما لقى مرشح مناسب بيرجع
`choice: null`. إذا `GROQ_API_KEY` مش مضبوط، استعملي `"suggest": false`.

**فروقات عن النوتبوك:** الاقتراح صار عبر Groq بدل تحميل Qwen2.5-7B محلياً،
ونموذج الـ embedding صار BGE-M3 (نفس نموذج باقي المشروع، من `model_registry`)
بدل `multilingual-e5-large` — حتى ما نحمّل نموذج تالت بالذاكرة. لو بدك نموذج
النوتبوك بالضبط: `CONTRACTS_EMBEDDING_MODEL=intfloat/multilingual-e5-large`
بملف `.env` ثم `python scripts/build_contracts_index.py --force`.

---

## الاختبار

### 1) منطق التحقق — سريع، بدون نماذج ولا مفاتيح

```bash
./venv/Scripts/python.exe scripts/test_judgment_logic.py
```

35 فحص بيشتغلوا بثانية: تحقق الاقتباسات (نفس حالات النوتبوك — إعادة صياغة،
اختلاف همزة، مسافات زايدة)، كشف الاستشهادات المختلقة، كشف المواد المتشابهة،
تعارض السوابق، وبناء الـ prompt. شغّليه كل ما تعدّلي `services/judgment.py`.

### 2) المسارات كاملة — بده السيرفر شغّال

بترمينال تاني:

```bash
./venv/Scripts/python.exe scripts/smoke_test.py

# ميزة وحدة
./venv/Scripts/python.exe scripts/smoke_test.py --only laws

# بدون الحكم الأولي (حتى ما تستهلكي حصة Groq)
./venv/Scripts/python.exe scripts/smoke_test.py --skip judgment
```

### طلبات يدوية

> ⚠️ **لا تكتبي عربي مباشرة بسطر أوامر curl على ويندوز.** الكونسول بيحوّله لـ cp1256
> والسيرفر بيرفضه بـ `400 Failed to decode JSON object`. المشكلة بالترمينال مش
> بالـ API. حطّي الطلب بملف واستعملي `--data-binary @`:

```bash
# اكتبي الطلب بملف UTF-8
cat > req.json <<'EOF'
{"text": "السرقة الموصوفة", "top_n": 3, "exclude_repealed": true}
EOF

curl -X POST http://127.0.0.1:5000/api/legal/laws/search \
  -H "Content-Type: application/json" --data-binary @req.json
```

الأسهل: استعملي `scripts/smoke_test.py` فوق، أو Postman / Thunder Client.
المسارات يلي بدون عربي بتشتغل عادي بـ curl:

```bash
curl http://127.0.0.1:5000/api/health
curl http://127.0.0.1:5000/api/legal/laws/config
```

---

## شكل الرد

كل المسارات بترجع نفس الغلاف:

```json
{ "status": "success", "data": { ... } }
{ "status": "error",   "error": "..." }
```

`data` بمسارات البحث فيه `results` و `count` و `took_ms` و **`params_used`** —
هاد الأخير بيوريكي بالضبط أي قيم استُعملت بهالطلب، مفيد لما تجرّبي بارامترات.

### رد الحكم الأولي — العقد مع الواجهة الأمامية

`data` عنده بنية ثابتة موثّقة، مستقلة عن أي تعديل بالـ prompt:

```jsonc
{
  "ok": true,
  "facts_summary": "...",
  "outcome": "إدانة",              // وحدة من: إدانة | براءة | إسقاط دعوى
  "outcome_is_standard": true,      // false = الموديل خرج عن القيم الثلاث، اعرضيه كتحذير
  "final_charge": "...",
  "candidate_charges": [{
    "charge": "...", "article_id": "...", "article_number": "656",
    "law_name": "...", "supporting_quote": "...", "elements_match": true,
    "flagged": false, "flag_reason": null   // لوّني منهن مباشرة، بلا ما تقرّي كتلة التحقق
  }],
  "reasoning": { "charges_analysis": "...", "exclusion_notes": "...",
                 "precedent_alignment": "..." },
  "cited_statutes": [...], "cited_precedents": [...],
  "suggested_penalty_range": "...", "confidence_note": "...",
  "verification": {
    "fully_grounded": false,
    "label": "...",                 // جملة عربية جاهزة للعرض بالشارة
    "unknown_statute_citations": [], "unknown_case_citations": [],
    "possible_penalty_hallucination_in": [],
    "unsupported_charge_quotes": [], "precedent_article_mismatches": []
  },
  "evidence": {                     // السياق الكامل يلي شافه الموديل
    "statutes": [...], "cases": [...], "classifier_candidates": [...],
    "reorganized_facts": {...}, "confusable_pairs": [...],
    "discrimination": [...], "retrieval_text": "..."
  },
  "raw_model_output": {...},        // مخرَج الموديل الخام كما هو
  "meta": { "took_ms": 0, "llm": {"backend": "hf", "model": "..."},
            "pipeline": {...}, "config_used": {...} }
}
```

اقري `GET /judgment/config` عند فتح الشاشة: `data.ready = false` معناها المفتاح
ناقص، و `data.blocking_reason` جملة عربية جاهزة للعرض. و `GET /judgment/llm/ping`
بيجرّب النموذج بتوكنات معدودة — أرخص بكتير من انتظار المسار كامل لتكتشفي إنو
الرمز غلط.

---

## حل المشاكل

| العرض | السبب والحل |
|---|---|
| `503` + «فهرس Qdrant غير موجود» | ما بنيتي الفهرس. شغّلي `python scripts/build_laws_index.py` |
| `503` + «مجموعة … غير موجودة (بناء ناقص)» | البناء انقطع بنص الطريق. `python scripts/build_laws_index.py --force` |
| `503` + «HF_TOKEN غير مضبوط» | حطّي الرمز بملف `.env`. جرّبيه بـ `GET /api/legal/judgment/llm/ping` |
| `502` + «رفض المفتاح (HTTP 401/403)» | الرمز غلط، أو حسابك ما وافق على شروط مستودع Llama. افتحي صفحة النموذج على HuggingFace واقبلي الشروط |
| `502` + «النموذج … غير متاح (HTTP 404)» | المزوّد ما بيخدم هالاسم. غيّري `HF_MODEL` بـ `.env` أو `llm_model` بالطلب |
| الحكم الأولي بطيء جداً | كل استدعاء LLM بياخد ثواني. طفّي `use_fact_reorganization` و `use_statute_discrimination` |
| `ChunkedEncodingError` وقت التنزيل | النت انقطع. أعيدي `python scripts/download_models.py` — بيكمّل من وين وقف |
| `UnicodeEncodeError: 'charmap' codec` | كونسول ويندوز cp1256. كل سكربتات المشروع بتصلّحها لحالها؛ إذا ظهرت بسكربت جديد ضيفي `sys.stdout.reconfigure(encoding="utf-8")` |
| `400 Failed to decode JSON object` | كتبتي عربي بسطر أوامر curl — شوفي قسم «طلبات يدوية» فوق |
| `Storage folder … is already accessed by another instance` | السيرفر شغّال وأنتِ عم تبني الفهرس (أو نسختين سيرفر). طفّي السيرفر أول شي |
| أول طلب بطيء جداً | طبيعي — تحميل كسول. حطّي `WARMUP=laws,cases` بـ `.env` إذا بدك الإقلاع يتحمّلها بدل أول طلب |
| البحث بطيء بكل طلب | نزّلي `hybrid_top_k` (15 → 10) و `rerank_max_length` (512 → 256) |

## ملاحظات مهمة

**مفاتيح API المكشوفة.** النوتبوك فيه مفتاحين Groq مكتوبين حرفياً بالكود
(`gsk_BYoAU...` و `gsk_0G7uJ...`). هون انتقلوا لـ `.env` وما بينرفعوا على git —
بس **لازم تلغيهن من لوحة Groq**، لأنهم موجودين بتاريخ النوتبوك. المشروع صار
يستعمل HuggingFace افتراضياً فما عاد بده مفتاح Groq إطلاقاً.

**تبديل مزوّد النموذج اللغوي.** كل الاستدعاءات بتمرق من
[services/llm_client.py](services/llm_client.py) بواجهة متوافقة مع OpenAI.
لتبديل المزوّد: `LLM_BACKEND=groq` بـ `.env`، أو `"llm_backend": "groq"` بجسم
الطلب لهالطلب وحده. لإضافة مزوّد جديد: ضيفي مدخل بقاموس `_BACKENDS`، وبس.

**`case_number` مش مفتاح فريد.** بمجموعة السوابق (301 صف) في 48 رقم بيتكرر
مرتين، و**47 منهن قضايا مختلفة فعلاً** بتتصادم بالرقم — مثلاً 510 هي «دخول غير
مشروع» بصف و«شيك بلا رصيد» بصف تاني، تشابه وقائعهن 12% بس. لهيك:

- ما تستعملي `case_number` كمفتاح صف بالواجهة ولا لإزالة التكرار — بيحذف قضايا
  حقيقية. استعملي **`case_uid`** (مشتق من `file_name`، فريد 301/301).
- التكرار الحقيقي بينكشف بالمحتوى: تجزئة `facts_text` + `judgment_text` +
  `claims_text`. بتطوي صفّين بس من أصل 301 — القضية **579** (مُدخَلة مرتين،
  الفرق أثر OCR بالتسبيب) والقضيتين **730/731** (نفس المحتوى برقمين).
- الطيّ بيصير **وقت الاستعلام** قبل إعادة الترتيب، فما بده إعادة بناء الفهرس
  والبيانات الأصلية بتضل كما هي. إذا بدك تنضّفي المصدر نهائياً، نضّفي
  `data/cases/standard_cases.json` وأعيدي `python scripts/build_cases_index.py`.

**المصنّف الإحصائي مش موجود محلياً.** ملفات `judicial_classifier.joblib` و
`word_vectorizer.joblib` و `char_vectorizer.joblib` و `label_binarizer.joblib`
موجودة على Drive بس. انسخيهن لـ `data/classifier/` وبينشتغل تلقائياً. بدونهن
الحكم الأولي بيشتغل عادي، بس بدون طبقة التهم المرشحة من المصنّف.

**Qdrant المحلي بياخد قفل حصري.** يعني: ما تشغّلي `build_laws_index.py` والسيرفر
شغّال، وما تشغّلي نسختين من السيرفر. لهيك `use_reloader=False` بـ `app.py` —
لا تشيليها.

**ما في GPU على هالجهاز.** كل شي مضبوط على CPU (`fp16` بينطفي لحاله). البحث
بياخد ثواني بدل أجزاء من الثانية. إذا صار في GPU، حطّي `DEVICE=cuda` بـ `.env`
وارفعي `LAWS_HYBRID_TOP_K` لـ 30 لترجعي لسلوك النوتبوك بالضبط.

**استهلاك الذاكرة.** لما تشتغل الميزات الثلاثة سوا: BGE-M3 مرتين (نسخة
SentenceTransformer للقضايا، ونسخة FlagEmbedding للمواد لأنها بدها المتجهات
الـ sparse كمان) + reranker وحدة = حوالي 6GB RAM.

**فروقات مقصودة عن النوتبوك.** كلها للسرعة على CPU، وكلها قابلة للإرجاع:

| القيمة | النوتبوك | هون | ليش |
|---|---|---|---|
| `hybrid_top_k` (مواد) | 30 | 15 | نص عدد تمريرات الـ cross-encoder |
| `hybrid_top_k` (قضايا) | 20 | 15 | نفس السبب |
| `rerank_max_length` (قضايا) | 8192 | 2048 | نافذة أقصر = أسرع بكتير على CPU |
| `decompose` (تفكيك الاستعلام) | مفعّل لو في مفتاح Gemini | مطفي | بيضيف استدعاء LLM لكل بحث |
| `use_fp16` | `True` | `False` على CPU | fp16 على CPU أبطأ وغير دقيق |

باقي كل شي — منطق الـ RRF، عتبات التشابه، دمج المواد المرتبطة، الـ prompts،
طبقات التحقق الثلاثة — منقول حرفياً.
