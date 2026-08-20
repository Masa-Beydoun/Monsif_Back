"""
الاسترجاع الدلالي للمواد القانونية.

مسار الاسترجاع: بحث هجين dense و sparse ← دمج RRF ← إعادة ترتيب ← عتبة ←
دمج المواد المرتبطة.

يُبنى الفهرس بسكربت مستقل (scripts/build_laws_index.py)، ولا يُحمَّل هنا سوى
الفهرس الجاهز من القرص. وكل بارامتر قابل للتمرير مع كل استدعاء.
"""

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from services import model_registry


class IndexNotBuilt(RuntimeError):
    """الفهرس غير مبني أو ناقص؛ يحوّلها الـ route إلى 503 مع رسالة الإصلاح."""


# تطبيع النص العربي

DIACRITICS_RE = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]"
)
TATWEEL = "ـ"


def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    t = DIACRITICS_RE.sub("", text)
    t = t.replace(TATWEEL, "")
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي")
    t = t.replace("ة", "ه")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_scrape_artifacts(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"<[^>]+>", " ", text)
    t = t.replace("&nbsp;", " ").replace("&amp;", "&")
    t = t.replace("�", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# نماذج البيانات

@dataclass
class Article:
    """مادة قانونية موحّدة.

    الملف المُدخل (articles_unified*.jsonl) مُطبَّع مسبقاً، فلا حاجة لإعادة التحقق.
    """
    article_id: str
    law_id: str
    law_name: str
    article_number: str
    status: str
    body_raw: str
    body_normalized: str
    short_name: Optional[str] = None
    category: Optional[str] = None
    law_category_raw: Optional[str] = None
    hierarchy: List[Dict] = field(default_factory=list)
    references: List[Dict] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dependency_ids: List[str] = field(default_factory=list)


@dataclass
class DependencyArticle:
    """مادة مدموجة، سُحبت لأن مادة مسترجَعة تعتمد عليها."""
    article_id: str
    law_name: str
    article_number: str
    status: str
    body: str
    depth: int      # 1 = تابعة مباشرة للنتيجة، 2 = تابعة لتابعة
    via: str        # رقم المادة التي استدعتها

    def to_dict(self) -> Dict:
        return {
            "article_id": self.article_id, "law_name": self.law_name,
            "article_number": self.article_number, "status": self.status,
            "body": self.body, "depth": self.depth, "via": self.via,
        }


@dataclass
class SearchResult:
    article_id: str
    law_name: str
    article_number: str
    status: str
    score: float
    body: str
    dependencies: List[DependencyArticle] = field(default_factory=list)

    @property
    def referenced(self):       # توافق مع التسمية السابقة
        return self.dependencies

    def to_dict(self) -> Dict:
        return {
            "article_id": self.article_id,
            "law_name": self.law_name,
            "article_number": self.article_number,
            "status": self.status,
            "score": round(float(self.score), 4),
            "similarity_score": round(float(self.score) * 100, 2),
            "body": self.body,
            "dependencies": [d.to_dict() for d in self.dependencies],
        }


def load_corpus(path=None) -> List[Article]:
    """قراءة articles_unified*.jsonl → قائمة Article."""
    path = path or config.LAWS_CORPUS_FILE
    articles: List[Article] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            law_id = str(r.get("law_id") or r.get("law_name") or "unknown")
            art_no = str(r.get("article_number") or "")
            art_id = str(r.get("article_id") or f"{law_id}:{art_no}")
            if art_id in seen:
                continue
            body = clean_scrape_artifacts(str(r.get("body_raw") or r.get("body") or ""))
            if not body:
                continue                    # عناوين أقسام أو فجوات استخلاص؛ لا تُفهرس
            seen.add(art_id)
            articles.append(Article(
                article_id=art_id,
                law_id=law_id,
                law_name=str(r.get("law_name") or ""),
                short_name=r.get("short_name"),
                category=r.get("category"),
                article_number=art_no,
                status=str(r.get("status") or "عادية"),
                body_raw=body,
                body_normalized=r.get("body_normalized") or normalize_arabic(body),
                law_category_raw=r.get("law_category_raw") or r.get("law_category"),
                hierarchy=r.get("hierarchy") or [],
                references=r.get("references") or [],
                dependencies=[str(x).strip() for x in (r.get("dependencies") or []) if str(x).strip()],
            ))
    return articles


# رسم الارتباطات بين المواد

def build_dependency_graph(articles: List[Article]) -> Dict[str, List[str]]:
    """تحويل أرقام المواد المذكورة داخل كل مادة إلى article_id فعلية في المجموعة.

    لا يستعمل نموذجاً لغوياً؛ العملية لحظية وحتمية.
    """
    num_index: Dict[tuple, str] = {}
    for a in articles:
        num_index.setdefault((a.law_id, normalize_arabic(a.article_number).strip()), a.article_id)
    valid_ids = {a.article_id for a in articles}

    def resolve(law_id: str, target: str) -> Optional[str]:
        t = str(target or "").strip()
        if not t:
            return None
        if t in valid_ids:
            return t
        if f"{law_id}:{t}" in valid_ids:
            return f"{law_id}:{t}"
        key = normalize_arabic(t.rsplit(":", 1)[-1]).strip()
        key = re.sub(r"^\s*(?:الماده|الماده\s*رقم|ماده)\s*", "", key).strip()
        return num_index.get((law_id, key))

    def targets_of(a: Article) -> List[str]:
        out, seen = [], set()
        for t in a.dependencies:
            t = str(t).strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        for r in a.references:
            if isinstance(r, dict):
                if r.get("resolved") is False:
                    continue
                t = str(r.get("target") or "").strip()
            else:
                t = str(r).strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    graph: Dict[str, List[str]] = {}
    for a in articles:
        resolved_ids = []
        for t in targets_of(a):
            rid = resolve(a.law_id, t)
            if rid is None or rid == a.article_id or rid in resolved_ids:
                continue
            resolved_ids.append(rid)
        a.dependency_ids = resolved_ids
        if resolved_ids:
            graph[a.article_id] = resolved_ids
    return graph


# محرك الاسترجاع

class LawsRAG:
    """يحمّل فهرس Qdrant الجاهز والمجموعة، ويوفّر search().

    يأخذ Qdrant في الوضع المحلي (embedded) قفلاً حصرياً على المجلد، فلا تفتحه
    إلا عملية واحدة. لذلك يعمل الخادم بلا reloader (انظر app.py).
    """

    def __init__(self):
        self.articles: List[Article] = []
        self.by_id: Dict[str, Article] = {}
        self.graph: Dict[str, List[str]] = {}
        self.client = None
        self._ready = False

    def load(self) -> None:
        from qdrant_client import QdrantClient

        t0 = time.time()
        print(f"[laws] تحميل المجموعة من {config.LAWS_CORPUS_FILE} ...", flush=True)
        self.articles = load_corpus()
        self.by_id = {a.article_id: a for a in self.articles}
        self.graph = build_dependency_graph(self.articles)
        n_edges = sum(len(v) for v in self.graph.values())
        print(f"[laws] {len(self.articles)} مادة | {len(self.graph)} مادة لها ارتباطات "
              f"({n_edges} حافة) — {time.time() - t0:.1f}s", flush=True)

        import os
        if not os.path.exists(config.LAWS_QDRANT_PATH):
            raise IndexNotBuilt(
                f"فهرس Qdrant غير موجود: {config.LAWS_QDRANT_PATH}\n"
                f"نفّذ أولاً:  python scripts/build_laws_index.py"
            )

        # يأخذ Qdrant المحلي قفلاً حصرياً على المجلد. عند فشل التحقق أدناه يجب
        # إغلاق العميل فوراً، وإلا بقي ماسكاً للقفل فمنع سكربت البناء من العمل،
        # وفتح كل طلب جديد عميلاً آخر لأن warmup() لم يكتمل.
        client = QdrantClient(path=config.LAWS_QDRANT_PATH)
        try:
            if not client.collection_exists(config.LAWS_COLLECTION):
                raise IndexNotBuilt(
                    f"مجموعة «{config.LAWS_COLLECTION}» غير موجودة داخل الفهرس "
                    f"(بناء ناقص أو منقطع). أعد البناء:\n"
                    f"    python scripts/build_laws_index.py --force"
                )
            n = client.count(config.LAWS_COLLECTION).count
            if n == 0:
                raise IndexNotBuilt(
                    "فهرس المواد القانونية فارغ (0 نقطة). أعد البناء:\n"
                    "    python scripts/build_laws_index.py --force"
                )
        except BaseException:
            try:
                client.close()
            except Exception:
                pass
            raise

        self.client = client
        print(f"[laws] فهرس Qdrant جاهز: {n} نقطة", flush=True)
        self._ready = True

    # دمج المواد المرتبطة

    def _sort_key(self, article_id: str):
        a = self.by_id.get(article_id)
        num = re.sub(r"\D", "", a.article_number) if a else ""
        return (int(num) if num else 10 ** 9, article_id)

    def collect_dependencies(self, article_id: str, exclude_repealed: bool = False,
                             max_depth: int = 2, max_total: int = 12,
                             skip_ids: frozenset = frozenset()) -> List[DependencyArticle]:
        """مشي عرضي (breadth-first) في رسم الارتباطات انطلاقاً من مادة واحدة.

        النتيجة مرتّبة بالعمق ثم برقم المادة، منزوعة التكرار وبلا المادة نفسها.
        يستبعد exclude_repealed المواد الملغاة من النتيجة، لكن المشي يستمر عبرها
        كي لا تضيع مادة سارية واقعة خلف مادة ملغاة.
        """
        if max_depth < 1 or article_id not in self.by_id:
            return []
        seen = {article_id} | set(skip_ids)
        frontier, out = [article_id], []
        for depth in range(1, max_depth + 1):
            next_frontier, level = [], []
            for src in frontier:
                src_num = self.by_id[src].article_number
                for tid in self.graph.get(src, []):
                    if tid in seen:
                        continue
                    seen.add(tid)
                    dep = self.by_id.get(tid)
                    if dep is None:
                        continue
                    next_frontier.append(tid)
                    if exclude_repealed and dep.status == "ملغاة":
                        continue
                    level.append(DependencyArticle(
                        article_id=dep.article_id, law_name=dep.law_name,
                        article_number=dep.article_number, status=dep.status,
                        body=dep.body_raw, depth=depth, via=src_num,
                    ))
            out.extend(sorted(level, key=lambda d: self._sort_key(d.article_id)))
            if len(out) >= max_total:
                return out[:max_total]
            frontier = next_frontier
            if not frontier:
                break
        return out[:max_total]

    def attach_dependencies(self, results: List[SearchResult], exclude_repealed: bool = False,
                            max_depth: int = 2, max_total: int = 12,
                            dedupe_vs_hits: bool = True) -> List[SearchResult]:
        hit_ids = frozenset(r.article_id for r in results) if dedupe_vs_hits else frozenset()
        for r in results:
            r.dependencies = self.collect_dependencies(
                r.article_id, exclude_repealed=exclude_repealed, max_depth=max_depth,
                max_total=max_total, skip_ids=hit_ids - {r.article_id},
            )
        return results

    # البحث

    def hybrid_search(self, query: str, top_k: int = 15, exclude_repealed: bool = False):
        from qdrant_client import models

        embedder = model_registry.get_bgem3_flag()
        q = normalize_arabic(query)
        enc = embedder.encode([q], return_dense=True, return_sparse=True,
                              return_colbert_vecs=False)
        dense = enc["dense_vecs"][0].tolist()
        lw = enc["lexical_weights"][0]
        sparse = models.SparseVector(indices=[int(k) for k in lw],
                                     values=[float(v) for v in lw.values()])

        flt = None
        if exclude_repealed:
            flt = models.Filter(must_not=[
                models.FieldCondition(key="status", match=models.MatchValue(value="ملغاة"))
            ])

        res = self.client.query_points(
            config.LAWS_COLLECTION,
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=top_k, filter=flt),
                models.Prefetch(query=sparse, using="sparse", limit=top_k, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return res.points

    def rerank(self, query: str, points, top_n: int = 7,
               max_length: int = 512) -> List[SearchResult]:
        """تسجيل النتائج وقصّها. تُضاف المواد المرتبطة في مرحلة لاحقة."""
        if not points:
            return []
        reranker = model_registry.get_reranker()
        pairs = [[query, p.payload["body_raw"]] for p in points]
        scores = reranker.compute_score(pairs, normalize=True, max_length=max_length)
        ranked = sorted(zip(points, scores), key=lambda x: -x[1])[:top_n]
        return [SearchResult(p.payload["article_id"], p.payload["law_name"],
                             p.payload["article_number"], p.payload["status"],
                             float(s), p.payload["body_raw"])
                for p, s in ranked]

    def _run_pipeline(self, query: str, p: Dict):
        """مسار الاسترجاع: تفكيك ← بحث هجين ← إزالة تكرار ← إعادة ترتيب ←
        عتبة ← دمج المواد المرتبطة. يشترك فيه search() و search_raw()."""
        issues = decompose_query(query, p) if p["decompose"] else [query]

        seen, pooled = set(), []
        for issue in issues:
            for pt in self.hybrid_search(issue, top_k=p["hybrid_top_k"],
                                         exclude_repealed=p["exclude_repealed"]):
                aid = pt.payload["article_id"]
                if aid not in seen:
                    seen.add(aid)
                    pooled.append(pt)

        results = self.rerank(query, pooled, top_n=p["top_n"],
                              max_length=p["rerank_max_length"])
        results = [r for r in results if r.score >= p["min_score"]]

        if results and p["with_dependencies"]:
            results = self.attach_dependencies(
                results, exclude_repealed=p["exclude_repealed"],
                max_depth=p["dep_depth"], max_total=p["dep_max"],
                dedupe_vs_hits=p["dep_dedupe_vs_hits"],
            )
        return results, issues

    def _params(self, kw: Dict) -> Dict:
        """القيم الافتراضية من config، وتعلوها أي قيمة أرسلها الطلب."""
        if not self._ready:
            raise RuntimeError("فهرس المواد القانونية غير محمّل.")
        p = dict(config.LAWS_DEFAULTS)
        p.update({k: v for k, v in kw.items() if v is not None})
        return p

    def search(self, query: str, **kw) -> Dict:
        """يعيد قاموساً جاهزاً لاستجابة JSON.

        جميع البارامترات اختيارية وتأخذ قيمها الافتراضية من config.LAWS_DEFAULTS.
        """
        p = self._params(kw)
        if not query or not query.strip():
            return {"query": query, "error": "الاستعلام فارغ.", "results": []}

        t0 = time.time()
        results, issues = self._run_pipeline(query, p)

        return {
            "query": query,
            "issues": issues if len(issues) > 1 else None,
            "found": bool(results),
            "count": len(results),
            "results": [r.to_dict() for r in results],
            "message": None if results else "لا توجد مواد قانونية تتجاوز عتبة التشابه المطلوبة.",
            "params_used": p,
            "took_ms": int((time.time() - t0) * 1000),
        }

    def search_raw(self, query: str, **kw) -> List[SearchResult]:
        """مثل search() لكنه يعيد كائنات SearchResult؛ يستعمله محرك الحكم الأولي."""
        results, _ = self._run_pipeline(query, self._params(kw))
        return results


# تفكيك الاستعلامات الطويلة (اختياري، يتطلب Gemini)

def decompose_query(text: str, params: Dict) -> List[str]:
    """تفكيك واقعة طويلة إلى مسائل قانونية منفصلة. يعيد [text] عند غياب المفتاح."""
    if len(text.split()) < params.get("decompose_min_words", 25) or not config.GOOGLE_API_KEY:
        return [text]
    try:
        from google import genai

        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        prompt = (
            "حلّل الواقعة التالية واستخرج المسائل القانونية المنفصلة التي تحتاج للبحث "
            "في النصوص القانونية السورية. أعد مصفوفة JSON من جمل بحث قصيرة فقط.\n\n" + text
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.0,
                    "max_output_tokens": 1024},
        )
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", (resp.text or "").strip()).strip()
        issues = json.loads(txt) if txt else None
        if isinstance(issues, dict):
            issues = next((v for v in issues.values() if isinstance(v, list)), None)
        if isinstance(issues, list):
            issues = [str(i).strip() for i in issues if str(i).strip()]
            return issues or [text]
    except Exception as e:
        print(f"[laws] تعذّر تفكيك الاستعلام ({e}) — سيُبحث كاستعلام واحد.", flush=True)
    return [text]


# واجهة الاستخدام

_instance: Optional[LawsRAG] = None
_init_lock = threading.Lock()


def warmup() -> None:
    """تحميل المجموعة والفهرس. يُستدعى تلقائياً عند أول طلب، أو عند الإقلاع مع WARMUP=laws."""
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                rag = LawsRAG()
                rag.load()
                _instance = rag


def get_rag() -> LawsRAG:
    if _instance is None:
        warmup()
    return _instance


def search_laws(query: str, **kw) -> Dict:
    """الدالة الوحيدة التي يحتاجها الـ route."""
    return get_rag().search(query, **kw)


def is_loaded() -> bool:
    return _instance is not None
