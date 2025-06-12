"""
SEO Article Outline Generator
-----------------------------
* Vyhledá TOP 3 výsledky Googlu přes SerpAPI.
* Stáhne jejich HTML, vybere klíčová slova + spočítá jednoduché TF-IDF.
* Vygeneruje jen **osnovu** článku (H-headery, bullet-pointy),
  meta-title, meta-description a návrhy interních odkazů.

Potřebné proměnné prostředí / Streamlit Secrets:
  SERPAPI_API_KEY
  OPENAI_API_KEY
"""

import os
import re
import math
import requests
from collections import Counter, defaultdict

import streamlit as st
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from openai import OpenAI
import tldextract
import nltk
from nltk.stem.snowball import SnowballStemmer

nltk.download("punkt", quiet=True)

# ──────────────────────────────────────────────────────────
# API klíče
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
client = OpenAI()  # načte OPENAI_API_KEY z env

# Stop-slova (CZ + EN základ)
STOP_WORDS = set("""
the a an and or of to in on for with is are be as at by this that from it its
will was were has have had but not your you
a i k o u s v z na že se je jsou by byl byla bylo aby do od po pro pod nad
který která které co to toto tyto ten ta tím tuto tu jako kde kdy jak tak také bez
""".split())

CS_STEM = SnowballStemmer("czech")
EN_STEM = SnowballStemmer("english")

# ──────────────────────────────────────────────────────────
def detect_intent(query: str) -> str:
    q = query.lower()
    trans_kw = {"koupit", "cena", "objednat", "recenze", "srovnání", "discount"}
    if any(w in q for w in trans_kw):
        return "transactional"
    if q.startswith(("jak", "co", "proč", "kdy", "kde", "who", "how", "what", "why")):
        return "informational"
    return "informational"

def fetch_html(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (SEOOutlineBot/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.ok:
            if not r.encoding or r.encoding.lower() == "utf-8":
                r.encoding = r.apparent_encoding or "utf-8"
            return r.text
    except requests.RequestException:
        pass
    return ""

def keyword_frequency(text: str, lang="cz", top_n: int = 20):
    # jen písmena ≥3 znaky
    tokens = re.findall(r"\b[^\W\d_]{3,}\b", text.lower(), flags=re.UNICODE)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    stemmer = CS_STEM if lang == "cz" else EN_STEM
    stems = [stemmer.stem(t) for t in tokens]
    counts = Counter(stems)
    # mapování na reprezentativní originální tvar
    rep = {}
    for w, st in zip(tokens, stems):
        rep.setdefault(st, w)
    return [(rep[s], c) for s, c in counts.most_common(top_n)]

def get_tfidf_keywords(docs, top_n: int = 10):
    N = len(docs)
    df = defaultdict(int)
    all_tokens = []

    for txt in docs:
        tokens = re.findall(r"\b[^\W\d_]{3,}\b", txt.lower(), flags=re.UNICODE)
        tokens = [t for t in tokens if t not in STOP_WORDS]
        unique = set(tokens)
        for t in unique:
            df[t] += 1
        all_tokens += tokens

    idf = {t: math.log(N / (1 + df[t])) for t in df}
    tf = Counter(all_tokens)
    tfidf = {t: tf[t] * idf[t] for t in tf}

    top = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [t for t, _ in top]

def analyse_page(url: str):
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = " ".join(soup.stripped_strings)
    kw = keyword_frequency(text)
    return text[:2000], kw, text  # náhled, klíčová slova, celý text

def search_google(query: str, n: int = 3):
    search = GoogleSearch({
        "engine": "google",
        "q": query,
        "num": n,
        "api_key": SERP_API_KEY,
        "hl": "cs",
    })
    return search.get_dict().get("organic_results", [])[:n]

def propose_outline(query, intent, top_kw, lsi_kw, analyses):
    system = (
        "You are an expert Czech SEO strategist. "
        "Generate ONLY a detailed outline (H1, H2, optional H3) with bullet-point notes, "
        "a meta-title (<=60 char) and meta-description (<=155 char). "
        "Also suggest 3–5 internal links (anchor text + slug). "
        "Do NOT write full paragraphs."
    )
    user = (
        f"Search query: {query}\n"
        f"Search intent: {intent}\n"
        f"Primary keywords: {', '.join(top_kw[:10])}\n"
        f"TF-IDF keywords: {', '.join(lsi_kw)}\n\n"
        "Competitor snapshot:\n"
    )
    for i, a in enumerate(analyses, 1):
        user += f"{i}. {a['url']}\n   KW: {', '.join(a['keywords'][:10])}\n"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=900,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="SEO Outline Generator", page_icon="🔍")
st.title("🔍 SEO Article Outline Generator")

query = st.text_input("Zadej vyhledávací dotaz")
if query:
    if not SERP_API_KEY or not os.getenv("OPENAI_API_KEY"):
        st.error("Chybí API klíče.")
        st.stop()

    intent = detect_intent(query)
    st.caption(f"Detekovaný intent: **{intent}**")
    st.info("⏳ Vyhledávám konkurenci…")

    results = search_google(query)
    if not results:
        st.error("Žádné výsledky.")
        st.stop()

    analyses, docs = [], []
    for res in results:
        url = res.get("link")
        title = res.get("title") or url
        st.subheader(title)
        ext = tldextract.extract(url)
        st.caption(".".join(p for p in [ext.domain, ext.suffix] if p))

        preview, kw, full = analyse_page(url)
        docs.append(full)
        st.markdown("**Top klíčová slova:** " + ", ".join(f"`{w}`" for w, _ in kw))
        with st.expander("Ukázka textu"):
            st.write(preview)

        analyses.append({"url": url, "keywords": [w for w, _ in kw]})

    combined = Counter()
    for a in analyses:
        combined.update(a["keywords"])
    top_kw = [w for w, _ in combined.most_common(40)]
    lsi_kw = get_tfidf_keywords(docs, top_n=12)

    st.info("📝 Generuji osnovu článku…")
    outline = propose_outline(query, intent, top_kw, lsi_kw, analyses)

    st.markdown("---")
    st.subheader("📄 Výstup")
    st.markdown(outline, unsafe_allow_html=True)
    st.success("✅ Osnova vygenerována!")
