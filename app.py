"""
SEO Article Outline Generator (2025-ready)
-----------------------------------------
* Google/SerpAPI vyhledá top 3 URL
* Pro každou:
    – stáhne HTML, detekuje encoding
    – vytáhne top KW + TF-IDF LSI KW
* Vygeneruje **pouze outline** článku:
    H1 → H2 → H3, bullet-pointy, meta-title, meta-description,
    návrhy interních odkazů.
"""

import os
import re
import requests
from collections import Counter, defaultdict

import streamlit as st
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from openai import OpenAI
import tldextract

# NLP & TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.snowball import CzechStemmer, EnglishStemmer

nltk.download("punkt", quiet=True)

# ── API klíče ──────────────
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
client = OpenAI()                     # vezme OPENAI_API_KEY z env

# ── stop-wordy ─────────────
STOP_WORDS = set("""
the a an and or of to in on for with is are be as at by this that from it its
will was were has have had but not your you
a i k o u s v z na že se je jsou by byl byla bylo aby do od po pro pod nad
který která které co to toto tyto ten ta tím tuto tu jako kde kdy jak tak také bez
""".split())

CS_STEM = CzechStemmer()
EN_STEM = EnglishStemmer()

# ── heuristická detekce intentu ─────
def detect_intent(q: str) -> str:
    q = q.lower()
    trans_kw = {"koupit", "cena", "objednat", "recenze", "srovnání", "discount"}
    if any(w in q for w in trans_kw):
        return "transactional"
    if q.startswith(("jak", "co", "proč", "kdy", "kde", "who", "how", "what", "why")):
        return "informational"
    return "informational"

# ── parsování HTML + KW ─────
def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (SEOOutlineBot/1.0; +https://example.com/bot)"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.ok:
            if not r.encoding or r.encoding.lower() == "utf-8":
                r.encoding = r.apparent_encoding or "utf-8"
            return r.text
    except requests.RequestException:
        pass
    return ""

def keyword_frequency(text: str, lang="cz", top_n=20):
    tokens = re.findall(r"\b[^\W\d_]{3,}\b", text.lower(), flags=re.UNICODE)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    stemmer = CS_STEM if lang == "cz" else EN_STEM
    stems = [stemmer.stem(t) for t in tokens]
    counts = defaultdict(int)
    for original, stem in zip(tokens, stems):
        counts[stem] += 1
    # map back to most common original word for each stem
    stem_to_word = {}
    for w in tokens:
        st = stemmer.stem(w)
        if st not in stem_to_word:
            stem_to_word[st] = w
    return [(stem_to_word[s], c) for s, c in Counter(counts).most_common(top_n)]

def get_lsi_keywords(docs, top_n=10):
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words=list(STOP_WORDS))
    X = vectorizer.fit_transform(docs)
    tfidf_scores = X.sum(axis=0).A1
    idx_sorted = tfidf_scores.argsort()[::-1][: top_n]
    feats = vectorizer.get_feature_names_out()
    return [feats[i] for i in idx_sorted]

def analyse_page(url: str):
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.extract()
    text = " ".join(soup.stripped_strings)
    kws = keyword_frequency(text)
    return text[:2000], kws, text

def search_google(query: str, n=3):
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
        "Also suggest 3-5 internal links (anchor text + slug). "
        "Do NOT write full paragraphs."
    )
    user = (
        f"Search query: {query}\n"
        f"Search intent: {intent}\n"
        f'Primary keywords: {", ".join(top_kw[:10])}\n'
        f'LSI keywords to sprinkle: {", ".join(lsi_kw)}\n\n'
        f"Competitor snapshot:\n"
    )
    for i, a in enumerate(analyses, 1):
        user += f"{i}. {a['url']}\n   KW: {', '.join(a['keywords'][:10])}\n"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=900,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

# ── UI ─────────────────────
st.set_page_config("SEO Outline Generator", "🔍")
st.title("🔍 SEO Article Outline Generator")

q = st.text_input("Zadej vyhledávací dotaz")
if q:
    if not SERP_API_KEY or not os.getenv("OPENAI_API_KEY"):
        st.error("Chybí API klíče.")
        st.stop()

    intent = detect_intent(q)
    st.caption(f"Detekovaný intent: **{intent}**")

    st.info("Vyhledávám konkurenci…")
    serp = search_google(q)
    if not serp:
        st.error("SerpAPI nic nevrátilo.")
        st.stop()

    analyses, docs_all = [], []

    for res in serp:
        url = res.get("link")
        title = res.get("title", url)
        with st.container():
            st.subheader(title)
            ext = tldextract.extract(url)
            st.caption(".".join(p for p in [ext.domain, ext.suffix] if p))

            preview, kw, full_text = analyse_page(url)
            docs_all.append(full_text)
            st.markdown("**Top KW:** " + ", ".join(f"`{w}`" for w, _ in kw))
            with st.expander("Ukázka textu"):
                st.write(preview)
            analyses.append({"url": url, "keywords": [w for w, _ in kw]})

    # agregace KW + LSI
    comb = Counter()
    for a in analyses:
        comb.update(a["keywords"])
    top_kw = [w for w, _ in comb.most_common(40)]
    lsi_kw = get_lsi_keywords(docs_all, top_n=12)

    st.info("Generuji osnovu…")
    outline = propose_outline(q, intent, top_kw, lsi_kw, analyses)

    st.markdown("---")
    st.subheader("📄 Výstup")
    st.markdown(outline, unsafe_allow_html=True)
    st.success("Hotovo ✔")
