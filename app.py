"""
SEO Article Outline Generator
-----------------------------
* Vyhledá TOP 3 výsledky Googlu přes SerpAPI.
* Stáhne jejich HTML a analyzuje klíčová slova (frekvence + TF-IDF LSI).
* Vygeneruje pouze **osnovu** článku (H-headery, bullet-pointy),
  meta-title, meta-description a návrhy interních odkazů.

Potřebné proměnné prostředí / Streamlit Secrets:
  SERPAPI_API_KEY
  OPENAI_API_KEY
"""

import os
import re
import requests
from collections import Counter, defaultdict

import streamlit as st
from bs4 import BeautifulSoup
from serpapi import GoogleSearch            # správný namespace
from openai import OpenAI
import tldextract

# NLP & ML
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.snowball import SnowballStemmer  # univerzální stemmer

nltk.download("punkt", quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# API klíče
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
client = OpenAI()                            # načte OPENAI_API_KEY z env

# Stop-slova (CZ + EN, základ)
STOP_WORDS = set("""
the a an and or of to in on for with is are be as at by this that from it its
will was were has have had but not your you
a i k o u s v z na že se je jsou by byl byla bylo aby do od po pro pod nad
který která které co to toto tyto ten ta tím tuto tu jako kde kdy jak tak také bez
""".split())

CS_STEM = SnowballStemmer("czech")
EN_STEM = SnowballStemmer("english")

# ─────────────────────────────────────────────────────────────────────────────
# Heuristická detekce search-intent
def detect_intent(query: str) -> str:
    q = query.lower()
    trans_kw = {"koupit", "cena", "objednat", "recenze", "srovnání", "discount"}
    if any(w in q for w in trans_kw):
        return "transactional"
    if q.startswith(("jak", "co", "proč", "kdy", "kde", "who", "how", "what", "why")):
        return "informational"
    return "informational"

# ─────────────────────────────────────────────────────────────────────────────
# Stahování a analýza HTML
def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (SEOOutlineBot/1.0; +https://example.com/bot)"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.ok:
            # rozpoznání kódování, prevence mojibake
            if not r.encoding or r.encoding.lower() == "utf-8":
                r.encoding = r.apparent_encoding or "utf-8"
            return r.text
    except requests.RequestException:
        pass
    return ""

def keyword_frequency(text: str, lang="cz", top_n: int = 20):
    # pouze písmena, min. 3 znaky (UNICODE)
    tokens = re.findall(r"\b[^\W\d_]{3,}\b", text.lower(), flags=re.UNICODE)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    stemmer = CS_STEM if lang == "cz" else EN_STEM
    stems = [stemmer.stem(t) for t in tokens]

    counts = defaultdict(int)
    for stem in stems:
        counts[stem] += 1

    # mapujeme zpět na „nejreprezentativnější“ původní slovo
    stem_to_word = {}
    for word in tokens:
        st = stemmer.stem(word)
        if st not in stem_to_word:
            stem_to_word[st] = word

    return [(stem_to_word[s], c) for s, c in Counter(counts).most_common(top_n)]

def get_lsi_keywords(docs, top_n: int = 12):
    vec = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words=list(STOP_WORDS)
    )
    X = vec.fit_transform(docs)
    scores = X.sum(axis=0).A1
    idx = scores.argsort()[::-1][:top_n]
    feats = vec.get_feature_names_out()
    return [feats[i] for i in idx]

def analyse_page(url: str):
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = " ".join(soup.stripped_strings)
    kw = keyword_frequency(text)
    return text[:2000], kw, text  # preview, KW, celý text

# ─────────────────────────────────────────────────────────────────────────────
# SerpAPI vyhledávání
def search_google(query: str, n: int = 3):
    search = GoogleSearch({
        "engine": "google",
        "q": query,
        "num": n,
        "api_key": SERP_API_KEY,
        "hl": "cs",
    })
    return search.get_dict().get("organic_results", [])[:n]

# ─────────────────────────────────────────────────────────────────────────────
# Generování osnovy přes OpenAI
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
        f"Primary keywords: {', '.join(top_kw[:10])}\n"
        f"LSI keywords: {', '.join(lsi_kw)}\n\n"
        f"Competitor snapshot:\n"
    )
    for i, a in enumerate(analyses, 1):
        user += f"{i}. {a['url']}\n   KW: {', '.join(a['keywords'][:10])}\n"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=900,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
st.set_page_config(page_title="SEO Outline Generator", page_icon="🔍")
st.title("🔍 SEO Article Outline Generator")

query = st.text_input("Zadej vyhledávací dotaz")

if query:
    if not SERP_API_KEY or not os.getenv("OPENAI_API_KEY"):
        st.error("❌ Chybí SERPAPI_API_KEY nebo OPENAI_API_KEY.")
        st.stop()

    intent = detect_intent(query)
    st.caption(f"Detekovaný intent: **{intent}**")
    st.info("⏳ Vyhledávám konkurenci…")

    serp_results = search_google(query)
    if not serp_results:
        st.error("SerpAPI nevrátilo žádné výsledky.")
        st.stop()

    analyses, all_docs = [], []

    for res in serp_results:
        url = res.get("link")
        title = res.get("title", url)

        st.subheader(title)
        ext = tldextract.extract(url)
        domain = ".".join(p for p in [ext.domain, ext.suffix] if p) or url
        st.caption(domain)

        preview, kw, full = analyse_page(url)
        all_docs.append(full)
        st.markdown("**Top klíčová slova:** " + ", ".join(f"`{w}`" for w, _ in kw))
        with st.expander("Ukázka textu"):
            st.write(preview)

        analyses.append({"url": url, "keywords": [w for w, _ in kw]})

    # agregace KW + LSI
    combined = Counter()
    for a in analyses:
        combined.update(a["keywords"])
    top_kw = [w for w, _ in combined.most_common(40)]
    lsi_kw = get_lsi_keywords(all_docs)

    st.info("📝 Generuji osnovu článku…")
    outline_md = propose_outline(query, intent, top_kw, lsi_kw, analyses)

    st.markdown("---")
    st.subheader("📄 Výstup")
    st.markdown(outline_md, unsafe_allow_html=True)
    st.success("✅ Osnova vygenerována!")
