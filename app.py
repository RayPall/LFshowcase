"""
seo_article_generator.py
Streamlit aplikace, která:
1) provede Google vyhledávání (SerpAPI) na zadaný dotaz
2) stáhne a SEO-analyzuje TOP 3 stránky
3) vygeneruje jen **osnovu SEO článku** (H1/H2/H3 + poznámky)

Potřebné proměnné prostředí / Streamlit Secrets:
  SERPAPI_API_KEY
  OPENAI_API_KEY
"""

import os
import re
import requests
from collections import Counter

import streamlit as st
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from openai import OpenAI
import tldextract

# ── API klíče ────────────────────────────────────────────────────────────────
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()  # načte OPENAI_API_KEY automaticky

# ── stop-slova EN + CZ (základ) ─────────────────────────────────────────────
STOP_WORDS = {
    # english
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "be", "as", "at", "by", "this", "that", "from", "it", "its",
    "will", "was", "were", "has", "have", "had", "but", "not", "your", "you",
    # czech
    "a", "i", "k", "o", "u", "s", "v", "z", "na", "že", "se", "je", "jsou",
    "by", "byl", "byla", "bylo", "aby", "do", "od", "po", "pro", "pod", "nad",
    "který", "která", "které", "co", "to", "toto", "tyto", "ten", "ta", "tím",
    "tuto", "tu", "jako", "kde", "kdy", "jak", "tak", "také", "bez",
}

# ── pomocné funkce ──────────────────────────────────────────────────────────
def keyword_frequency(text: str, top_n: int = 20):
    """
    Vrátí top_n nejčastějších tokenů (≥3 znaky, pouze písmena),
    očistí od stop-slov a čísel.
    """
    tokens = re.findall(r"\b[^\W\d_]{3,}\b", text.lower(), flags=re.UNICODE)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return Counter(tokens).most_common(top_n)


def fetch_html(url: str) -> str:
    """Stáhne HTML stránku a pokusí se správně nastavit kódování."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; SEOArticleBot/1.0; "
            "+https://example.com/bot)"
        )
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.ok:
            # 👉 pokud server neposlal encoding, použij heuristiku
            if not r.encoding or r.encoding.lower() == "utf-8":
                r.encoding = r.apparent_encoding or "utf-8"
            return r.text
    except requests.RequestException:
        pass
    return ""


def analyse_page(url: str):
    """Vrátí ukázku textu (max 2000 znaků) a seznam klíčových slov stránky."""
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    # odstranění skriptů / stylů
    for s in soup(["script", "style", "noscript"]):
        s.extract()

    text = " ".join(soup.stripped_strings)
    kw = keyword_frequency(text)
    return text[:2000], kw


def search_google(query: str, num_results: int = 3):
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": SERP_API_KEY,
        "hl": "cs",
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])[:num_results]


def propose_outline(query: str, top_keywords, analyses):
    """Vrátí **pouze osnovu** článku (Markdown)."""
    system = (
        "You are an expert Czech SEO strategist. "
        "Generate ONLY a detailed outline (H1, H2, optional H3 headings and "
        "bullet-point notes) for an SEO article that can rank top-3 for the "
        "given query. Do NOT write full paragraphs."
    )

    user = (
        f"Search query: {query}\n"
        f"Aggregated competitor keywords: {', '.join(top_keywords)}\n\n"
        f"Competitor notes:\n"
    )
    for i, a in enumerate(analyses, 1):
        user += (
            f"{i}. {a['url']}\n"
            f"   keywords: {', '.join(a['keywords'])[:120]}\n"
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# ── Streamlit UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="SEO Article Idea Generator", page_icon="🔍")
st.title("🔍 SEO Article Idea Generator")

query = st.text_input("Zadej vyhledávací dotaz", value="")

if query:
    if not SERP_API_KEY or not OPENAI_API_KEY:
        st.error("❌ Chybí `SERPAPI_API_KEY` nebo `OPENAI_API_KEY`.")
        st.stop()

    st.info("⏳ Vyhledávám a analyzuji konkurenci…")

    results = search_google(query)
    if not results:
        st.error("Google/SerpAPI nevrátil žádné výsledky.")
        st.stop()

    analyses = []

    for res in results:
        url = res.get("link")
        title = res.get("title") or url

        st.subheader(title)
        # doména (ext.domain + ext.suffix)
        try:
            ext = tldextract.extract(url)
            domain_parts = [ext.domain, ext.suffix]
            domain = ".".join(p for p in domain_parts if p)
        except Exception:
            domain = url
        st.caption(domain)

        preview, kw = analyse_page(url)
        st.markdown(
            "**Top klíčová slova konkurence:** "
            + ", ".join(f"`{w}`" for w, _ in kw)
        )
        with st.expander("Ukázka textu"):
            st.write(preview)

        analyses.append({"url": url, "keywords": [w for w, _ in kw]})

    # agregované klíčové fráze přes všechny konkurenční stránky
    combined = Counter()
    for a in analyses:
        combined.update(a["keywords"])
    top_kw = [w for w, _ in combined.most_common(30)]

    st.info("📝 Generuji osnovu článku…")
    outline_md = propose_outline(query, top_kw, analyses)

    st.markdown("---")
    st.subheader("📄 Návrh (outline) SEO článku")
    st.markdown(outline_md, unsafe_allow_html=True)
    st.success("✅ Hotovo – osnova vygenerována!")
