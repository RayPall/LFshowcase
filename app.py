"""
seo_article_generator.py
Streamlit aplikace, která:
1. Provede Google vyhledávání (SerpAPI) na zadaný dotaz
2. Stáhne a SEO-analyzuje 3 nejvýše postavené výsledky
3. Vygeneruje návrh SEO článku, který cílí na TOP 3

⚙️  Potřeba API klíče:
   • SERPAPI_API_KEY
   • OPENAI_API_KEY
   (ulož jako proměnné prostředí nebo ve Streamlit Cloud → Secrets)
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

# ── Klíče & klienti ───────────────────────────────────────────────────────────

SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()  # načte OPENAI_API_KEY z prostředí

# ── Stop-slova pro EN + CZ (základní) ─────────────────────────────────────────

STOP_WORDS = {
    # english
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "be", "as", "at", "by", "this", "that", "from", "it", "its", "will",
    "was", "were", "has", "have", "had", "but", "not", "your", "you",
    # czech
    "a", "i", "k", "o", "u", "s", "v", "z", "na", "že", "se", "je", "jsou",
    "by", "byl", "byla", "bylo", "aby", "do", "od", "po", "pro", "pod", "nad",
    "který", "která", "které", "co", "to", "toto", "tyto", "ten", "ta", "tím",
    "tuto", "tu", "jako", "kde", "kdy", "jak", "tak", "také", "bez",
}

# ── Pomocné funkce ────────────────────────────────────────────────────────────


def keyword_frequency(text: str, top_n: int = 20):
    """
    Vrátí `top_n` nejčastějších tokenů s délkou ≥ 2 znaky,
    očištěné o stop-slova a čísla.
    """
    tokens = re.findall(r"\b\w{2,}\b", text.lower(), flags=re.UNICODE)
    tokens = [t for t in tokens if t not in STOP_WORDS and not t.isdigit()]
    return Counter(tokens).most_common(top_n)


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; SEOArticleBot/1.0; "
            "+https://example.com/bot)"
        )
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.ok:
            return r.text
    except requests.RequestException:
        pass
    return ""


def analyse_page(url: str):
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    # zahodíme skripty / styly
    for s in soup(["script", "style", "noscript"]):
        s.extract()

    text = " ".join(soup.stripped_strings)
    kw = keyword_frequency(text)
    return text[:2000], kw  # vrátíme ukázku + klíčová slova


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


def propose_article(query: str, top_keywords, analyses):
    """
    Zavolá OpenAI Chat Completion a vrátí Markdown s článkem.
    """
    system = (
        "You are an expert Czech SEO copywriter. "
        "Generate a detailed SEO article outline followed by the full article "
        "text in Czech that can rank in Google's top 3 for the given query. "
        "Integrate the provided keywords naturally."
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
        max_tokens=2048,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="SEO Article Idea Generator", page_icon="🔍")
st.title("🔍 SEO Article Idea Generator")

query = st.text_input("Zadej vyhledávací dotaz", value="")

if query:
    # kontrola klíčů
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
        title = res.get("title")

        # Vykreslení titulku + domény
        st.subheader(title or url)
        try:
            ext = tldextract.extract(url)
            domain_parts = [ext.domain, ext.suffix]
            domain = ".".join(p for p in domain_parts if p)
        except Exception:
            domain = url
        st.caption(domain)

        # SEO analýza stránky
        preview, kw = analyse_page(url)
        st.markdown(
            "**Top klíčová slova konkurence:** "
            + ", ".join(f"`{w}`" for w, _ in kw)
        )
        with st.expander("Ukázka textu"):
            st.write(preview)

        analyses.append({"url": url, "keywords": [w for w, _ in kw]})

    # agregace klíčových slov napříč konkurencí
    combined = Counter()
    for a in analyses:
        combined.update(a["keywords"])
    top_kw = [w for w, _ in combined.most_common(30)]

    st.info("📝 Generuji návrh článku…")
    article_md = propose_article(query, top_kw, analyses)

    st.markdown("---")
    st.subheader("📄 Návrh SEO článku")
    st.markdown(article_md, unsafe_allow_html=True)
    st.success("✅ Hotovo – článek vygenerován!")
