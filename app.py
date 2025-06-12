"""
seo_article_generator.py

Streamlit aplikace, která:
1) Vyhledá TOP 3 výsledky Googlu přes SerpAPI
2) Stáhne jejich HTML (vždy dekóduje přes apparent_encoding) a vytáhne klíčová slova
3) Vygeneruje **pouze osnovu** článku (H1/H2/H3 + bullet-pointy),
   meta‐title, meta‐description a návrhy interních odkazů.

Potřebné proměnné prostředí / Streamlit Secrets:
  • SERPAPI_API_KEY
  • OPENAI_API_KEY
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

# ─────────────────────────────────────────────────────────────────────────────
# API klíče & klient
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
client       = OpenAI()  # načte OPENAI_API_KEY z env

# základní stop-slova (CZ + EN)
STOP_WORDS = {
    # English
    "the","a","an","and","or","of","to","in","on","for","with","is","are","be",
    "as","at","by","this","that","from","it","its","will","was","were","has","have",
    "had","but","not","your","you",
    # Czech
    "a","i","k","o","u","s","v","z","na","že","se","je","jsou","by","byl","byla",
    "bylo","aby","do","od","po","pro","pod","nad","který","která","které","co","to",
    "ten","ta","tím","tuto","tu","jako","kde","kdy","jak","tak","také","bez"
}

# ─────────────────────────────────────────────────────────────────────────────
def fetch_html(url: str) -> str:
    """
    Stáhne HTML a vždy dekóduje bajty přes apparent_encoding (ne .text),
    aby se předešlo mojibake.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SEOArticleBot/1.0)"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.ok:
            encoding = r.apparent_encoding or "utf-8"
            return r.content.decode(encoding, errors="replace")
    except requests.RequestException:
        pass
    return ""

def keyword_frequency(text: str, top_n: int = 20):
    """
    Extrahuje tokeny pouze z písmen (min. 3 znaky),
    odfiltruje stop-slova a vrátí top_n nejčastějších.
    """
    # \b hranice slova, [^\W\d_] = jen písmena Unicode, {3,} minimálně tři
    tokens   = re.findall(r"\b[^\W\d_]{3,}\b", text.lower(), flags=re.UNICODE)
    filtered = [t for t in tokens if t not in STOP_WORDS]
    return Counter(filtered).most_common(top_n)

def analyse_page(url: str):
    """
    Stáhne stránku, odstraní skripty/styly, vrátí náhled textu a KW.
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript"]):
        tag.extract()
    text = " ".join(soup.stripped_strings)
    kw   = keyword_frequency(text)
    return text[:2000], kw  # náhled prvních 2 000 znaků + seznam (slovo, četnost)

def search_google(query: str, num_results: int = 3):
    """
    Vrátí organické výsledky SerpAPI pro zadaný dotaz.
    """
    params  = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": SERP_API_KEY,
        "hl": "cs",
    }
    results = GoogleSearch(params).get_dict()
    return results.get("organic_results", [])[:num_results]

def propose_outline(query: str, top_keywords, analyses):
    """
    Vygeneruje **pouze osnovu** článku v Markdownu:
      • H1/H2/H3 + bullet-pointy
      • meta-title (≤60 char), meta-description (≤155 char)
      • návrh 3–5 interních odkazů
    """
    system = (
        "You are an expert Czech SEO strategist. "
        "Generate ONLY a detailed outline (H1, H2, optional H3) with bullet-point notes, "
        "a meta-title (<=60 chars) and meta-description (<=155 chars), "
        "and suggest 3–5 internal links (anchor text + slug). "
        "Do NOT write full paragraphs."
    )
    user = (
        f"Search query: {query}\n"
        f"Aggregated competitor keywords: {', '.join(top_keywords)}\n\n"
        "Competitor snapshot:\n"
    )
    for i, a in enumerate(analyses, 1):
        user += f"{i}. {a['url']}\n   keywords: {', '.join(a['keywords'])[:120]}\n"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content": system},
            {"role":"user",   "content": user}
        ],
        max_tokens=1024,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
st.set_page_config(page_title="SEO Article Outline Generator", page_icon="🔍")
st.title("🔍 SEO Article Outline Generator")

query = st.text_input("Zadej vyhledávací dotaz", value="")
if query:
    if not SERP_API_KEY or not os.getenv("OPENAI_API_KEY"):
        st.error("❌ Chybí `SERPAPI_API_KEY` nebo `OPENAI_API_KEY`.")
        st.stop()

    st.info("⏳ Vyhledávám a analyzuji konkurenci…")
    results = search_google(query)
    if not results:
        st.error("Google/SerpAPI nevrátil žádné výsledky.")
        st.stop()

    analyses = []
    for res in results:
        url   = res.get("link")
        title = res.get("title") or url

        st.subheader(title)
        try:
            ext = tldextract.extract(url)
            domain = ".".join(p for p in [ext.domain, ext.suffix] if p)
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

    combined = Counter()
    for a in analyses:
        combined.update(a["keywords"])
    top_kw = [w for w, _ in combined.most_common(30)]

    st.info("📝 Generuji osnovu článku…")
    outline_md = propose_outline(query, top_kw, analyses)

    st.markdown("---")
    st.subheader("📄 Návrh (outline) SEO článku")
    st.markdown(outline_md, unsafe_allow_html=True)
    st.success("✅ Osnova vygenerována!")
