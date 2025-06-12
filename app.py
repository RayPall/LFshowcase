"""Streamlit SEO Article Idea Generator
Author: ChatGPT (OpenAI)
Date: 2025‑06‑12

Instructions
============
1. Export environment variables (or create a .env file) with your keys:
   export OPENAI_API_KEY="sk‑..."
   export SERPAPI_API_KEY="..."
2. Install dependencies:
   pip install streamlit requests beautifulsoup4 tldextract openai python-dotenv
3. Run the app:
   streamlit run seo_generator_streamlit.py

The app takes a search query in Czech, fetches the top‑3 Google results via SerpAPI,
parses them, extracts headings & keywords, and uses the OpenAI API to draft a
1500‑word Markdown article structured for SEO.
"""

import collections
import os
import re
from urllib.parse import urlencode

import openai
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ---------------------------
# Configuration & Constants
# ---------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")

CZECH_STOPWORDS = {
    "a","i","ale","že","se","na","s","ve","v","je","jsem","jsi","jsou","byla","byli","bylo","by",
    "už","do","pro","k","o","u","z","ze","si","tak","to","tento","tato","toto","tyto","jak","při","od",
    "který","která","které","když","proto","co","být","mít","musí","může","více","méně","ne","ano"
}

# ---------------------------
# Helper functions
# ---------------------------

def google_search(query: str, num_results: int = 3):
    """Return list of dicts with 'title', 'link', 'snippet' using SerpAPI."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": num_results,
        "hl": "cs",
    }
    url = f"https://serpapi.com/search?{urlencode(params)}"
    res = requests.get(url, timeout=15)
    res.raise_for_status()
    data = res.json()
    return [
        {
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
        }
        for item in data.get("organic_results", [])[:num_results]
    ]


def fetch_html(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SEOGenerator/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception:
        return ""


def extract_headings(soup: BeautifulSoup):
    headings = []
    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            text = tag.get_text(strip=True)
            if text:
                headings.append({"level": level, "text": text})
    return headings


def keyword_frequency(text: str, top_n: int = 20):
    words = re.findall(r"\b[\wá-žÁ-Ž]{4,}\b", text.lower())
    tokens = [w for w in words if w not in CZECH_STOPWORDS]
    return collections.Counter(tokens).most_common(top_n)


def analyze_page(html: str):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title else ""
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_desc = meta_desc_tag["content"].strip() if meta_desc_tag else ""
    headings = extract_headings(soup)
    body_text = soup.get_text(" ", strip=True)
    keywords = keyword_frequency(body_text)
    return {
        "title": title,
        "meta_description": meta_desc,
        "headings": headings,
        "keywords": keywords,
        "word_count": len(body_text.split()),
    }


def summarize_keywords(analyses):
    combined = collections.Counter()
    for a in analyses:
        combined.update(dict(a["keywords"]))
    return combined.most_common(30)


def propose_article(query: str, top_keywords, analyses, language="cs") -> str:
    prompt = f"""
Jsi zkušený SEO specialista a copywriter.

Napiš originální SEO článek v jazyce {language}, který cílí na klíčové slovo "{query}" a má potenciál umístit se v top 3 Google.

Podklady k analýze konkurence:
- Častá klíčová slova: {top_keywords}
- Struktura nadpisů konkurence: {[a['headings'] for a in analyses]}

Požadavky na článek:
- cca 1500 slov
- Struktura H1–H3 (případně H4) s vhodným rozložením klíčových slov (žádný keyword stuffing)
- Atraktivní úvod, hluboká expertíza, závěr s CTA
- Návrh *meta title* do 60 znaků a *meta description* do 155 znaků
- Vrať výsledek ve formátu Markdown.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()

# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(page_title="SEO Generátor Nápadů", layout="wide")
    st.title("📝 SEO Generátor Nápadů na Články")

    query = st.text_input("Zadejte vyhledávací dotaz / klíčové slovo", placeholder="např. jak vybrat elektrokolo")

    if st.button("Generovat") and query.strip():
        with st.spinner("🔎 Vyhledávám konkurenci na Googlu…"):
            results = google_search(query)
            if not results:
                st.error("Nepodařilo se získat výsledky. Zkontrolujte API klíč SerpAPI.")
                st.stop()

        st.subheader("Top 3 výsledky")
        for res in results:
            st.markdown(f"- [{res['title']}]({res['link']}) — {res['snippet']}")

        analyses = []
        with st.spinner("📊 Analyzuji stránky konkurence…"):
            for res in results:
                html = fetch_html(res["link"])
                analyses.append(analyze_page(html))

        st.subheader("SEO analýza konkurence")
        for idx, analysis in enumerate(analyses, 1):
            with st.expander(f"Výsledek {idx}: {results[idx-1]['title']}"):
                st.write("**Titulek:**", analysis["title"])
                st.write("**Meta popisek:**", analysis["meta_description"] or "—")
                st.write("**Počet slov:**", analysis["word_count"])
                st.write("**Top klíčová slova:**", analysis["keywords"])
                st.write("**Nadpisy:**")
                for h in analysis["headings"]:
                    indent = " " * 2 * (h["level"]-1)
                    st.markdown(f"{indent}- H{h['level']} {h['text']}")

        top_keywords = summarize_keywords(analyses)

        with st.spinner("✍️ Generuji návrh článku…"):
            article_md = propose_article(query, top_keywords, analyses)

        st.subheader("📄 Návrh článku")
        st.markdown(article_md)


if __name__ == "__main__":
    main()
