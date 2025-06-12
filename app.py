"""seo_article_generator.py – Streamlit app for SEO‑friendly article ideation
--------------------------------------------------------------------------

Usage (local):
    $ export SERPAPI_API_KEY="..."
    $ export OPENAI_API_KEY="..."   # or set in .streamlit/secrets.toml on Streamlit Cloud
    $ streamlit run seo_article_generator.py

The app:
1. Uses SerpAPI to fetch the Top 3 Google results for a user query.
2. Scrapes and analyses their content structure & keywords.
3. Calls OpenAI Chat Completions (>= v1.x SDK) to propose an article outline & draft
   optimised to rank in the same query’s Top 3.

Compatible with **openai >= 1.0** (new Python SDK).
"""
from __future__ import annotations

import os
import re
import json
import textwrap
from collections import Counter
from urllib.parse import urlparse

import requests
import streamlit as st
import tldextract
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from openai import OpenAI

###############################################################################
# --------------------------- Configuration ----------------------------------
###############################################################################
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", st.secrets.get("SERPAPI_API_KEY", ""))

# Initialise OpenAI client (reads OPENAI_API_KEY from env var or secrets.toml)
openai_client = OpenAI()

# Minimal CZ+EN stop‑words list (extend as needed)
STOP_WORDS = {
    "a", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
    "se", "a", "co", "jak", "je", "jsou", "na", "o", "od", "pro", "s", "se", "si", "z", "ve", "že", "do", "podle", "které"
}

###############################################################################
# ------------------------------ Utilities -----------------------------------
###############################################################################

def google_search(query: str, num_results: int = 3) -> list[dict]:
    """Return SerpAPI results list (each with 'title' & 'link')."""
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "hl": "cs",
        "api_key": SERPAPI_API_KEY,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])[:num_results]


def fetch_html(url: str, timeout: int = 10) -> str:
    """Fetch page HTML with a desktop UA & basic error handling."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException:
        return ""


def extract_structure(text_html: str) -> tuple[str, list[str]]:
    """Return plain text & list of headings (H1‑H6)."""
    soup = BeautifulSoup(text_html, "html.parser")

    # Extract headings in order of appearance
    headings = [h.get_text(" ", strip=True) for h in soup.find_all(re.compile(r"^h[1-6]$"))]

    # Remove script/style and get body text
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return text, headings


def keyword_frequency(text: str, top_n: int = 20) -> list[tuple[str, int]]:
    """Return list of (keyword, freq) sorted by freq desc."""
    # Simple word tokenisation; lowercase & remove non‑letters
    tokens = re.findall(r"[\p{L}]+", text.lower(), re.UNICODE)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    counts = Counter(tokens)
    return counts.most_common(top_n)


def propose_article(query: str, top_keywords: list[str], analyses: list[dict]) -> str:
    """Call OpenAI chat completion to propose an article draft."""
    # System prompt primes assistant for SEO expertise
    sys_prompt = (
        "You are an experienced Czech SEO copywriter. Create an article outline and "
        "short draft that can rank Top 3 in Google for the user query. Use modern "
        "SEO best‑practices: compelling H1, semantically‑rich H2/H3, FAQ, meta description, "
        "and natural keyword usage."
    )

    # Prepare message with context (we trim to keep token count reasonable)
    anal_json = json.dumps(analyses, ensure_ascii=False, indent=2)[:6_000]
    content_prompt = textwrap.dedent(
        f"""
        USER QUERY: "{query}"
        TOP KEYWORDS (frequency sorted): {", ".join(top_keywords[:15])}

        COMPETITOR ANALYSIS (JSON):
        {anal_json}

        Please output:
        1. Suggested SEO Title (max 65 chars)
        2. Meta description (~155 chars)
        3. Outline with H1‑H3 headings
        4. Short draft (~400‑500 words) in Czech
        5. List of 10 important keywords (no comma‑keyword stuffing)
        """
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content_prompt},
        ],
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

###############################################################################
# ----------------------------- Streamlit UI ---------------------------------
###############################################################################

st.set_page_config(page_title="SEO Article Idea Generator", layout="wide")
st.title("🔎 SEO Article Idea Generator")

with st.sidebar:
    st.header("🔧 Nastavení")
    if not SERPAPI_API_KEY:
        SERPAPI_API_KEY = st.text_input("SerpAPI klíč", type="password")
        st.info("Klíč se použije pouze lokálně a neukládá se.")
    st.markdown(
        "[SerpAPI](https://serpapi.com) – potřebuješ bezplatný či placený API key.\n\n"
        "OpenAI klíč se načítá z proměnné **OPENAI_API_KEY** nebo z `secrets.toml`."
    )

query = st.text_input("Zadej vyhledávací dotaz", placeholder="např. jak vybrat elektrokolo")

if query and SERPAPI_API_KEY:
    with st.spinner("Hledám nejlepší výsledky …"):
        search_results = google_search(query, num_results=3)

    if not search_results:
        st.error("Nenalezeny žádné výsledky nebo SerpAPI quota vyčerpána.")
        st.stop()

    analyses = []
    all_text = ""

    for idx, res in enumerate(search_results, start=1):
        url = res.get("link")
        title = res.get("title", urlparse(url).netloc)

        with st.spinner(f"Stahuji a analyzuji: {title}"):
            html = fetch_html(url)
            text, headings = extract_structure(html)
            kw_freq = keyword_frequency(text)

        # Display competitor card
        with st.expander(f"#{idx} {title}"):
            st.write("**URL:**", url)
            st.write("**H nadpisy:**", headings[:10])
            st.write("**Top klíčová slova:**",
                     ", ".join(f"{w} ({c})" for w, c in kw_freq[:10]))

        analyses.append({
            "url": url,
            "title": title,
            "headings": headings,
            "top_keywords": kw_freq,
        })
        all_text += " " + text

    # Global keyword list
    global_kw = keyword_frequency(all_text)
    top_keywords = [w for w, _ in global_kw]

    st.markdown("---")
    st.subheader("📄 Návrh článku")

    with st.spinner("Generuji článek pomocí OpenAI …"):
        article_md = propose_article(query, top_keywords, analyses)

    st.markdown(article_md)

else:
    st.info("Zadej dotaz a připoj platný SerpAPI klíč.")
