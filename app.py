"""seo_article_generator.py – Streamlit app for SEO article ideation
====================================================================

Fix 06‑2025‑06‑12
-----------------
* **Bugfix:** replace unsupported regex using `\p{L}` (causing `re.PatternError`) with
  a safe Unicode‑aware pattern `\b\w{2,}\b`. No extra dependency required.
* Added richer Czech + English stop‑word list.
* Minor: show full traceback in Streamlit `st.exception` for easier debugging.

Usage
-----
```bash
# local run
export SERPAPI_API_KEY="..."
export OPENAI_API_KEY="..."
streamlit run seo_article_generator.py
```

Dependencies are pinned in `requirements.txt` (must include
`streamlit`, `google-search-results`, `beautifulsoup4`, `tldextract`, `openai`).

"""
from __future__ import annotations

import os
import re
import html
import json
import requests
import collections
from typing import List, Tuple

import streamlit as st
from bs4 import BeautifulSoup
from tldextract import extract as tld_extract
from serpapi import GoogleSearch  # provided by google-search-results
from openai import OpenAI

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
client = OpenAI()  # OPENAI_API_KEY picked up from env vars

TOP_N_KEYWORDS = 25

STOP_WORDS = {
    # Czech
    "a", "i", "aby", "aj", "ale", "anebo", "ani", "asi", "atd", "atp", "bez", "bude", "budou", "by", "byl", "byla", "byli", "bylo", "byly", "co", "či", "dál", "další", "den", "deset", "dnes", "do", "ho", "i", "jak", "jedna", "jedné", "jedni", "jedno", "jsme", "jiný", "kam", "kde", "kdo", "kdy", "když", "ke", "kolik", "která", "které", "který", "mít", "mně", "může", "my", "na", "nad", "nam", "námi", "nás", "náš", "ne", "některý", "než", "nic", "nich", "ním", "nímž", "níž", "nyní", "o", "od", "pak", "po", "pod", "podle", "pokud", "proto", "proč", "před", "přes", "při", "s", "se", "si", "tak", "tato", "tě", "těch", "to", "tohle", "tom", "tomto", "tomu", "toto", "třeba", "tu", "tuto", "ty", "tím", "tímto", "u", "v", "vám", "vámi", "vás", "váš", "ve", "vedle", "však", "všechen", "vy", "z", "za", "zatím", "že",
    # English
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
}

# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def keyword_frequency(text: str, top_n: int = TOP_N_KEYWORDS) -> List[Tuple[str, int]]:
    """Return the `top_n` most common keywords from *text* (after stop‑word removal).
    Uses a safe Unicode‑aware regex. Skips tokens shorter than 2 chars.
    """
    tokens = re.findall(r"\b\w{2,}\b", text.lower(), flags=re.UNICODE)
    tokens = [t for t in tokens if t not in STOP_WORDS and not t.isdigit()]
    return collections.Counter(tokens).most_common(top_n)


def serpapi_search(query: str, num_results: int = 3) -> List[dict]:
    search = GoogleSearch({
        "q": query,
        "num": num_results,
        "api_key": SERPAPI_API_KEY,
        "hl": "cs",
        "gl": "cz",
    })
    results = search.get_dict()
    return results.get("organic_results", [])[:num_results]


def fetch_page_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }
    try:
        r = requests.get(url, timeout=20, headers=headers)
        r.raise_for_status()
        return r.text
    except Exception as exc:
        st.warning(f"Nepodařilo se stáhnout {url}: {exc}")
        return ""


def extract_visible_text(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def analyse_page(url: str) -> tuple[str, list[tuple[str,int]]]:
    html_raw = fetch_page_html(url)
    plain = extract_visible_text(html_raw)
    kw = keyword_frequency(plain)
    return plain, kw


def propose_article(query: str, top_keywords: list[str], analyses: list[dict]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Czech SEO copywriter. Based on provided keyword "
                "list and competitor analyses, draft an outline and 400‑600 word "
                "article that can rank in Google Top 3. Write in Czech. Use clear "
                "H2/H3 headings and short paragraphs."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "query": query,
                "top_keywords": top_keywords,
                "competitors": analyses,
            }, ensure_ascii=False),
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.set_page_config(page_title="SEO Article Idea Generator", page_icon="🔍", layout="centered")
st.title("🔍 SEO Article Idea Generator")

query = st.text_input("Zadej vyhledávací dotaz")

if query:
    with st.spinner("Analyzuji výsledky…"):
        organic = serpapi_search(query)
        if not organic:
            st.error("Nenalezeny žádné výsledky.")
            st.stop()

        analyses = []
        for res in organic:
            url = res.get("link")
            domain = ".".join(part for part in tld_extract(url) if part)
            st.markdown(f"### 🔗 {domain}")
            plain, kw = analyse_page(url)
            st.markdown("**Top klíčová slova:** " + ", ".join([f"`{w}`" for w, _ in kw]))
            analyses.append({"url": url, "keywords": [w for w, _ in kw]})

        # agreguj klíčová slova (průnik/top TF‑IDF není třeba pro demo)
        all_kw = collections.Counter([w for a in analyses for w in a["keywords"]]).most_common(40)
        top_kw = [w for w, _ in all_kw[:25]]

    st.subheader("📝 Návrh článku")
    try:
        article_md = propose_article(query, top_kw, analyses)
        st.markdown(article_md)
    except Exception as exc:
        st.exception(exc)
