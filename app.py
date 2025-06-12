"""
seo_article_generator.py

Streamlit aplikace, která:
1) Vyhledá TOP 5 výsledků Googlu přes SerpAPI
2) Stáhne jejich HTML (vždy dekóduje přes apparent_encoding) a vytáhne klíčová slova
3) Vygeneruje **pouze osnovu** článku (H1/H2/H3 + bullet-pointy),
   meta-title, meta-description a návrhy interních odkazů.

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

def fetch_html(url: str) -> str:
    """
    Stáhne HTML a vždy dekóduje bajty přes apparent_encoding (ne .text),
    aby se předešlo mojibake.
    """
    headers = {"User-Agent": "Mozilla/5.0 (SEOArticleBot/1.0)"}
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
    return text[:2000], kw  # náhled prvních 2000 znaků + seznam (slovo, četnost)

def search_google(query: str, num_results: int = 5):
    """
    Vrátí prvních num_results organických výsledků SerpAPI pro zadaný dota_
