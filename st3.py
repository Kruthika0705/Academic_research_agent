# streamlit_research_agent_llama_plus.py
import os
import re
import time
import json
import math
import hashlib
import requests
import urllib.parse
import streamlit as st
from typing import List, Dict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="📚 Research Agent (LLaMA)", layout="wide")
st.title("📚 Academic Research Agent — LLaMA (Groq)")
st.caption("Search ➜ Summarize ➜ Draft ➜ Review ➜ Check novelty ➜ Export")

# 🔑 Set your Groq API key here (or via environment)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "api key")

# Model + temperature in sidebar so you can tweak quickly
with st.sidebar:
    st.header("⚙️ Settings")
    MODEL = st.selectbox(
        "Groq LLaMA model",
        ["llama-3.3-70b-versatile", "llama3-70b-8192", "llama-3.1-8b-instant"],
        index=0
    )
    TEMP = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    MAX_LINKS = st.slider("Max links per source", 1, 10, 5)
    MAX_CONTEXT_ITEMS = st.slider("Items to feed LLM", 1, 15, 6)
    st.divider()
    st.markdown("**Tip:** Keep temperature low for factual tasks.")

# Session state
if "results" not in st.session_state:
    st.session_state.results = {}
if "combined_text" not in st.session_state:
    st.session_state.combined_text = ""
if "bib" not in st.session_state:
    st.session_state.bib = []
if "draft" not in st.session_state:
    st.session_state.draft = ""
if "abstract" not in st.session_state:
    st.session_state.abstract = ""

# ==========================================
# UTILS
# ==========================================
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"}

def _safe_get(url, timeout=12):
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout)
    except Exception as e:
        return None

def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def _dedupe_by_title(items: List[Dict], key="title"):
    seen, out = set(), []
    for it in items:
        t = _clean_text(it.get(key, "")).lower()
        if t and t not in seen:
            seen.add(t)
            out.append(it)
    return out

def jaccard_similarity(a: str, b: str) -> float:
    """Very simple novelty proxy: unique word overlap."""
    wa = set(re.findall(r"\b[a-zA-Z]{3,}\b", a.lower()))
    wb = set(re.findall(r"\b[a-zA-Z]{3,}\b", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

def top_overlap_passages(query_text: str, passages: List[str], k=5):
    scored = []
    for p in passages:
        scored.append((jaccard_similarity(query_text, p), p))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]

def to_bibtex(title: str, url: str):
    key = re.sub(r"[^a-z0-9]+", "", title.lower())[:20] or hashlib.md5(title.encode()).hexdigest()[:8]
    return f"@misc{{{key},\n  title={{\"{title}\"}},\n  howpublished={{\\url{{{url}}}}},\n  year={{n.d.}}\n}}"

# ==========================================
# SCRAPERS
# ==========================================




def scrape_google_scholar(query):
    try:
        search_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(query)}"
        resp = _safe_get(search_url)
        if not resp:
            return [{"title": "Error fetching Google Scholar", "link": search_url}]
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select(".gs_r.gs_or.gs_scl"):
            a = item.select_one(".gs_rt a")
            title = _clean_text(a.get_text()) if a else "Untitled"
            link = a.get("href") if a else ""
            snippet_tag = item.select_one(".gs_rs")
            snippet = _clean_text(snippet_tag.get_text()) if snippet_tag else ""
            results.append({"title": title, "link": link, "snippet": snippet, "source": "Scholar"})
        return _dedupe_by_title(results)[:MAX_LINKS]
    except Exception as e:
        return [{"title": f"Error: {e}", "link": ""}]

def fetch_page_text(url, max_chars=4000):
    """Fetch main paragraphs from a URL to enrich context."""
    resp = _safe_get(url, timeout=12)
    if not resp:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    paras = [ _clean_text(p.get_text()) for p in soup.select("p") if _clean_text(p.get_text()) ]
    text = " ".join(paras)
    return text[:max_chars]

# ==========================================
# LLM HELPERS (Groq / LLaMA)
# ==========================================
def make_llm():
    return ChatGroq(model=MODEL, temperature=TEMP)

def run_chain(prompt_tmpl: str, **kwargs) -> str:
    llm = make_llm()
    prompt = PromptTemplate(input_variables=list(kwargs.keys()), template=prompt_tmpl)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(**kwargs)

# Prompts
P_SUMMARY = """
You are a meticulous research assistant.
Topic: {topic}

Use ONLY this evidence to produce a structured summary:
{context}

Write:
- Key findings (bullets)
- Recent advances (bullets)
- Gaps / open problems (bullets)
- 5–7 references (bulleted list of title + URL from the context)
Be concise, factual, and avoid speculation.
"""

P_OUTLINE = """
You are an expert academic writer. Create a detailed research paper outline for the topic:
"{topic}"

Include sections:
1. Abstract (2–3 sentences)
2. Introduction (bulleted)
3. Related Work (bulleted, cite items from context)
4. Method/Approach (bulleted, generic but plausible)
5. Experiments/Evaluation (bulleted, what to measure)
6. Results & Discussion (bulleted, what you'd expect)
7. Limitations & Future Work (bulleted)
8. Conclusion (2–3 bullets)

Context you can cite:
{context}
"""

P_EXPAND_SECTION = """
You are an academic writing assistant. Expand the following section for a paper on "{topic}".

Section Title: {section_title}
Guidance / Notes: {notes}

Base your writing on this context if relevant (do not invent citations):
{context}

Write 2–5 coherent paragraphs in a scholarly but readable tone. Avoid hallucinations.
"""

P_PARAPHRASE = """
Paraphrase the following text to improve clarity, reduce redundancy, and keep meaning intact.
Preserve citations if present.

Text:
{text}
"""

P_REVIEWER = """
Act as a critical peer reviewer (top-tier venue). For a paper with topic "{topic}" and the following draft content:

DRAFT:
{draft}

CONTEXT (from retrieved literature):
{context}

Provide:
- Summary (2–3 sentences)
- Strengths (bullets)
- Weaknesses (bullets)
- Specific suggestions to improve novelty, methodology, and evaluation
- Final verdict (Accept / Weak Accept / Borderline / Weak Reject / Reject) with 1–2 sentence justification
"""

P_NOVELTY_LLM = """
You are assessing the novelty of a proposed idea/abstract.

ABSTRACT / IDEA:
{abstract}

CONTEXT (titles/snippets from literature):
{context}

Judge the novelty on a 0–10 scale, where:
0 = already well-known / saturated; 10 = clearly novel and distinct.
Explain briefly which sources overlap and why, and what angle could increase novelty.
Return JSON with keys: "score", "rationale", "overlapping_sources" (list of titles).
"""

# ==========================================
# UI — INPUTS
# ==========================================
col1, col2 = st.columns([2,1])
with col1:
    topic = st.text_input("🔎 Enter your research topic", placeholder="e.g., AI in Healthcare 2025")
with col2:
    run_btn = st.button("Run Web Search")

# ==========================================
# SEARCH PHASE
# ==========================================
def run_search(q: str):
    with st.spinner("Fetching search results..."):
        with ThreadPoolExecutor() as ex:
            futs = {
                ex.submit(scrape_google_scholar, q): "scholar",
            }
            out = {}
            for f in as_completed(futs):
                name = futs[f]
                out[name] = f.result()
    return out


if run_btn and topic:
    st.session_state.results = run_search(topic)

if st.session_state.results:
    st.subheader("🔗 Retrieved Results")
    st.markdown("**🎓 Google Scholar**")
    for r in st.session_state.results.get("scholar", [])[:MAX_LINKS]:
        link = r.get('link', '')
        st.markdown(f"- **{r['title']}** — {r.get('snippet', '')} {f'([PDF/Link]({link}))' if link else ''}")


    # Fetch page texts to enrich context
    with st.spinner("Extracting context from top links..."):
        texts = []
        bib_entries = []
        # pick top N across sources
        all_items = []
        for src in ["britannica","wikipedia","scholar"]:
            all_items.extend(st.session_state.results.get(src, [])[:MAX_CONTEXT_ITEMS])
        for item in all_items[:MAX_CONTEXT_ITEMS]:
            url = item.get("link","")
            if not url:
                continue
            txt = fetch_page_text(url)
            if txt:
                texts.append(f"[{item.get('source','web')}] {item['title']}: {txt}")
                bib_entries.append(to_bibtex(item['title'], url))
        st.session_state.bib = bib_entries

    # Combined context
    st.session_state.combined_text = "\n\n".join(texts)[:15000]

# ==========================================
# TABS: SUMMARY | WRITING | NOVELTY | DRAFT | EXPORT
# ==========================================
if st.session_state.combined_text:
    tabs = st.tabs(["🧠 Summary", "✍️ Writing Assistant", "🧪 Novelty Check", "📝 Draft Workspace", "📤 Export"])

    # ---------- SUMMARY ----------
    with tabs[0]:
        st.subheader("LLM Summary from Retrieved Evidence")
        if st.button("Generate Summary", key="gen_summary"):
            with st.spinner("Summarizing with LLaMA..."):
                summary = run_chain(P_SUMMARY, topic=topic, context=st.session_state.combined_text)
            st.markdown(summary)

    # ---------- WRITING ASSISTANT ----------
    with tabs[1]:
        st.subheader("Outline & Section Drafting")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Generate Paper Outline"):
                with st.spinner("Drafting outline..."):
                    outline = run_chain(P_OUTLINE, topic=topic, context=st.session_state.combined_text)
                st.markdown(outline)
        with c2:
            section_title = st.text_input("Section Title", value="Introduction")
            notes = st.text_area("Guidance / Notes (optional)", height=120)
            if st.button("Expand Section"):
                with st.spinner("Expanding section..."):
                    section_text = run_chain(
                        P_EXPAND_SECTION,
                        topic=topic,
                        section_title=section_title,
                        notes=notes or "No extra notes.",
                        context=st.session_state.combined_text
                    )
                st.markdown(section_text)
                # Add to draft
                st.session_state.draft += f"\n\n## {section_title}\n\n{section_text}\n"

        st.markdown("---")
        st.markdown("**Paraphrase Helper**")
        para_in = st.text_area("Paste text to paraphrase", height=150, key="para_in")
        if st.button("Paraphrase"):
            with st.spinner("Paraphrasing..."):
                para_out = run_chain(P_PARAPHRASE, text=para_in)
            st.markdown(para_out)

        st.markdown("---")
        st.markdown("**Reviewer-Style Critique**")
        draft_for_review = st.text_area("Paste draft (or it will use current workspace draft)", value=st.session_state.draft, height=200)
        if st.button("Get Reviewer Feedback"):
            with st.spinner("Reviewing..."):
                review = run_chain(P_REVIEWER, topic=topic, draft=draft_for_review, context=st.session_state.combined_text)
            st.markdown(review)

    # ---------- NOVELTY CHECK ----------
    with tabs[2]:
        st.subheader("Novelty Estimation")
        st.markdown("Paste your **abstract / idea** to check overlap with retrieved literature.")
        st.session_state.abstract = st.text_area("Your abstract / idea", value=st.session_state.abstract, height=160)

        if st.button("Compute Simple Overlap Score"):
            # lexical overlap vs top passages
            passages = [p[:1200] for p in st.session_state.combined_text.split("\n\n") if p.strip()]
            topk = top_overlap_passages(st.session_state.abstract, passages, k=5)
            if not topk:
                st.info("No passages to compare.")
            else:
                scores = [round(s*100, 1) for s,_ in topk]
                st.write("Top overlap scores (%):", scores)
                for s, p in topk:
                    st.markdown(f"- **{round(s*100,1)}%** overlap snippet: {p[:300]}...")

        if st.button("LLM Novelty Judgment"):
            ctx_list = []
            # use titles + snippets only for the novelty prompt to reduce noise
            for src in ["britannica","wikipedia","scholar"]:
                for r in st.session_state.results.get(src, [])[:MAX_CONTEXT_ITEMS]:
                    ctx_list.append(f"{r.get('title','')}: {r.get('snippet','')} ({r.get('link','')})")
            novelty_ctx = "\n".join(ctx_list)[:8000]
            with st.spinner("Asking LLaMA for novelty assessment..."):
                raw = run_chain(P_NOVELTY_LLM, abstract=st.session_state.abstract, context=novelty_ctx)
            # try parse JSON
            try:
                data = json.loads(raw.strip().split("```json")[-1].split("```")[0]) if "```json" in raw else json.loads(raw)
            except Exception:
                # fallback: naive extraction
                data = {"score": None, "rationale": raw, "overlapping_sources": []}
            st.json(data)

    # ---------- DRAFT WORKSPACE ----------
    with tabs[3]:
        st.subheader("Your Draft (Markdown)")
        st.session_state.draft = st.text_area("Edit your draft here", value=st.session_state.draft, height=350)
        st.markdown("**Quick Insert**")
        qi_col1, qi_col2, qi_col3 = st.columns(3)
        with qi_col1:
            if st.button("Insert References Section"):
                st.session_state.draft += "\n\n## References\n"
                for i, b in enumerate(st.session_state.bib, 1):
                    # also place plain refs
                    st.session_state.draft += f"- Ref{i}: see BibTeX entry\n"
        with qi_col2:
            if st.button("Insert Methods Skeleton"):
                st.session_state.draft += "\n\n## Method\n- Problem setup\n- Model/Algorithm\n- Training details\n- Baselines\n- Metrics\n"
        with qi_col3:
            if st.button("Insert Limitations"):
                st.session_state.draft += "\n\n## Limitations\n- Data constraints\n- Generalization risks\n- Ethical considerations\n"

        st.success("Edits saved in session.")

    # ---------- EXPORT ----------
    with tabs[4]:
        st.subheader("Export")
        md_bytes = st.session_state.draft.encode("utf-8")
        st.download_button("📥 Download Markdown", data=md_bytes, file_name=f"{topic.replace(' ','_')}_draft.md", mime="text/markdown")

        if st.session_state.bib:
            bib_txt = "\n\n".join(st.session_state.bib)
            st.download_button("📚 Download BibTeX", data=bib_txt.encode("utf-8"), file_name=f"{topic.replace(' ','_')}.bib", mime="text/plain")
        else:
            st.info("No BibTeX entries yet. Generate by running a search.")

else:
    st.info("Start by entering a topic and running the web search.")
