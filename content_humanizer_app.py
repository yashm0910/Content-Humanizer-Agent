
import os
import re
import textwrap
from difflib import SequenceMatcher
from statistics import pstdev
from typing import Any, Dict, List, TypedDict

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, START, END

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.35"))
DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1200"))


class HumanizerState(TypedDict, total=False):
    input_text: str
    normalized_text: str
    analysis: Dict[str, Any]
    rewrite_brief: str
    draft_text: str
    final_text: str
    quality: Dict[str, Any]
    needs_polish: bool
    target_style: str
    model_name: str
    temperature: float
    max_tokens: int


def clean_text(text: str) -> str:
    """Normalize whitespace but preserve paragraph breaks."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    cleaned = []
    for para in paragraphs:
        para = re.sub(r"[ \t]+", " ", para)
        para = re.sub(r"\s+([.,;:!?])", r"\1", para)
        cleaned.append(para.strip())
    return "\n\n".join(cleaned).strip()


def split_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def analyze_text(text: str) -> Dict[str, Any]:
    sentences = split_sentences(text)
    words = tokenize_words(text)

    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_len = (word_count / sentence_count) if sentence_count else 0.0
    sentence_lengths = [len(tokenize_words(s)) for s in sentences]
    sentence_variance = pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    unique_ratio = (len(set(words)) / word_count) if word_count else 0.0
    paragraph_count = len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()])

    transition_terms = [
        "furthermore",
        "moreover",
        "additionally",
        "in conclusion",
        "it is important to note",
        "therefore",
        "thus",
    ]
    transition_hits = {
        t: len(re.findall(rf"\b{re.escape(t)}\b", text, flags=re.I))
        for t in transition_terms
    }
    repetitive_transition_total = sum(v for v in transition_hits.values() if v > 0)
    passive_hint_count = len(re.findall(r"\b(was|were|is|are|been|being)\b\s+\w+ed\b", text, flags=re.I))

    return {
        "characters": len(text),
        "words": word_count,
        "sentences": sentence_count,
        "paragraphs": paragraph_count,
        "avg_sentence_len": round(avg_sentence_len, 2),
        "sentence_variance": round(sentence_variance, 2),
        "unique_word_ratio": round(unique_ratio, 3),
        "passive_hint_count": passive_hint_count,
        "repetitive_transition_total": repetitive_transition_total,
        "transition_hits": transition_hits,
    }


def build_rewrite_brief(analysis: Dict[str, Any], target_style: str) -> str:
    return textwrap.dedent(
        f"""
        Rewrite the provided text into natural, polished writing.

        Target style:
        {target_style}

        Text profile:
        - words: {analysis['words']}
        - sentences: {analysis['sentences']}
        - paragraphs: {analysis['paragraphs']}
        - average sentence length: {analysis['avg_sentence_len']}
        - sentence length variance: {analysis['sentence_variance']}
        - unique word ratio: {analysis['unique_word_ratio']}
        - passive-voice hints: {analysis['passive_hint_count']}
        - repeated transition hits: {analysis['repetitive_transition_total']}

        Constraints:
        - Preserve the original meaning and all technical facts.
        - Do not add new claims, citations, numbers, results, or references.
        - Keep names, formulas, code symbols, table labels, figure labels, and citations unchanged.
        - Improve flow, readability, and paragraph structure.
        - Vary sentence length naturally.
        - Prefer direct, clear academic/formal language.
        - Remove mechanical repetition and overused transitions.
        - Keep the output in the same language as the input.
        - Return only the rewritten text, nothing else.
        """
    ).strip()


def build_polish_brief(analysis: Dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        Second-pass polish only.

        Goal:
        Make the draft sound more natural and readable while keeping the meaning unchanged.

        Known issues to reduce:
        - repetitive phrasing
        - sentence uniformity
        - awkward transitions
        - overly stiff tone

        Draft profile:
        - draft words: {analysis['words']}
        - draft sentences: {analysis['sentences']}
        - draft paragraphs: {analysis['paragraphs']}
        - unique word ratio: {analysis['unique_word_ratio']}

        Rules:
        - Preserve all facts, numbers, and references exactly.
        - Do not introduce new ideas.
        - Improve coherence and sentence rhythm.
        - Return only the revised text.
        """
    ).strip()


def quality_metrics(source: str, draft: str) -> Dict[str, Any]:
    source_clean = clean_text(source)
    draft_clean = clean_text(draft)

    source_words = tokenize_words(source_clean)
    draft_words = tokenize_words(draft_clean)
    draft_sentences = split_sentences(draft_clean)
    draft_lengths = [len(tokenize_words(s)) for s in draft_sentences]

    similarity = SequenceMatcher(None, source_clean, draft_clean).ratio()
    unique_ratio = (len(set(draft_words)) / len(draft_words)) if draft_words else 0.0
    avg_sentence_len = (len(draft_words) / len(draft_sentences)) if draft_sentences else 0.0
    sentence_variance = pstdev(draft_lengths) if len(draft_lengths) > 1 else 0.0

    score = 100.0
    score -= max(0.0, (similarity - 0.25) * 90.0)
    score += min(12.0, unique_ratio * 12.0)
    score += min(8.0, sentence_variance * 1.5)
    if avg_sentence_len > 28:
        score -= min(18.0, (avg_sentence_len - 28) * 0.9)
    if avg_sentence_len < 8 and len(draft_sentences) > 3:
        score -= 8.0
    score = max(0.0, min(100.0, score))

    needs_polish = (
        similarity > 0.80
        or unique_ratio < 0.42
        or avg_sentence_len > 30
        or (sentence_variance < 2.0 and len(draft_sentences) >= 3)
    )

    return {
        "similarity_to_input": round(similarity, 3),
        "lexical_diversity": round(unique_ratio, 3),
        "avg_sentence_len": round(avg_sentence_len, 2),
        "sentence_variance": round(sentence_variance, 2),
        "naturalness_score": round(score, 1),
        "needs_polish": needs_polish,
        "word_delta": len(draft_words) - len(source_words),
    }


@st.cache_resource(show_spinner=False)
def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Set it in your environment or Streamlit secrets.")
    return Groq(api_key=api_key)


def groq_rewrite(
    client: Groq,
    model_name: str,
    prompt: str,
    source_text: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful academic writing assistant. "
                    "Rewrite the input naturally, preserve all facts, and do not add new content."
                ),
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nSOURCE TEXT:\n{source_text}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def node_normalize(state: HumanizerState) -> Dict[str, Any]:
    return {"normalized_text": clean_text(state["input_text"])}


def node_analyze(state: HumanizerState) -> Dict[str, Any]:
    return {"analysis": analyze_text(state["normalized_text"])}


def node_plan(state: HumanizerState) -> Dict[str, Any]:
    return {"rewrite_brief": build_rewrite_brief(state["analysis"], state["target_style"])}


def node_rewrite(state: HumanizerState) -> Dict[str, Any]:
    client = get_client()
    draft = groq_rewrite(
        client=client,
        model_name=state["model_name"],
        prompt=state["rewrite_brief"],
        source_text=state["normalized_text"],
        temperature=state["temperature"],
        max_tokens=state["max_tokens"],
    )
    return {"draft_text": draft}


def node_quality_gate(state: HumanizerState) -> Dict[str, Any]:
    metrics = quality_metrics(state["normalized_text"], state["draft_text"])
    return {
        "quality": metrics,
        "needs_polish": metrics["needs_polish"],
        "final_text": state["draft_text"] if not metrics["needs_polish"] else "",
    }


def node_polish(state: HumanizerState) -> Dict[str, Any]:
    client = get_client()
    polish_brief = build_polish_brief(state["analysis"])
    polished = groq_rewrite(
        client=client,
        model_name=state["model_name"],
        prompt=polish_brief,
        source_text=state["draft_text"],
        temperature=max(0.2, state["temperature"] - 0.05),
        max_tokens=state["max_tokens"],
    )
    return {"final_text": polished}


def node_final_quality(state: HumanizerState) -> Dict[str, Any]:
    final_text = state.get("final_text") or state.get("draft_text") or ""
    return {
        "final_text": final_text,
        "quality": quality_metrics(state["normalized_text"], final_text),
    }


@st.cache_resource(show_spinner=False)
def build_graph():
    graph = StateGraph(HumanizerState)
    graph.add_node("normalize", node_normalize)
    graph.add_node("analyze", node_analyze)
    graph.add_node("plan", node_plan)
    graph.add_node("rewrite", node_rewrite)
    graph.add_node("quality_gate", node_quality_gate)
    graph.add_node("polish", node_polish)
    graph.add_node("final_quality", node_final_quality)

    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "rewrite")
    graph.add_edge("rewrite", "quality_gate")

    def route_after_quality(state: HumanizerState):
        return "polish" if state.get("needs_polish") else END

    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality,
        {"polish": "polish", END: END},
    )
    graph.add_edge("polish", "final_quality")
    graph.add_edge("final_quality", END)

    return graph.compile()


def run_humanizer(
    text: str,
    target_style: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    graph = build_graph()
    return graph.invoke(
        {
            "input_text": text,
            "target_style": target_style,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    )


st.set_page_config(page_title="Content Humanizer Agent", page_icon="✍️", layout="wide")
st.title("Content Humanizer Agent")
st.caption("LangGraph + Groq + Streamlit")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Groq model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05)
    max_tokens = st.number_input("Max tokens", min_value=256, max_value=4096, value=DEFAULT_MAX_TOKENS, step=64)
    target_style = st.selectbox(
        "Writing style",
        ["General formal writing", "Academic paper / research prose", "Report / documentation style"],
        index=1,
    )
    st.markdown(
        """
        **Workflow**
        1. Normalize text
        2. Analyze style signals
        3. Plan rewrite
        4. Rewrite with Groq
        5. Run a quality gate
        6. Polish once more if needed
        """
    )

if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "last_output" not in st.session_state:
    st.session_state.last_output = ""
if "last_quality" not in st.session_state:
    st.session_state.last_quality = {}
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = {}

left, right = st.columns(2)

with left:
    st.subheader("Input")
    input_text = st.text_area(
        "Paste text here",
        value=st.session_state.last_input,
        height=340,
        placeholder="Paste a paragraph or section here...",
        label_visibility="collapsed",
    )
    b1, b2 = st.columns(2)
    with b1:
        humanize_clicked = st.button("Humanize", use_container_width=True)
    with b2:
        again_clicked = st.button("Humanize again", use_container_width=True)

with right:
    st.subheader("Output")
    st.text_area(
        "Rewritten text",
        value=st.session_state.last_output,
        height=340,
        placeholder="Your rewritten text will appear here.",
        label_visibility="collapsed",
    )

if st.session_state.last_output:
    st.subheader("Quality snapshot")
    q = st.session_state.last_quality
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Naturalness score", q.get("naturalness_score", "—"))
    c2.metric("Similarity to input", q.get("similarity_to_input", "—"))
    c3.metric("Lexical diversity", q.get("lexical_diversity", "—"))
    c4.metric("Avg sentence length", q.get("avg_sentence_len", "—"))

    with st.expander("Analysis details", expanded=False):
        st.json(st.session_state.last_analysis)

    with st.expander("Quality details", expanded=False):
        st.json(st.session_state.last_quality)

    st.download_button(
        "Download output as .txt",
        data=st.session_state.last_output,
        file_name="humanized_text.txt",
        mime="text/plain",
        use_container_width=True,
    )

if humanize_clicked or again_clicked:
    source = input_text.strip()
    if again_clicked and st.session_state.last_output.strip():
        source = st.session_state.last_output.strip()

    if not source:
        st.error("Paste some text first.")
    else:
        with st.spinner("Rewriting..."):
            try:
                result = run_humanizer(
                    text=source,
                    target_style=target_style,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=int(max_tokens),
                )
                st.session_state.last_input = source
                st.session_state.last_output = result.get("final_text", "")
                st.session_state.last_quality = result.get("quality", {})
                st.session_state.last_analysis = result.get("analysis", {})
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

st.markdown(
    """
    ---
    **Note:** This tool is for rewriting and readability improvement. It preserves meaning and should be used with proper citations and review for academic work.
    """
)
