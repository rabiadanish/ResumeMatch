# utils.py

import streamlit as st
from io import BytesIO
import tempfile
import PyPDF2
import os
import re
import docx
from typing import List, Dict, Tuple
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import numpy as np
import ast
import spacy


# -------------------------
# Model loading (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_ner_model(path: str = None):
    if path is None:
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, "model-ner")
    try:
        return spacy.load(path)
    except Exception as e:
        st.error(f"Error loading NER model: {e}")

@st.cache_resource(show_spinner=False)
def load_embed_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"Embedding model failed to load: {e}")
        return None

ner_model = load_ner_model()
embed_model = load_embed_model()

# lightweight tokenizer/lemmatizer for parse_and_normalize
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # Minimal fallback (no lemmatizer)
    nlp = spacy.blank("en")

# -------------------------
# File reading helpers
# -------------------------
def read_pdf_bytes(data: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(BytesIO(data))
        text_pages = []
        for p in reader.pages:
            text_pages.append(p.extract_text() or "")
        return "\n".join(text_pages)
    except Exception:
        return ""

def read_docx_bytes(data: bytes) -> str:
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
        doc = docx.Document(tmp_path)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)
    except Exception:
        return ""
    finally:
        try:
            if tmp is not None:
                os.unlink(tmp.name)
        except Exception:
            pass

def read_text_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return str(data)

def load_file_contents(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        return read_pdf_bytes(data)
    if name.endswith(".docx") or name.endswith(".doc"):
        return read_docx_bytes(data)
    if name.endswith(".txt"):
        return read_text_bytes(data)
    # fallback
    return read_text_bytes(data)

# -------------------------
# Text preprocessing & normalization
# -------------------------
def preprocess_resume_text(raw_text):
    """
    Cleans and preprocesses raw text, but avoids breaking hyphenated words.
    - Removes ISO dates and page markers
    - Normalizes bullets to a single hyphen only when they appear as standalone list markers
    - Collapses excessive whitespace while preserving single newlines
    """
    if not raw_text:
        return ""

    text = raw_text

    # Remove ISO dates like 2023-08-11
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", text, flags=re.IGNORECASE)

    # Remove "Page 1", "Page 2" etc.
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)

    # Replace common bullet characters ONLY when they act as list markers at line start
    bullet_pat = r"(?m)^\s*[•\u2022\u25CF\-\*\uf0b7]\s+"
    text = re.sub(bullet_pat, "- ", text)

    # Remove stray emojis/symbol bullets that sometimes sneak in mid-line
    text = re.sub(r"[•\u2022\u25CF\uf0b7]+", " ", text)

    # Normalize spaces (but keep single newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\w)\s+(?=\w)", " ", text)

    return text.strip()

def parse_and_normalize(x):
    # Using the exact function you provided (lowercasing + lemmatizing + dedupe)
    if isinstance(x, list):
        items = x
    elif isinstance(x, str):
        try:
            items = ast.literal_eval(x)
        except Exception:
            items = [item.strip() for item in x.split(',') if item.strip()]
    else:
        items = []

    normalized = []
    for s in items:
        s = re.sub(r"\s*-\s*", "-", s.strip().lower())
        s = re.sub(r"\s+", " ", s.strip(" ."))
        doc = nlp(s)
        lemmatized = " ".join([token.lemma_ for token in doc if not token.is_punct])
        parts = re.split(r"\s+or\s+|\s+and\s+", lemmatized)
        for part in parts:
            if part.strip():
                normalized.append(part.strip())

    return list(dict.fromkeys(normalized))

# -------------------------
# Entity extraction (cached)
# -------------------------
@st.cache_data
def extract_entities_with_spacy(text: str) -> Dict[str, List[str]]:
    """Extracts custom NER entities from text using the SpaCy pipeline.
    Expects labels: hard_skill, soft_skill, education, experience
    Returns keys: 'hard_skills','soft_skills','education','experience' (deduped)
    """
    if not text:
        return {
            "hard_skills": [],
            "soft_skills": [],
            "education": [],
            "experience": []
        }

    doc = ner_model(text)
    entities = {
        "hard_skills": [],
        "soft_skills": [],
        "education": [],
        "experience": []
    }

    for ent in doc.ents:
        lbl = ent.label_.lower()
        if lbl == "hard_skill":
            entities["hard_skills"].append(ent.text.strip())
        elif lbl == "soft_skill":
            entities["soft_skills"].append(ent.text.strip())
        elif lbl == "education":
            entities["education"].append(ent.text.strip())
        elif lbl == "experience":
            entities["experience"].append(ent.text.strip())

    return {k: list(dict.fromkeys(v)) for k, v in entities.items()}

# -------------------------
# Embedding helpers
# -------------------------
def aggregate_embeddings(items: List[str], embedder: SentenceTransformer) -> np.ndarray:
    """
    Returns 1 x D vector (mean) for list of item strings. If empty, return zero vector of shape (1,D)
    """
    dim = embedder.get_sentence_embedding_dimension()
    if not items:
        return np.zeros((1, dim))
    emb = embedder.encode(items, convert_to_numpy=True, show_progress_bar=False)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb.mean(axis=0, keepdims=True)

def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    try:
        if a is None or b is None:
            return 0.0
        # ensure shapes are (1, D)
        a = np.array(a)
        b = np.array(b)
        if a.size == 0 or b.size == 0:
            return 0.0
        # if norms zero, return 0
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(cosine_similarity(a, b)[0, 0])
    except Exception:
        return 0.0

# -------------------------
# Match scoring
# -------------------------
def compute_match_score(res_entities: Dict[str, List[str]],
                        jd_entities: Dict[str, List[str]],
                        embedder: SentenceTransformer,
                        weights: Dict[str, float]) -> Dict[str, float]:
    """
    Compute per-section similarity (cosine) between the resume and JD entity sets, then weighted final.
    Returns dict: {'hard_skills':, 'soft_skills':, 'education':, 'experience':, 'final':}
    """
    s = {}
    # per-section embeddings (res_entities and jd_entities MUST be normalized lists)
    res_h = aggregate_embeddings(res_entities.get("hard_skills", []), embedder)
    jd_h = aggregate_embeddings(jd_entities.get("hard_skills", []), embedder)
    s["hard_skills"] = safe_cosine(res_h, jd_h)

    res_s = aggregate_embeddings(res_entities.get("soft_skills", []), embedder)
    jd_s = aggregate_embeddings(jd_entities.get("soft_skills", []), embedder)
    s["soft_skills"] = safe_cosine(res_s, jd_s)

    res_e = aggregate_embeddings(res_entities.get("education", []), embedder)
    jd_e = aggregate_embeddings(jd_entities.get("education", []), embedder)
    s["education"] = safe_cosine(res_e, jd_e)

    res_x = aggregate_embeddings(res_entities.get("experience", []), embedder)
    jd_x = aggregate_embeddings(jd_entities.get("experience", []), embedder)
    s["experience"] = safe_cosine(res_x, jd_x)

    # Normalize weights so they sum to 1 (if all zero -> equal weights)
    total_w = sum(weights.values())
    if total_w == 0:
        norm_w = {k: 1.0 / len(weights) for k in weights}
    else:
        norm_w = {k: (v / total_w) for k, v in weights.items()}

    final = (s["hard_skills"] * norm_w.get("hard_skills", 0.0) +
             s["soft_skills"] * norm_w.get("soft_skills", 0.0) +
             s["education"] * norm_w.get("education", 0.0) +
             s["experience"] * norm_w.get("experience", 0.0))
    s["final"] = float(final)
    return s

# -------------------------
# Raw vs Fuzzy Section Score
# -------------------------
def compute_section_score(res_items: list, jd_items: list, threshold: int = 80, mode: str = "fuzzy") -> float:
    """
    Computes a section-level score between resume items and JD items.
    
    Parameters:
    - res_items: list of resume entities (normalized or original)
    - jd_items: list of JD entities (normalized or original)
    - threshold: for fuzzy matching (0-100)
    - mode: "raw" for strict keyword match, "fuzzy" for fuzzy Partial match
    
    Returns:
    - score: 0-1 float
    """
    if not res_items or not jd_items:
        return 0.0

    matched = 0
    if mode == "raw":
        # Exact match: case-insensitive
        res_items_orig = [r.lower() for r in res_items]
        jd_items_orig = [j.lower() for j in jd_items]
        for r in res_items_orig:
            if r in jd_items_orig:
                matched += 1
    elif mode == "fuzzy":
        for r in res_items:
            best = process.extractOne(r, jd_items, scorer=fuzz.token_sort_ratio)
            if best and best[1] >= threshold:
                matched += 1
    else:
        raise ValueError("mode must be 'Exact Match' or 'Partial Match'")

    score = matched / len(jd_items)  # % of JD items covered by resume
    return float(score)

# -------------------------
# RapidFuzz overlap/gap analysis (case-insensitive due to normalized inputs)
# -------------------------
def overlaps_and_gaps(res_list: List[str], jd_list: List[str], threshold: int = 80):
    """
    Returns overlaps: list of (resume_item, jd_item, score)
            gaps (jd items not matched)
    Both lists are expected to be normalized (lowercased, lemmatized) for robust matching.
    """
    overlaps = []
    remaining_jd = set(jd_list)
    if not jd_list:
        return overlaps, []
    # iterate resume items and find best JD match
    for r in res_list:
        best = process.extractOne(r, jd_list, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= threshold:
            overlaps.append((r, best[0], best[1]))
            if best[0] in remaining_jd:
                remaining_jd.remove(best[0])
    gaps = list(remaining_jd)
    return overlaps, gaps

# -----------------
# JD Top-N Skills
# -----------------
def get_top_n_skills(jd_text, jd_items, n=10):
    """Return the top n skills from jd_items ranked by frequency in jd_text."""
    if not jd_text or not jd_items:
        return []

    text_lower = jd_text.lower()
    counts = Counter()

    for skill in jd_items:
        if not skill:
            continue
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        freq = len(re.findall(pattern, text_lower))
        if freq > 0:
            counts[skill] = freq

    # Sort by frequency then alphabetically for stability
    return [skill for skill, _ in counts.most_common(n)]

def is_skill_matched_by_resume_single(skill_raw: str, resume_norm_list: list, threshold: int) -> bool:
    """
    Determine if a single JD skill (skill_raw) is matched by ANY item in resume_norm_list.
    - resume_norm_list: list of normalized candidate items (lowercased & lemmatized strings)
    - We normalize the JD skill (via parse_and_normalize) and compare each normalized JD part
      to resume items using:
        1) direct normalized equality
        2) fuzzy token_sort_ratio >= threshold
    """
    if not skill_raw:
        return False
    if not resume_norm_list:
        return False

    # Normalize JD skill into parts using same parse_and_normalize logic
    jd_norm_parts = parse_and_normalize([skill_raw])  # returns list of normalized parts
    if not jd_norm_parts:
        # fallback to direct lowercase
        target = skill_raw.strip().lower()
        # exact match
        if any(target == r.lower() for r in resume_norm_list):
            return True
        # fuzzy match
        best = process.extractOne(target, resume_norm_list, scorer=fuzz.token_sort_ratio)
        return bool(best and best[1] >= threshold)

    # Check each normalized part for a match in resume items
    for jp in jd_norm_parts:
        # exact normalized match
        if any(jp == r.lower() for r in resume_norm_list):
            return True
        # fuzzy match against resume items
        best = process.extractOne(jp, resume_norm_list, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= threshold:
            return True
    return False

# -------------------------
# Visualization helpers
# -------------------------
def section_bars(section_scores: dict, title=None):
    """
    Plots horizontal bar chart for section-level scores.
    
    section_scores: dict with keys like 'hard_skills', 'soft_skills', etc. and values 0-1 float
    title: optional chart title
    """
    sections = list(section_scores.keys())
    scores = [int(round(section_scores[s]*100)) for s in sections]

    fig = go.Figure(go.Bar(
        x=scores,
        y=[s.replace('_',' ').title() for s in sections],
        orientation='h',
        text=[f"{v}%" for v in scores],
        textposition='auto',
        marker=dict(color=["#FF3377", "#3B00B3", "#009919", "#E6BF00"],
                    line=dict(color=["#FF3377", "#3B00B3", "#009919", "#E6BF00"], width=1)),
        hovertemplate='%{y}: %{x:.0f}%'
    ))

    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=200, xaxis=dict(range=[0,100]))
    if title:
        fig.update_layout(title=title, title_x=0.5, title_font_size=12)
    return fig

def plot_donut_score(pct: float, size: int = 150):
    val = float(pct)
    color = "red" if val < 55 else ("gold" if val < 80 else "green")
    val = max(0.0, min(val, 100.0))
    fig = go.Figure(data=[go.Pie(
        values=[val, 100.0 - val],
        labels=["Match", "Gap"],
        hole=0.6,
        marker=dict(colors=[color, "lightgray"]),
        textinfo="none"
    )])
    fig.update_layout(
        width=size, height=size, margin=dict(t=0, b=0, l=0, r=0),
        annotations=[dict(text=f"{int(round(val))}%", x=0.5, y=0.5, font=dict(size=14), showarrow=False)]
    )
    return fig

def colored_progress_bar(pct):
    """
    Renders a colored progress bar in HTML (red <55%, yellow <80%, green >=80%).
    """
    if pct < 55:
        color = "#FF4B4B"  # red
    elif pct < 80:
        color = "#FFD93D"  # yellow
    else:
        color = "#4CAF50"  # green

    return f"""
    <div style="background-color: #e0e0e0; border-radius: 8px; height: 20px; width: 100%;">
        <div style="background-color: {color}; width: {pct}%; height: 100%;
                    border-radius: 8px; text-align: center; color: white;
                    font-size: 12px; line-height: 20px;">
            {pct:.0f}%
        </div>
    </div>
    """

# --------------------------
# Resume Clustering helpers
# --------------------------
def concat_entities(entities_norm: dict) -> list:
    """Combine all normalized entities (hard, soft, education, experience) into a single list."""
    items = []
    SECTIONS = ["hard_skills", "soft_skills", "education", "experience"]
    for k in SECTIONS:
        items.extend(entities_norm.get(k, []))
    # dedupe while preserving order
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def vector_from_entities(entities_norm: dict) -> np.ndarray:
    """
    Build a single (1, D) embedding for a resume/JD by aggregating embeddings
    of all normalized entity strings (across sections).
    """
    combined = concat_entities(entities_norm)
    # aggregate_embeddings returns shape (1, D)
    return aggregate_embeddings(combined, embed_model)