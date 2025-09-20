# 🚀 ResumeMatch — A Dual-Role AI-Driven Platform for Transparent Resume-Job Compatibility and Enhanced Hiring Outcomes

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Prototype-red)](https://streamlit.io/)
[![spaCy](https://img.shields.io/badge/spaCy-Transformer%20NER-09b)](https://spacy.io/)
[![HuggingFace](https://img.shields.io/badge/HF-Models-yellow)](https://huggingface.co/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash%20Lite-black)](https://ai.google.dev/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> ResumeMatch goes beyond keyword ATS by combining **structured NER extraction**, **semantic embeddings**, and a **Streamlit** UI with **Gemini**-powered interview questions.

## 🔎 Problem & Motivation
Traditional Applicant Tracking Systems (ATS) rely on **keyword filters**. They miss context, penalize formatting, and overlook **transferable skills**—leading to **misaligned matches** and **inefficient screening**.

**ResumeMatch** addresses this by:
- Extracting structured entities (**Skills, Education, Experience**) from JDs and resumes,
- Matching candidates and jobs with **semantic similarity models**,
- Providing **explainable** recruiter tools: ranking, clustering, and skill-gap analysis,
- Generating **tailored interview questions** with Gemini.

## 🧪 Methodology (Research Pipeline)

### 1) Job Description Parsing (10 methods, 3 groups)
- **Skills/Embeddings-based:** basic keywords, YAKE, spaCy PhraseMatcher, embedding similarity  
- **Model-based:** multiple **pre-trained NER** models (Hugging Face) + **LLM-based extraction** (using Prompt engineering)  
- **Transformer NER (custom):** **spaCy** transformer model trained on **150 annotated JDs** for Skills/Education/Experience

### 2) Resume Parsing
- Compared **PyResparser**, **Custom transformer NER**, and **LLM-based** extraction

### 3) Resume–Job Matching
- Benchmarked **TF-IDF**, **LDA**, **Word2Vec**, and **BERT** embeddings  
- Agreement & ranking metrics + **cluster visualization** (TF-IDF, LDA, Word2Vec and BERT)

## 📈 Results
- **Custom NER** → **Best for Skills extraction** (Highest precision/F1 & Jaccard)  
- **LLM parser** → **Best for Education**; **higher recall on Experience** (NER led on Experience F1/Jaccard)  
- **LDA** → **Best ranking separation** (Separation@K)  
- **BERT** → **Best reciprocal agreement & mutual rank;** clearest clusters in PCA  
- **Prototype** → Demonstrated **candidate ranking**, **skill-gap analysis**, and **interview question support**

> `Docs/clustering.png` (LDA vs BERT) for a quick visual of cluster separation.
<img width="967" height="717" alt="image" src="https://github.com/user-attachments/assets/01debaff-36e6-42cd-829f-0510a96e09a3" />

## 🧰 Streamlit Prototype (Product View)

The app is designed for **two audiences**:

### 🧑 Job Seeker View
- Upload one resume and up to five job descriptions.
- Automatic **entity extraction** (hard skills, soft skills, education, experience).
- **Match scores** (semantic / exact / partial) across sections with interactive donut and bar charts.
- **Skill-gap analysis**: see overlaps and missing skills per JD.
- **LLM-powered interview prep**: generate tailored questions based on job description.

### 🏢 Recruiter View
- Upload one job description and multiple candidate resumes.
- **Clustering**: resumes are grouped using **PCA-reduced BERT embeddings + K-Means**.
- The JD is matched to the **closest cluster centroid** (highest similarity).
- **Ranking**: within the cluster, candidates are ranked by weighted semantic similarity (skills, education, experience).
- **Candidate scorecards**:
  - Overall match % with light indicators (🟢/🟡/🔴).
  - Section-wise breakdown (skills, education, experience).
  - Top missing skills (hard + soft).
  - Overlaps & gaps via interactive tables.

## ⚙️ Installation

### 1) Clone & Environment
```bash
git clone https://github.com/rabiadanish/ResumeMatch.git
cd ResumeMatch
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
# Install the required dependencies
pip install -r requirements.txt
# Run Streamlit app
streamlit run app/app.py
```
### 2) Gemini API Configuration
To configure:

1. Sign up or log in to your Gemini account to obtain your **API key**.
2. In your environment (local `.env` or Colab), set the key:

or in Python:
```python
import os
os.environ["GEMINI_API_KEY"] = "your_api_key"
```
3. ResumeMatch automatically reads this key when generating interview questions.
4. Ensure your network allows outbound API calls to Gemini.

## ▶️ Demo Video
[🎥 Watch ResumeMatch Demo](https://github.com/user-attachments/assets/16c9ce12-f0da-425f-9f1a-7fab65e3b5f5)

## 🔐 Data & Privacy

- No personal/proprietary resumes are included.
- Use data/sample_resumes/ and data/sample_jobs for local demo.
- Kaggle datasets: link in data/README.md.
