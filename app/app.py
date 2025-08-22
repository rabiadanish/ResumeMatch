import streamlit as st
from job_seeker import job_seeker_tab
from recruiter import recruiter_tab

# -------------------------
# UI Layout & Logic
# -------------------------
st.set_page_config(page_title="ResumeMatch", layout="wide")
st.title("ResumeMatch — AI-powered Job ↔️ Resume Matching")
st.sidebar.markdown("### 🔍 Top Match Score")
score_placeholder = st.sidebar.empty()
st.sidebar.header("Matching Configuration")
st.sidebar.write("Tune how scores are computed:")
tab1, tab2 = st.tabs(["👩🏼‍💻 **Job Seeker**", "🏢 **Recruiter**"])

# Sidebar weights (kept UI identical but keys will be normalized below)
w_h = st.sidebar.slider("Hard Skills weight", 0.0, 1.0, 0.4, 0.05)
w_s = st.sidebar.slider("Soft Skills weight", 0.0, 1.0, 0.2, 0.05)
w_eu = st.sidebar.slider("Education weight", 0.0, 1.0, 0.2, 0.05)
w_ex = st.sidebar.slider("Experience weight", 0.0, 1.0, 0.2, 0.05)

# Use consistent keys for internal computation
section_weights = {
    "hard_skills": w_h,
    "soft_skills": w_s,
    "education": w_eu,
    "experience": w_ex
}

threshold_slider = st.sidebar.slider(
    "Similarity highlight threshold (RapidFuzz %)",
    min_value=50,
    max_value=100,
    value=80
)

# --- Sidebar Quick Link ---
with st.sidebar:
    st.markdown("### Quick Links")
    # Clickable link to README (hosted on GitHub or local file URL)
    st.markdown(
        "[📄 View README](https://github.com/rabiadanish/ResumeMatch/blob/main/README.md)",
        unsafe_allow_html=True
    )

with tab1:
    job_seeker_tab(section_weights=section_weights,
                   threshold_slider=threshold_slider,
                   score_placeholder=score_placeholder)

with tab2:
     recruiter_tab(section_weights=section_weights,
                   threshold_slider=threshold_slider,
                   score_placeholder=score_placeholder)