# recruiter.py

import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from utils import (
    ner_model,
    embed_model,
    load_file_contents,
    preprocess_resume_text,
    extract_entities_with_spacy,
    parse_and_normalize,
    compute_match_score,
    compute_section_score,
    overlaps_and_gaps,
    section_bars,
    plot_donut_score,
    colored_progress_bar,
    get_top_n_skills,
    is_skill_matched_by_resume_single,      
    vector_from_entities,         
)

SECTIONS = ["hard_skills", "soft_skills", "education", "experience"]
def _pretty(s: str) -> str:
    return s.replace("_", " ").title()

def recruiter_tab(section_weights, threshold_slider, score_placeholder):
    st.header("Find Your Perfect Candidate Matches!")
    st.markdown(
        "Upload a **Job Description** and  **multiple resume** files. "
        "We'll cluster similar resumes, match the JD to the best cluster, and then rank candidates."
    )

    # --- Uploads ---
    col_left, col_right = st.columns(2)
    with col_left:
        jd_file = st.file_uploader("📄 Upload Job Description", type=["pdf", "docx", "txt"], key="rec_jd")
    with col_right:
        resumes = st.file_uploader(
            "👥 Upload Candidate Resumes (multi-select supported)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="rec_resumes_folder",
        )

    # Guardrails & model checks
    if jd_file is None or not resumes:
        st.info("Upload a job description and at least one resume to proceed.")
        return
    if embed_model is None:
        st.warning("Embedding model not loaded; semantic similarity won’t work.")
    if ner_model is None:
        st.warning("spaCy NER model not loaded; entity extraction may be empty.")

    # --- JD extraction ---
    jd_raw_text = load_file_contents(jd_file)
    jd_text = preprocess_resume_text(jd_raw_text)
    jd_entities_raw = extract_entities_with_spacy(jd_text) if ner_model is not None else {k: [] for k in SECTIONS}
    jd_entities_norm = {k: parse_and_normalize(v) for k, v in jd_entities_raw.items()}
    jd_vec = vector_from_entities(jd_entities_norm)

    # Precompute JD's "important skills" by frequency in JD text (for missing-skill display)
    TOP_N_IMPORTANT = 10
    jd_top_hard_skills = get_top_n_skills(jd_text, jd_entities_raw.get("hard_skills", []), n=TOP_N_IMPORTANT)
    jd_top_soft_skills = get_top_n_skills(jd_text, jd_entities_raw.get("soft_skills", []), n=TOP_N_IMPORTANT)

    # --- Candidate extraction ---
    candidates = []
    progress = st.progress(0, text="Parsing resumes...")
    for idx, rf in enumerate(resumes):
        txt_raw = load_file_contents(rf)
        txt = preprocess_resume_text(txt_raw)
        ents_raw = extract_entities_with_spacy(txt) if ner_model is not None else {k: [] for k in SECTIONS}
        ents_norm = {k: parse_and_normalize(v) for k, v in ents_raw.items()}
        vec = vector_from_entities(ents_norm)  # (1, D)

        candidates.append(
            {
                "name": rf.name,
                "text": txt,
                "ents_raw": ents_raw,
                "ents_norm": ents_norm,
                "vec": vec,  # (1, D)
            }
        )
        progress.progress((idx + 1) / len(resumes), text=f"Parsing {rf.name}...")
    progress.empty()

    # --- Clustering step ---
    st.subheader("Resume Clustering")
    n_resumes = len(candidates)

    if n_resumes >= 2 and embed_model is not None:
        max_clusters = min(15, n_resumes)
        n_clusters = st.slider(
            "Number of clusters (KMeans)",
            min_value=2,
            max_value=max_clusters,
            value=min(5, max_clusters),
        )

        # Stack vectors into (N, D)
        X = np.vstack([c["vec"] for c in candidates])  # each vec is (1, D)

        # --- NEW: Dimensionality reduction with PCA ---
               
        max_n_components = min(50, X.shape[0], X.shape[1])
        pca = PCA(n_components=max_n_components, random_state=42)
        X_reduced = pca.fit_transform(X)

        # --- MiniBatchKMeans on reduced embeddings ---
        
        km = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init="auto"
        )
        labels = km.fit_predict(X_reduced)

        # Assign cluster labels
        for i, lbl in enumerate(labels):
            candidates[i]["cluster"] = int(lbl)

        # Centroids (n_clusters, reduced_D)
        centroids = km.cluster_centers_

        # --- Compute similarity JD -> cluster centroids ---
        # Project jd_vec into PCA space
        jd_vec_reduced = pca.transform(jd_vec)
        sim_to_clusters = cosine_similarity(jd_vec_reduced, centroids)[0]  # shape: (n_clusters,)

        cluster_df = (
            pd.DataFrame(
                {
                    "Cluster": list(range(n_clusters)),
                    "Members": [sum(1 for c in candidates if c["cluster"] == k) for k in range(n_clusters)],
                    "JD→Cluster Similarity": [float(s) for s in sim_to_clusters],
                }
            )
            .sort_values("JD→Cluster Similarity", ascending=False)
            .reset_index(drop=True)
        )

        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

        # Suggested best cluster
        best_cluster = int(cluster_df.loc[0, "Cluster"])
        st.caption(f"Suggested cluster: **{best_cluster}** (highest JD similarity)")

        # Let recruiter pick the cluster to review
        selected_cluster = st.selectbox(
            "Choose a cluster to review",
            options=list(range(n_clusters)),
            index=list(range(n_clusters)).index(best_cluster),
            key="rec_cluster_pick",
        )

        # Filter down to selected cluster
        review_candidates = [c for c in candidates if c.get("cluster", -1) == selected_cluster]

    else:
        # Not enough resumes to cluster or no embed model
        selected_cluster = None
        for c in candidates:
            c["cluster"] = 0
        review_candidates = candidates

    st.write(
        f"**Reviewing {len(review_candidates)} resume(s)**"
        + (f" from cluster **{selected_cluster}**." if selected_cluster is not None else ".")
    )

    if not review_candidates:
        st.warning("No candidates in the selected cluster.")
        return

    # --- Rank candidates (semantic baseline) for the selected cluster ---
    semantic_results = []
    for idx, cand in enumerate(review_candidates):
        s = compute_match_score(cand["ents_norm"], jd_entities_norm, embed_model, section_weights)
        semantic_results.append({"cand_idx": idx, "candidate": cand["name"], "scores": s})

    # Build ranking table
    rows = []
    for r in semantic_results:
        s = r["scores"]
        final_pct = int(round(s["final"] * 100))
        rows.append(
            {
                "Candidate": r["candidate"],
                "Hard Skills (%)": int(round(s["hard_skills"] * 100)),
                "Soft Skills (%)": int(round(s["soft_skills"] * 100)),
                "Education (%)": int(round(s["education"] * 100)),
                "Experience (%)": int(round(s["experience"] * 100)),
                "Final Match (%)": final_pct,
                "Status": "🔴" if final_pct < 55 else ("🟡" if final_pct < 80 else "🟢"),
            }
        )

    df = pd.DataFrame(rows).sort_values("Final Match (%)", ascending=False).reset_index(drop=True)

    # Sidebar/top donut - top candidate (unique key)
    if not df.empty:
        top_score = int(df.loc[0, "Final Match (%)"])
        donut_fig = plot_donut_score(top_score)
        donut_key = f"recruiter_top_donut_cluster_{selected_cluster if selected_cluster is not None else 0}"
        score_placeholder.plotly_chart(donut_fig, use_container_width=True, key=donut_key)

    # Ranking table
    st.subheader("Candidate Ranking")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Final Match (%)": st.column_config.ProgressColumn(
                label="Final Match (%)",
                format="%d%%",
                min_value=0,
                max_value=100,
            )
        },
    )

    # --- Candidate Score Cards ---
    st.subheader("Candidate Score Cards")
    grid = st.columns(3)  # 3 cards per row

    # Precompute a name - candidate map for quick lookups
    name_to_cand = {c["name"]: c for c in review_candidates}

    for rank, r in enumerate(df.itertuples(index=False)):
        cand_name = r.Candidate
        cand = name_to_cand[cand_name]

        # Find semantic result for this candidate
        res_obj = next(x for x in semantic_results if x["candidate"] == cand_name)
        idx_local = res_obj["cand_idx"]  # index inside review_candidates
        sem_scores = res_obj["scores"]
        overall = int(round(sem_scores["final"] * 100))

        card = grid[rank % 3]
        with card.container(border=True):
            st.markdown(f"### {cand_name}")
            st.caption(f"Cluster: **{cand['cluster']}**")
            st.markdown("**Overall Match:**", unsafe_allow_html=True)
            st.markdown(colored_progress_bar(overall), unsafe_allow_html=True)

            # Mode toggle (affects the mini section bars only)
            mode_choice = st.radio(
                "Matching Mode",
                ["Semantic Match", "Exact Match", "Partial Match"],
                horizontal=True,
                key=f"rec_mode_{cand_name}_{rank}",
            )

            # Compute section bars depending on the mode
            if mode_choice == "Semantic Match":
                sec_scores_float = {
                    "hard_skills": float(sem_scores["hard_skills"]),
                    "soft_skills": float(sem_scores["soft_skills"]),
                    "education": float(sem_scores["education"]),
                    "experience": float(sem_scores["experience"]),
                }
            elif mode_choice == "Exact Match":
                sec_scores_float = {}
                for sec in SECTIONS:
                    sec_scores_float[sec] = compute_section_score(
                        cand["ents_norm"].get(sec, []),
                        jd_entities_norm.get(sec, []),
                        threshold=threshold_slider,
                        mode="raw",
                    )
            else:  # Partial Match (fuzzy)
                sec_scores_float = {}
                for sec in SECTIONS:
                    sec_scores_float[sec] = compute_section_score(
                        cand["ents_norm"].get(sec, []),
                        jd_entities_norm.get(sec, []),
                        threshold=threshold_slider,
                        mode="fuzzy",
                    )

            st.plotly_chart(
                section_bars(sec_scores_float, title=None),
                use_container_width=True,
                key=f"rec_bars_{cand_name}_{rank}_{mode_choice}",
            )

            # ---------- Top 3 Missing Skills (JD-priority, dynamic via threshold slider) ----------
            # We are computing missing skills using threshold_slider (fuzzy/partial logic inside is_skill_matched_by_resume_single).
            top_n = 3

            # HARD skills
            missing_hard = []
            res_hard_norm = cand["ents_norm"].get("hard_skills", [])
            for skill in jd_top_hard_skills:
                if len(missing_hard) >= top_n:
                    break
                if not is_skill_matched_by_resume_single(skill, res_hard_norm, threshold_slider):
                    missing_hard.append(skill)

            # SOFT skills
            missing_soft = []
            res_soft_norm = cand["ents_norm"].get("soft_skills", [])
            for skill in jd_top_soft_skills:
                if len(missing_soft) >= top_n:
                    break
                if not is_skill_matched_by_resume_single(skill, res_soft_norm, threshold_slider):
                    missing_soft.append(skill)

            st.markdown("**Top Missing Skills (from JD):**")
            st.write(f"- Hard Skills: {', '.join(missing_hard) if missing_hard else 'None'}")
            st.write(f"- Soft Skills: {', '.join(missing_soft) if missing_soft else 'None'}")

            # ---------- Detailed gaps & overlaps (tabs) ----------
            with st.expander("Show Detailed Gaps & Overlaps", expanded=False):
                tabs = st.tabs([_pretty(s) for s in SECTIONS])
                for t, sec in zip(tabs, SECTIONS):
                    with t:
                        res_list = cand["ents_norm"].get(sec, [])
                        jd_list = jd_entities_norm.get(sec, [])
                        overlaps, gaps = overlaps_and_gaps(res_list, jd_list, threshold_slider)

                        st.markdown(f"**{_pretty(sec)}**")
                        if overlaps:
                            df_over = pd.DataFrame(overlaps, columns=["Resume Item", "JD Item", "Score"])
                            df_over["Score"] = df_over["Score"].astype(int)
                            st.write("Overlaps")
                            st.dataframe(df_over, use_container_width=True, hide_index=True)
                        else:
                            st.write("Overlaps: —")

                        if gaps:
                            st.write("Gaps (JD items not matched by resume)")
                            st.markdown("\n".join([f"- {g}" for g in gaps]))
                        else:
                            st.write("Gaps: —")
