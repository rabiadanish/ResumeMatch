# job_seeker.py

import streamlit as st
import pandas as pd
import math
from utils import (
    ner_model,
    embed_model,
    load_file_contents,
    preprocess_resume_text,
    extract_entities_with_spacy,
    parse_and_normalize,
    compute_match_score,
    plot_donut_score,
    section_bars,
    overlaps_and_gaps,
    compute_section_score,
)
from dotenv import load_dotenv
load_dotenv()
from questions_llm import generate_interview_questions

def job_seeker_tab(section_weights, threshold_slider, score_placeholder):
    MAX_JDS = 5
    st.header("Your Perfect Matches Await!")
    st.markdown("Upload your resume and up to 5 job descriptions to see how you stack up.")

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader(
            "👤 Upload Your Resume",
            type=['pdf', 'docx', 'txt'],
            key="resume_js"
        )
    with col2:
        jd_files = st.file_uploader(
            "📄 Upload Job Descriptions (up to 5)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            key="jds_js"
        )
    if jd_files and len(jd_files) > MAX_JDS:
        st.warning(f"Only first {MAX_JDS} job descriptions will be used.")
        jd_files = jd_files[:MAX_JDS]
    if jd_files is None or not resume_file:
        st.info("Upload your resume and at least one job description to discover your best job match.")    
    if resume_file is not None and ner_model is None:
        st.warning("spaCy NER model not loaded; cannot extract entities. Please provide a valid model path.")
    if resume_file and embed_model is None:
        st.warning("Embedder not loaded; cannot compute similarity.")

    if resume_file and jd_files:
        raw_resume = load_file_contents(resume_file)
        resume_text = preprocess_resume_text(raw_resume)

        resume_entities_raw = extract_entities_with_spacy(resume_text) if ner_model else {
            "hard_skills": [], "soft_skills": [], "education": [], "experience": []
        }
        resume_entities = {k: parse_and_normalize(v) for k, v in resume_entities_raw.items()}

        jd_names, jd_entities_list_raw, jd_texts, jd_entities_list = [], [], [], []

        for jd in jd_files:
            raw_jd = load_file_contents(jd)
            jd_text = preprocess_resume_text(raw_jd)
            jd_names.append(jd.name)
            jd_texts.append(jd_text)
            entities_raw = extract_entities_with_spacy(jd_text) if ner_model else {
                "hard_skills": [], "soft_skills": [], "education": [], "experience": []
            }
            jd_entities_list_raw.append(entities_raw)
            jd_entities_list.append({k: parse_and_normalize(v) for k, v in entities_raw.items()})

        # Resume & JD expanders
        col1, col2 = st.columns(2)
        with col1.expander("### **📄 Resume Entities (extracted) 📄**", expanded=False):
            for section in ["hard_skills","soft_skills","education","experience"]:
                items = resume_entities_raw.get(section, [])
                numbered = "\n".join(f"{i+1}. {e}" for i,e in enumerate(items)) or "—"
                st.markdown(f"**{section.replace('_',' ').title()}**\n\n{numbered}")

        with col2:
            for name, ents in zip(jd_names, jd_entities_list_raw):
                with st.expander(f"### **📄 {name} Entities (extracted) 📄**", expanded=False):
                    for section in ["hard_skills","soft_skills","education","experience"]:
                        items = ents.get(section, [])
                        numbered = "\n".join(f"{i+1}. {e}" for i,e in enumerate(items)) or "—"
                        st.markdown(f"**{section.replace('_',' ').title()}**\n\n{numbered}")

        # Compute match scores
        results = []
        for i, jd_ent_norm in enumerate(jd_entities_list):
            scores = compute_match_score(resume_entities, jd_ent_norm, embed_model, section_weights)
            results.append({"jd_index": i, "jd_name": jd_names[i], **scores})

        df = pd.DataFrame(results)
        if not df.empty:
            df["final_pct"] = (df["final"] * 100).round().astype(int)
            df = df.sort_values("final", ascending=False).reset_index(drop=True)
            top_score = float(df["final_pct"].iloc[0])
            donut_fig = plot_donut_score(top_score)
            score_placeholder.plotly_chart(donut_fig, use_container_width=True)

            st.subheader("Match Scores")

            chunk_size = 5
            num_rows = math.ceil(len(df) / chunk_size)

            for r in range(num_rows):
                row_df = df.iloc[r * chunk_size : (r + 1) * chunk_size]
                cols   = st.columns(len(row_df))

                for i, col in enumerate(cols):
                    idx = r * chunk_size + i
                    row = df.iloc[idx]  # or row_df.iloc[i]
                    col.markdown(f"**{row['jd_name']}**")
                    col.metric("Match Score", f"{row['final_pct']}%")

                    # Bar Matching Mode radios
                    mode_choice = col.radio(
                        "Matching Mode",
                        options=["Semantic Match", "Exact Match", "Partial Match"],
                        horizontal=True,
                        key=f"bar_mode_{idx}"
                    )
                    mode_map = {
                        "Semantic Match": "semantic",
                        "Exact Match":    "raw",
                        "Partial Match":  "fuzzy"
                    }
                    mode = mode_map[mode_choice]

                    # Compute per‐section scores
                    section_scores = {}
                    for sec in ["hard_skills", "soft_skills", "education", "experience"]:
                        if mode == "semantic":
                            section_scores[sec] = row[sec]
                        else:
                            section_scores[sec] = compute_section_score(
                            resume_entities.get(sec, []),
                            jd_entities_list[idx].get(sec, []),
                            threshold=threshold_slider,
                            mode=mode
                        )

                    # Plot the bars inside this same column
                    fig = section_bars(section_scores, title=None)
                    col.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{idx}")

            # Result Summary & Skill Gap 
            st.subheader("Result Summary & Detailed Comparison")
            section_key_map = ["hard_skills","soft_skills","education","experience"]
            tabs = st.tabs(["Hard Skills","Soft Skills","Education","Experience"])
            for tab,sec_key in zip(tabs,section_key_map):
                with tab:
                    pretty = sec_key.replace('_',' ').title()
                    st.write(f"**Section:** {pretty}")
                    rows=[]
                    for j,jd_ent_norm in enumerate(jd_entities_list):
                        res_list = resume_entities.get(sec_key,[])
                        jd_list = jd_ent_norm.get(sec_key,[])
                        overlaps,gaps = overlaps_and_gaps(res_list,jd_list,threshold_slider)
                        rows.append({
                            "JD": jd_names[j],
                            "JD_count": len(jd_list),
                            "Resume_count": len(res_list),
                            "Overlaps": overlaps,
                            "Gaps": gaps
                        })
                    st.write(pd.DataFrame([{"JD":r["JD"],"JD_count":r["JD_count"],"Resume_count":r["Resume_count"]} for r in rows]))
                    for r in rows:
                        overlaps = r["Overlaps"]
                        gaps = r["Gaps"]
                        st.markdown("**Skill Gap Analysis**")
                        with st.expander(f"{r['JD']} — Overlaps {len(overlaps)} / Gaps {len(gaps)}"):
                            st.write("**Overlaps (Resume Item — JD Item — Score)**")
                            if overlaps:
                                overlaps_df = pd.DataFrame(overlaps,columns=["Resume Item","JD Item","Score"])
                                overlaps_df["Score"] = overlaps_df["Score"].astype(int)
                                st.dataframe(overlaps_df,use_container_width=True)
                            else: st.write("No overlaps found for this JD/section at current threshold.")
                            st.write("**Gaps (JD items not matched by resume)**")
                            if gaps:
                                numbered="\n".join(f"{i+1}. {g}" for i,g in enumerate(gaps))
                                st.markdown(numbered)
                            else: st.write("No gaps — resume covers JD items for this section.")

    if st.button("✨ Generate Interview Questions", type="primary"):
        if not jd_texts:
            st.warning("Please upload at least one job description first.")
        else:
            st.markdown("##### 🎯 Sample Interview Questions")
            for name,jd_txt in zip(jd_names,jd_texts):
                with st.spinner(f"Generating questions for {name}…"):
                    qs = generate_interview_questions(jd_txt)
                with st.expander(f"📄 {name} — 5 Interview Questions",expanded=False):
                    if qs:
                        for i,q in enumerate(qs,start=1):
                            st.markdown(f"{i}. {q}")
                    else: st.write("No questions were generated for this JD.")
