# TalentGPT

AI resume screening and candidate ranking. Upload a CSV of resumes (columns: file_name, resume_text),
paste a job description, and get candidates ranked by semantic similarity (sentence-transformers + cosine similarity).

Run locally:

    pip install -r requirements.txt
    streamlit run app.py
