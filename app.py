import io
import os
import re
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "Resume ATS Scorer"
TABLE_NAME = "workspace.default.resumes_clean"
TOKEN_PATTERN = r"[A-Za-z][A-Za-z0-9+#]+"
PDF_FOLDER = Path("resumes_pdf")

# Defaults. Streamlit Secrets or environment variables override these.
DEFAULT_HOST = "dbc-796f3bee-f198.cloud.databricks.com"
DEFAULT_HTTP_PATH = "/sql/1.0/warehouses/e33124372ddd7ee7"

st.set_page_config(page_title=APP_NAME, layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #172b27;
        --muted: #64736d;
        --paper: #f5f6f1;
        --line: #dce3db;
        --green: #176b51;
        --lime: #d8f36a;
        --coral: #e78366;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background:
            linear-gradient(rgba(23, 43, 39, .025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(23, 43, 39, .025) 1px, transparent 1px),
            var(--paper);
        background-size: 32px 32px;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 1.5rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.4rem 2.6rem 2.2rem;
        margin: 0 0 2rem;
        border-radius: 6px;
        color: #f7faf4;
        background: var(--ink);
        border-bottom: 5px solid var(--lime);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 210px;
        height: 210px;
        right: 7%;
        top: -116px;
        border: 1px solid rgba(216, 243, 106, .42);
        border-radius: 50%;
        box-shadow: 0 0 0 25px rgba(216, 243, 106, .04), 0 0 0 52px rgba(216, 243, 106, .03);
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        color: var(--lime);
        text-transform: uppercase;
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: 1.4px;
    }
    .hero-kicker:before {
        content: "";
        width: 8px;
        height: 8px;
        background: var(--coral);
        border-radius: 50%;
    }
    .hero h1 {
        position: relative;
        z-index: 1;
        margin: .8rem 0 .45rem;
        font: 800 clamp(2.2rem, 5vw, 3.65rem)/1.05 'Manrope', sans-serif;
        letter-spacing: 0;
        color: #f7faf4;
    }
    .hero p {
        position: relative;
        z-index: 1;
        max-width: 620px;
        margin: 0;
        color: #c6d1ca;
        font-size: 1.02rem;
    }
    h2, h3 { font-family: 'Manrope', sans-serif; color: var(--ink); }
    h3 { margin-top: 1.7rem; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stRadio"] > label,
    [data-testid="stFileUploader"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stTextArea"] label,
    [data-testid="stNumberInput"] label {
        color: var(--ink);
        font-weight: 600;
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        background: rgba(255, 255, 255, .76);
        border-color: var(--line);
        border-radius: 4px;
    }
    .stButton > button, .stDownloadButton > button {
        min-height: 2.7rem;
        border-radius: 4px;
        border: 1px solid var(--green);
        font-weight: 700;
        transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
    }
    .stButton > button[kind="primary"] {
        color: #fff;
        background: var(--green);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 18px rgba(23, 43, 39, .12);
    }
    [data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        background: rgba(255, 255, 255, .72);
        border: 1px solid var(--line);
        border-top: 3px solid var(--green);
        border-radius: 4px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 4px; }
    .credit-footer {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: .75rem 1.5rem;
        margin-top: 3.5rem;
        padding: 1.2rem 0 .4rem;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: .88rem;
    }
    .credit-footer strong { color: var(--ink); font-weight: 700; }
    .credit-footer a { color: var(--green); text-decoration: none; }
    .credit-footer a:hover { text-decoration: underline; }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 1.5rem; }
        .hero { padding: 1.7rem 1.35rem; }
        .hero:after { right: -80px; }
        .credit-footer { align-items: flex-start; flex-direction: column; }
    }
    </style>
    <section class="hero">
        <div class="hero-kicker">Talent intelligence · Candidate insights</div>
        <h1>Resume ATS Scorer</h1>
        <p>Find the strongest candidates with semantic matching and focused keyword analysis.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


# ---------- Databricks settings ----------
def read_setting(name, default=None):
    """Read a value from Streamlit Secrets first, then environment variables."""
    try:
        value = st.secrets[name]
    except Exception:
        value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value)


def clean_token(raw):
    """Remove quotes, spaces, line breaks and a 'Bearer ' prefix from the token."""
    if not raw:
        return None
    raw = str(raw)
    match = re.search(r"dapi[A-Za-z0-9\-]+", raw)  # standard personal access token
    if match:
        return match.group(0)
    cleaned = re.sub(r"^\s*Bearer\s+", "", raw, flags=re.IGNORECASE)
    return re.sub(r"[\s\"']+", "", cleaned) or None


def clean_host(raw):
    host = str(raw).strip()
    host = re.sub(r"^https?://", "", host, flags=re.IGNORECASE)
    return host.split("/")[0].strip()


def get_databricks_settings():
    host = clean_host(read_setting("DATABRICKS_SERVER_HOSTNAME", DEFAULT_HOST))
    http_path = read_setting("DATABRICKS_HTTP_PATH", DEFAULT_HTTP_PATH).strip()
    raw_token = read_setting("DATABRICKS_TOKEN")

    st.write("RAW TOKEN START:", str(raw_token)[:10])

    token = clean_token(raw_token)

    st.write("CLEAN TOKEN START:", str(token)[:10])
    return host, http_path, token

db_host, db_http_path, db_token = get_databricks_settings()
st.write("SECRET KEYS:", list(st.secrets.keys()))



def check_token(host, token):
    """Ask Databricks whether it accepts the token. Returns (ok, message)."""
    try:
        import requests

        response = requests.get(
            f"https://{host}/api/2.0/preview/scim/v2/Me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
    except Exception:
        return True, ""  # network problem: let the SQL connector try anyway
    if response.status_code in (401, 403):
        return False, (
            f"Databricks rejected the token (HTTP {response.status_code}). "
            f"Create a new token in the workspace {host} and save it as "
            "DATABRICKS_TOKEN in Streamlit Secrets, then reboot the app."
        )
    return True, ""


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(ttl=600)
def load_databricks_resumes(host, http_path, token):
    from databricks import sql

    st.write("HOST:", host)
    st.write("HTTP PATH:", http_path)

    if token:
        st.write("TOKEN START:", token[:4])
        st.write("TOKEN LENGTH:", len(token))
    else:
        st.write("TOKEN IS NONE")

    connection = sql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token
    )

    query = """
    SELECT *
    FROM workspace.default.resumes_clean
    LIMIT 1000
    """

    df = pd.read_sql(query, connection)

    connection.close()

    return df

# ---------- Scoring helpers ----------
def extract_keywords(job_description, max_keywords=40):
    vectorizer = CountVectorizer(
        stop_words="english", lowercase=True, token_pattern=TOKEN_PATTERN
    )
    counts = vectorizer.fit_transform([job_description]).toarray()[0]
    words = vectorizer.get_feature_names_out()
    ranked = sorted(zip(words, counts), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:max_keywords]]


def score_resumes(df, text_col, job_description, model):
    texts = df[text_col].fillna("").astype(str).tolist()

    model.max_seq_length = 256
    jd_emb = model.encode([job_description])
    res_emb = model.encode(texts, batch_size=16, show_progress_bar=False)
    semantic = cosine_similarity(jd_emb, res_emb)[0].clip(0, 1)

    keywords = extract_keywords(job_description)
    matched_all, missing_all, keyword_scores = [], [], []
    for text in texts:
        tokens = set(re.findall(TOKEN_PATTERN, text.lower()))
        matched = [k for k in keywords if k in tokens]
        missing = [k for k in keywords if k not in tokens]
        matched_all.append(", ".join(matched))
        missing_all.append(", ".join(missing))
        keyword_scores.append(len(matched) / len(keywords) if keywords else 0)

    result = df.copy()
    result["semantic_score"] = (semantic * 100).round(1)
    result["keyword_score"] = [round(k * 100, 1) for k in keyword_scores]
    result["ats_score"] = (
        0.6 * result["semantic_score"] + 0.4 * result["keyword_score"]
    ).round(1)
    result["matched_keywords"] = matched_all
    result["missing_keywords"] = missing_all

    result = result.sort_values("ats_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    return result


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "resume"


def find_original_pdf(file_name):
    """Look for the original PDF in the resumes_pdf folder."""
    name = Path(str(file_name)).name
    for candidate in (PDF_FOLDER / name, PDF_FOLDER / f"{Path(name).stem}.pdf"):
        if candidate.is_file():
            return candidate
    return None


def text_to_pdf(text):
    """Build a simple PDF from extracted resume text."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    safe = str(text).encode("latin-1", "replace").decode("latin-1")
    safe = re.sub(r"(\S{60})", r"\1 ", safe)  # break very long unspaced strings
    pdf.multi_cell(0, 5, safe, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def get_pdf_bytes(file_name, text):
    """Return (pdf_bytes, is_original)."""
    original = find_original_pdf(file_name)
    if original is not None:
        return original.read_bytes(), True
    return text_to_pdf(text), False


def build_zip(rows, id_col, text_col):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, r in rows.iterrows():
            pdf_bytes, _ = get_pdf_bytes(r[id_col], r[text_col])
            zf.writestr(f"{int(r['rank']):03d}_{safe_name(r[id_col])}.pdf", pdf_bytes)
    return buffer.getvalue()


# ---------- Data source ----------
source = st.radio("Resume source", ["Upload CSV", "Load from Databricks"], horizontal=True)

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload Resume CSV", type=["csv"])

    if uploaded is not None:
        st.session_state["df"] = pd.read_csv(uploaded)

else:
    db_host, db_http_path, db_token = get_databricks_settings()

    if st.button("Connect & Load Resumes"):
        try:
            st.session_state["df"] = load_databricks_resumes(
                db_host,
                db_http_path,
                db_token
            )

            st.success(
                f"Loaded {len(st.session_state['df'])} resumes"
            )

        except Exception as exc:
            import traceback

            st.error(str(exc))
            st.code(traceback.format_exc())



df = st.session_state.get("df")

if df is not None:
    st.caption(f"{len(df)} resumes loaded")
    cols = list(df.columns)
    guess_text = next((c for c in cols if "resume" in c.lower() or "text" in c.lower()), cols[0])
    guess_id = next((c for c in cols if "name" in c.lower() or "id" in c.lower()), cols[0])

    c1, c2 = st.columns(2)
    text_col = c1.selectbox("Resume text column", cols, index=cols.index(guess_text))
    id_col = c2.selectbox("Candidate identifier column", cols, index=cols.index(guess_id))

    job_description = st.text_area("Paste Job Description", height=200)

    if st.button("Score Resumes", type="primary"):
        if not job_description.strip():
            st.warning("Please paste a job description.")
        else:
            with st.spinner("Scoring resumes..."):
                st.session_state["results"] = score_resumes(
                    df, text_col, job_description, load_model()
                )
                st.session_state["text_col"] = text_col
                st.session_state["id_col"] = id_col

# ---------- Results (persist across reruns) ----------
results = st.session_state.get("results")

if results is not None:
    text_col = st.session_state["text_col"]
    id_col = st.session_state["id_col"]

    st.success("Scoring complete")
    st.markdown("### Ranked candidates")

    show_cols = ["rank", id_col, "ats_score", "semantic_score", "keyword_score",
                 "matched_keywords", "missing_keywords"]
    show_cols = list(dict.fromkeys(show_cols))
    st.dataframe(
        results[show_cols],
        width="stretch",
        column_config={
            "ats_score": st.column_config.ProgressColumn(
                "ATS score", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )

    st.markdown("### Downloads")
    top_n = int(
        st.number_input("Top N candidates for shortlist", 1, len(results), min(10, len(results)))
    )
    shortlist = results.head(top_n)

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Full results (CSV)",
        results.to_csv(index=False).encode("utf-8"),
        "ats_ranked_candidates.csv",
        "text/csv",
    )
    d2.download_button(
        "Shortlist (CSV)",
        shortlist.to_csv(index=False).encode("utf-8"),
        "ats_shortlist.csv",
        "text/csv",
    )
    d3.download_button(
        "Shortlist resumes (ZIP of PDFs)",
        build_zip(shortlist, id_col, text_col),
        "ats_shortlist_resumes.zip",
        "application/zip",
    )

    st.markdown("### Candidate details")
    choice = st.selectbox(
        "Select a candidate",
        results.index,
        format_func=lambda i: (
            f"#{results.loc[i, 'rank']} - {results.loc[i, id_col]} "
            f"({results.loc[i, 'ats_score']})"
        ),
    )
    row = results.loc[choice]

    m1, m2, m3 = st.columns(3)
    m1.metric("ATS score", row["ats_score"])
    m2.metric("Semantic match", row["semantic_score"])
    m3.metric("Keyword match", row["keyword_score"])

    st.write("**Matched keywords:**", row["matched_keywords"] or "None")
    st.write("**Missing keywords:**", row["missing_keywords"] or "None")

    with st.expander("View resume text"):
        st.text(str(row[text_col]))

    pdf_bytes, is_original = get_pdf_bytes(row[id_col], row[text_col])
    if is_original:
        label = "Download original resume (PDF)"
    else:
        st.info(
            "Original PDF not found in the 'resumes_pdf' folder, so a PDF "
            "generated from the extracted text is offered instead."
        )
        label = "Download resume as PDF (generated from text)"

    st.download_button(
        label,
        pdf_bytes,
        f"{safe_name(row[id_col])}.pdf",
        "application/pdf",
    )

st.markdown(
    """
    <footer class="credit-footer">
        <span>Made by <strong>Kumar Harsh</strong></span>
        <span><a href="mailto:kh949118@gmail.com">kh949118@gmail.com</a></span>
        <span>Ph.no - <a href="tel:+919279157296">+91-9279157296</a></span>
    </footer>
    """,
    unsafe_allow_html=True,
)