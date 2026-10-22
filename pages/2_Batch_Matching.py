from pathlib import Path
import io

import joblib
import pandas as pd
import streamlit as st

from train_model_v3 import make_features
from decision_rules import decide_match

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "business_match_model_v3.joblib"

st.set_page_config(
    page_title="Batch Resolution | Entity Intelligence",
    page_icon="◇",
    layout="wide",
)

st.markdown("""
<style>
:root {
  --ivory:#F4F0E8; --paper:#FFFEFA; --ink:#20231F;
  --muted:#77766D; --gold:#B69A62; --line:#E5DDCE;
}
.stApp,[data-testid="stAppViewContainer"] {
  background:var(--ivory)!important; color:var(--ink)!important;
}
[data-testid="stAppViewContainer"] > .main .block-container {
  max-width:1360px!important; padding:2rem 2.4rem 5rem!important;
}
h1,h2,h3 {
  color:var(--ink)!important; letter-spacing:-.035em!important;
}
h1,h2 {font-family:Georgia,"Times New Roman",serif!important;font-weight:400!important}
h2 {font-size:1.8rem!important;margin-top:1.2rem!important}
h3 {font-size:1.15rem!important}
p,[data-testid="stMarkdownContainer"] {line-height:1.75}
.hero {
  position:relative!important; overflow:hidden!important;
  padding:52px 52px 46px!important; margin:8px 0 34px!important;
  border:1px solid #555345!important; border-radius:24px!important;
  background:
    radial-gradient(ellipse at 88% 8%,rgba(199,169,112,.25),transparent 32%),
    radial-gradient(ellipse at 100% 110%,rgba(125,130,102,.18),transparent 42%),
    linear-gradient(120deg,#20231F 0%,#30332B 62%,#454235 100%)!important;
  box-shadow:0 22px 50px rgba(39,35,26,.13)!important;
}
.hero .eyebrow {
  color:#D9C18E!important; text-transform:uppercase!important;
  letter-spacing:.23em!important; font-size:10px!important;
  font-weight:700!important; margin-bottom:18px!important;
}
.hero h1 {
  color:#FAF7EF!important; font-family:Georgia,"Times New Roman",serif!important;
  font-size:clamp(2.5rem,5vw,4rem)!important; font-weight:400!important;
  line-height:1.08!important; letter-spacing:-.05em!important;
  margin:0 0 16px!important;
}
.hero p {
  color:#D5D2C7!important; font-size:1.02rem!important;
  line-height:1.85!important; max-width:760px!important;
}
[data-testid="stMetric"] {
  background:var(--paper)!important; border:1px solid var(--line)!important;
  border-radius:16px!important; padding:20px!important;
  box-shadow:0 8px 24px rgba(45,39,27,.045)!important;
}
[data-testid="stMetricLabel"] {color:var(--muted)!important;font-size:.83rem!important}
[data-testid="stMetricValue"] {color:var(--ink)!important}
[data-testid="stFileUploader"] {
  background:var(--paper)!important; border:1px dashed #B69A62!important;
  border-radius:18px!important; padding:22px!important;
  box-shadow:0 10px 28px rgba(45,39,27,.04)!important;
}
[data-testid="stFileUploader"] section {background:transparent!important}
[data-testid="stExpander"] {
  background:var(--paper)!important; border:1px solid var(--line)!important;
  border-radius:14px!important; overflow:hidden!important;
}
[data-testid="stDataFrame"],[data-testid="stTable"] {
  border:1px solid var(--line)!important; border-radius:13px!important;
  overflow:hidden!important; box-shadow:0 8px 24px rgba(45,39,27,.04)!important;
}
.stButton > button,[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
  min-height:44px!important; border-radius:11px!important;
  font-weight:600!important; transition:all .18s ease!important;
}
.stButton > button {
  background:#292D25!important; color:#FFFEFA!important;
  border:1px solid #292D25!important;
}
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
  background:#292D25!important; color:#FFFEFA!important;
  border:1px solid var(--gold)!important;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
  background:#444638!important; border-color:#D6BC83!important;
  transform:translateY(-1px)!important;
  box-shadow:0 7px 18px rgba(45,39,27,.12)!important;
}
[data-testid="stAlert"] {border-radius:12px!important}
[data-testid="stSidebar"] {background:#EEE9DE!important}
[data-testid="stSidebar"] > div {border-right:1px solid #E1D8C8}
button[data-testid="stBaseButton-headerNoPadding"] {position:relative!important}
button[data-testid="stBaseButton-headerNoPadding"] [data-testid="stIconMaterial"] {display:none!important}
button[data-testid="stBaseButton-headerNoPadding"]::after {
  content:"≪";font-family:Arial,sans-serif;font-size:24px;
  color:rgba(49,51,63,.6);display:flex;align-items:center;
  justify-content:center;width:28px;height:28px;
}
hr {border-color:var(--line)!important;margin:1.8rem 0!important}
@media(max-width:700px) {
  [data-testid="stAppViewContainer"] > .main .block-container {padding:1rem!important}
  .hero {padding:30px 24px!important;border-radius:18px!important}
  .hero h1 {font-size:2.35rem!important}
}

/* BIZMATCH COHESIVE LUXURY THEME */
:root {
  --bm-bg: #F3EFE6;
  --bm-paper: #FFFEFA;
  --bm-ink: #242720;
  --bm-muted: #77766D;
  --bm-gold: #B69A62;
  --bm-line: #E3DAC9;
}
html, body, [data-testid="stApp"], .stApp,
[data-testid="stAppViewContainer"] {
  background: var(--bm-bg) !important;
  color: var(--bm-ink) !important;
}
[data-testid="stAppViewContainer"] > .main .block-container {
  max-width: 1360px !important;
  padding: 2.2rem 2.4rem 5rem !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
  background: #EAE4D8 !important;
}
[data-testid="stSidebar"] {
  border-right: 1px solid #DCD2C0 !important;
}
[data-testid="stSidebar"] * {
  color: #292C25;
}
h1, h2, h3, h4 {
  color: var(--bm-ink) !important;
  letter-spacing: -.035em !important;
}
h1, h2 {
  font-family: Georgia, "Times New Roman", serif !important;
  font-weight: 400 !important;
}
[data-testid="stMetric"],
[data-testid="stContainer"] > div,
[data-testid="stExpander"] {
  border-radius: 15px;
}
[data-testid="stMetric"] {
  background: var(--bm-paper) !important;
  border: 1px solid var(--bm-line) !important;
  padding: 18px !important;
  box-shadow: 0 8px 24px rgba(45,39,27,.045) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--bm-muted) !important;
}
[data-testid="stMetricValue"] {
  color: var(--bm-ink) !important;
}
[data-testid="stFileUploader"] {
  background: var(--bm-paper) !important;
  border: 1px dashed var(--bm-gold) !important;
  border-radius: 17px !important;
  padding: 18px !important;
}
[data-testid="stExpander"] {
  background: var(--bm-paper) !important;
  border: 1px solid var(--bm-line) !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid var(--bm-line) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}
.stButton > button,
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
  border-radius: 10px !important;
  min-height: 42px !important;
  font-weight: 600 !important;
  transition: all .18s ease !important;
}
.stButton > button {
  background: #292D25 !important;
  color: #FFFEFA !important;
  border: 1px solid #292D25 !important;
}
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
  background: #292D25 !important;
  color: #FFFEFA !important;
  border: 1px solid var(--bm-gold) !important;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
  background: #414438 !important;
  border-color: #D6BC83 !important;
  transform: translateY(-1px);
}
.hero {
  background:
    radial-gradient(ellipse at 92% 5%, rgba(190,160,103,.23), transparent 35%),
    linear-gradient(118deg, #20231F 0%, #30332B 64%, #494334 100%) !important;
  border: 1px solid #5C594A !important;
  border-radius: 22px !important;
  padding: 42px !important;
  box-shadow: 0 18px 42px rgba(39,35,26,.10) !important;
}
.hero .eyebrow {
  color: #D6BC83 !important;
  letter-spacing: .2em !important;
}
.hero h1 {
  color: #FAF7EF !important;
  font-family: Georgia, "Times New Roman", serif !important;
  font-weight: 400 !important;
}
.hero p {
  color: #D2D0C5 !important;
  line-height: 1.8 !important;
}
hr {
  border-color: var(--bm-line) !important;
}
@media (max-width: 700px) {
  [data-testid="stAppViewContainer"] > .main .block-container {
    padding: 1rem !important;
  }
  .hero { padding: 26px !important; }
}


/* BIZMATCH TOP BAR COLOR MATCH */
header[data-testid="stHeader"],
[data-testid="stHeader"] {
    background: #F3EFE6 !important;
    color: #242720 !important;
}
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: #F3EFE6 !important;
}

</style>
<div class="hero">
  <div class="eyebrow">Entity intelligence / Workspace 02</div>
  <h1>Resolve at scale.</h1>
  <p>Evaluate multiple business-record pairs with the existing V3 model,
  transparent decision safeguards, and a review queue for uncertain cases.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    required = {"model", "feature_names", "threshold"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("The V3 model bundle has an unexpected structure.")
    if not hasattr(bundle["model"], "predict_proba"):
        raise ValueError("The loaded model does not support predict_proba.")
    return bundle


try:
    bundle = load_model_bundle()
except Exception as exc:
    st.error(f"Could not load the V3 model: {exc}")
    st.stop()

st.markdown("### Upload a candidate-pair CSV")
st.write(
    "Each row represents one reference business and one candidate business. "
    "The uploaded file is processed in this session and is not saved to disk."
)

with st.expander("Required CSV columns and example", expanded=True):
    st.code(
        "source1_name,source1_address,source1_country,"
        "candidate_name,candidate_address,candidate_country\n"
        "Acme Industries,100 Main Street,US,"
        "Acme Industrial Ltd,100 Main St,US",
        language="csv",
    )
    st.caption(
        "Address and country columns may be blank, but all six columns must "
        "exist. Optional source1_id and candidate_id columns are preserved."
    )

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()

    required = [
        "source1_name", "source1_address", "source1_country",
        "candidate_name", "candidate_address", "candidate_country",
    ]
    missing = [col for col in required if col not in raw.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()

    if raw.empty:
        st.warning("The uploaded CSV contains no rows.")
        st.stop()

    st.caption(f"Loaded {len(raw):,} record pairs.")
    preview_count = min(5, len(raw))
    st.dataframe(raw.head(preview_count), use_container_width=True)

    max_rows = st.number_input(
        "Maximum rows to process",
        min_value=1,
        max_value=len(raw),
        value=min(len(raw), 1000),
        step=1,
        help="Start with a small sample when testing a new file.",
    )

    if st.button("Run batch matching", type="primary", use_container_width=True):
        batch = raw.head(int(max_rows)).copy().reset_index(drop=True)

        # Empty names cannot be meaningfully evaluated.
        valid_names = (
            batch["source1_name"].str.strip().ne("")
            & batch["candidate_name"].str.strip().ne("")
        )
        results = batch.copy()
        results["model_score"] = pd.NA
        results["decision"] = "INVALID_INPUT"
        results["decision_reason"] = (
            "Both reference and candidate business names are required."
        )

        valid = batch.loc[valid_names].copy()

        if not valid.empty:
            with st.spinner(f"Evaluating {len(valid):,} pairs..."):
                features = make_features(valid)
                features = features[bundle["feature_names"]]
                scores = bundle["model"].predict_proba(features)[:, 1]
                threshold = float(bundle["threshold"])

                decisions = []
                reasons = []

                for idx, score in zip(valid.index, scores):
                    row = batch.loc[idx]
                    feature_row = features.loc[idx]

                    decision, reason = decide_match(
                        name1=row["source1_name"],
                        name2=row["candidate_name"],
                        country1=row["source1_country"],
                        country2=row["candidate_country"],
                        name_similarity=float(
                            feature_row["name_similarity"]
                        ),
                        name_token_similarity=float(
                            feature_row["name_token_similarity"]
                        ),
                        model_score=float(score),
                        threshold=threshold,
                    )
                    decisions.append(decision)
                    reasons.append(reason)

                results.loc[valid_names, "model_score"] = scores
                results.loc[valid_names, "decision"] = decisions
                results.loc[valid_names, "decision_reason"] = reasons

        results["model_score"] = pd.to_numeric(
            results["model_score"], errors="coerce"
        )

        st.session_state["batch_results"] = results
        st.success(f"Processed {len(results):,} pairs.")

if "batch_results" in st.session_state:
    results = st.session_state["batch_results"].copy()

    st.markdown("---")
    st.markdown("### Batch overview")

    total = len(results)
    matches = int(results["decision"].eq("MATCH").sum())
    reviews = int(results["decision"].eq("REVIEW").sum())
    nonmatches = int(results["decision"].eq("NON-MATCH").sum())
    invalid = int(results["decision"].eq("INVALID_INPUT").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pairs processed", f"{total:,}")
    c2.metric("Match decisions", f"{matches:,}")
    c3.metric("Manual review", f"{reviews:,}")
    c4.metric("Non-match decisions", f"{nonmatches:,}")

    st.caption(
        f"{invalid:,} rows have missing business names. "
        "Decision counts describe this batch only, not model accuracy."
    )

    counts = (
        results["decision"]
        .value_counts()
        .reindex(
            ["MATCH", "REVIEW", "NON-MATCH", "INVALID_INPUT"],
            fill_value=0,
        )
    )
    st.bar_chart(counts, horizontal=True)

    st.markdown("### Explore results")
    chosen = st.multiselect(
        "Filter by decision",
        options=["MATCH", "REVIEW", "NON-MATCH", "INVALID_INPUT"],
        default=["MATCH", "REVIEW", "NON-MATCH", "INVALID_INPUT"],
    )

    filtered = results[results["decision"].isin(chosen)].copy()
    st.caption(f"Showing {len(filtered):,} of {len(results):,} rows.")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    review_queue = results[results["decision"].isin(
        ["REVIEW", "INVALID_INPUT"]
    )]
    st.markdown("### Review queue")
    st.write(
        f"{len(review_queue):,} rows need attention before a final decision."
    )
    if not review_queue.empty:
        st.dataframe(review_queue, use_container_width=True, hide_index=True)

    output = results.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download all batch results",
        data=output,
        file_name="entity_matching_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    review_output = review_queue.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download review queue",
        data=review_output,
        file_name="entity_matching_review_queue.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.info(
        "Scores are uncalibrated diagnostic scores. MATCH, NON-MATCH, and "
        "REVIEW are rule-based outcomes, not independently verified identities."
    )
