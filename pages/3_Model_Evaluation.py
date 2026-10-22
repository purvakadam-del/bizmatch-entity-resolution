import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Model Evaluation | Entity Resolution",
    page_icon="📊",
    layout="wide",
)

# Premium visual styling, independent of the existing app.
st.markdown("""
<style>
.stApp {
    background: #F7F5EF;
    color: #292A25;
}
.block-container {
    max-width: 1200px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}
.eyebrow {
    color: #A8874D;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.hero {
    background: #FFFEFA;
    border: 1px solid #E8E2D6;
    border-radius: 18px;
    padding: 2rem;
    margin: 0.8rem 0 1.5rem;
}
.hero h1 {
    font-family: Georgia, serif;
    font-weight: 500;
    letter-spacing: -0.035em;
    margin-bottom: 0.6rem;
}
.hero p {
    color: #77766D;
    max-width: 760px;
    line-height: 1.7;
}
.section-title {
    font-family: Georgia, serif;
    font-size: 1.55rem;
    margin: 1.6rem 0 0.35rem;
}
.metric-card {
    background: #FFFEFA;
    border: 1px solid #E8E2D6;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    min-height: 112px;
}
.metric-label {
    color: #77766D;
    font-size: 0.82rem;
}
.metric-value {
    color: #294D3B;
    font-family: Georgia, serif;
    font-size: 2rem;
    margin-top: 0.3rem;
}
.note {
    background: #F0E9D9;
    border-left: 3px solid #A8874D;
    border-radius: 5px;
    padding: 0.9rem 1rem;
    color: #49463D;
    line-height: 1.65;
}

.matrix-wrap {
    background: #FFFEFA;
    border: 1px solid #E8E2D6;
    border-radius: 18px;
    padding: 1.25rem;
    margin: 0.8rem 0 1.4rem;
    overflow-x: auto;
}
.matrix-grid {
    display: grid;
    grid-template-columns: minmax(125px, 1.1fr) repeat(2, minmax(110px, 1fr));
    gap: 8px;
    min-width: 380px;
}
.matrix-heading {
    color: #77766D;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.55rem 0.35rem;
    text-align: center;
}
.matrix-row-label {
    display: flex;
    align-items: center;
    color: #49463D;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.7rem 0.35rem;
}
.matrix-cell {
    min-height: 112px;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 0.8rem 0.35rem;
}
.matrix-cell.correct {
    background: #294D3B;
    color: #FFFFFF;
}
.matrix-cell.error {
    background: #F3E5D8;
    color: #8C482D;
    border: 1px solid #E7CBB7;
}
.matrix-number {
    font-family: Georgia, serif;
    font-size: 1.9rem;
    line-height: 1.2;
}
.matrix-description {
    font-size: 0.75rem;
    margin-top: 0.45rem;
    opacity: 0.9;
}
@media (max-width: 600px) {
    .matrix-cell { min-height: 88px; }
    .matrix-number { font-size: 1.45rem; }
}


/* Shared sidebar arrow styling */
button[data-testid="stBaseButton-headerNoPadding"] {
    position: relative !important;
}
button[data-testid="stBaseButton-headerNoPadding"] [data-testid="stIconMaterial"] {
    display: none !important;
}
button[data-testid="stBaseButton-headerNoPadding"]::after {
    content: "≪";
    font-family: Arial, sans-serif;
    font-size: 24px !important;
    font-weight: 400;
    line-height: 28px !important;
    color: rgba(49, 51, 63, 0.6);
    display: flex !important;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    flex: 0 0 28px;
    padding: 0;
    margin: 0;
    box-sizing: border-box;
    visibility: visible !important;
}


/* BIZMATCH SHARED PREMIUM PAGE STYLE */
.stApp,
[data-testid="stAppViewContainer"] {
    background: #F4F0E8 !important;
    color: #20231F !important;
}

[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 1320px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    color: #20231F !important;
    letter-spacing: -0.035em !important;
}

h1, h2 {
    font-family: Georgia, "Times New Roman", serif !important;
    font-weight: 500 !important;
}

[data-testid="stMetric"] {
    background: #FFFEFA !important;
    border: 1px solid #E7E0D3 !important;
    border-radius: 15px !important;
    padding: 17px 19px !important;
    box-shadow: 0 5px 16px rgba(45, 39, 27, .035) !important;
}

[data-testid="stFileUploader"] {
    background: #FFFEFA !important;
    border: 1px dashed #B69A62 !important;
    border-radius: 16px !important;
    padding: 12px !important;
}

[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
    background: #20231F !important;
    color: #FFFEFA !important;
    border: 1px solid #B69A62 !important;
    border-radius: 10px !important;
    transition: all .18s ease;
}

[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #3A3D33 !important;
    border-color: #D6BC83 !important;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #E7E0D3 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

hr {
    border-color: #E7E0D3 !important;
}
/* END BIZMATCH SHARED PREMIUM PAGE STYLE */


/* BIZMATCH PREMIUM EDITORIAL REDESIGN */
.hero {
    position: relative !important;
    overflow: hidden !important;
    padding: 44px 46px !important;
    margin: 10px 0 20px !important;
    border: 1px solid #5C594A !important;
    border-radius: 24px !important;
    background:
        radial-gradient(ellipse at 92% 5%, rgba(190,160,103,.22), transparent 34%),
        linear-gradient(118deg, #20231F 0%, #30332B 64%, #494334 100%) !important;
    box-shadow: 0 18px 42px rgba(39,35,26,.12) !important;
}
.hero .eyebrow {
    color: #D6BC83 !important;
    font-size: 10px !important;
    letter-spacing: .22em !important;
    margin-bottom: 15px !important;
}
.hero h1 {
    color: #FAF7EF !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: clamp(2.3rem, 4vw, 3.5rem) !important;
    font-weight: 400 !important;
    line-height: 1.1 !important;
    letter-spacing: -.045em !important;
    margin: 0 0 15px !important;
}
.hero p {
    color: #D2D0C5 !important;
    font-size: 1rem !important;
    line-height: 1.85 !important;
    max-width: 760px !important;
}
[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 1320px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}
.metric-card {
    position: relative !important;
    overflow: hidden !important;
    background: linear-gradient(145deg, #FFFEFA, #F8F4EB) !important;
    border: 1px solid #E3D9C8 !important;
    border-radius: 17px !important;
    padding: 22px 21px !important;
    min-height: 132px !important;
    box-shadow: 0 8px 24px rgba(45,39,27,.045) !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(45,39,27,.08) !important;
}
.metric-label {
    color: #77766D !important;
    font-size: .78rem !important;
    line-height: 1.6 !important;
}
.metric-value {
    color: #294D3B !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: clamp(1.8rem, 2.5vw, 2.5rem) !important;
    line-height: 1.35 !important;
    margin: 5px 0 !important;
    visibility: visible !important;
}
.section-title {
    color: #292A25 !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: 1.7rem !important;
    letter-spacing: -.035em !important;
    margin: 2rem 0 .55rem !important;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #E3D9C8 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 24px rgba(45,39,27,.04) !important;
}
hr {
    margin: 1.8rem 0 !important;
    border-color: #E4DCCF !important;
}
@media (max-width: 700px) {
    .hero { padding: 28px 24px !important; }
    .hero h1 { font-size: 2.2rem !important; }
}
/* END BIZMATCH PREMIUM EDITORIAL REDESIGN */


/* BIZMATCH FINAL POLISH */
[data-testid="stFileUploader"] {
    background: #FFFEFA !important;
    border: 1px dashed #C8B58D !important;
    border-radius: 16px !important;
    padding: 18px !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #E8E2D6 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] * {
    font-variant-numeric: tabular-nums;
}
.stDownloadButton button,
.stButton button,
[data-testid="stFileUploader"] button {
    border-radius: 10px !important;
    transition: all .18s ease !important;
}
.stDownloadButton button,
.stButton button {
    border: 1px solid #B49A68 !important;
}
.stDownloadButton button:hover,
.stButton button:hover {
    border-color: #8E713F !important;
    box-shadow: 0 5px 16px rgba(70, 58, 35, .10) !important;
    transform: translateY(-1px);
}
[data-testid="stExpander"] {
    border: 1px solid #E8E2D6 !important;
    border-radius: 12px !important;
    background: #FFFEFA !important;
}
[data-testid="stMetric"] {
    background: #FFFEFA !important;
    border: 1px solid #E8E2D6 !important;
    border-radius: 13px !important;
    padding: 15px 17px !important;
}
hr {
    border-color: #E8E2D6 !important;
}
@media (max-width: 768px) {
    .hero { padding: 28px 24px !important; }
    .hero h1 { font-size: 2.2rem !important; }
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
""", unsafe_allow_html=True)

# Recorded V8 clean-split TEST evaluation results.
# This is a separate experiment, not the V3 model currently used by the app.
accuracy = 0.9983465000918611
precision = 0.9994202898550725
recall = 0.9979739507959479
f1_score = 0.9986965966690804
tn, fp, fn, tp = 1986, 2, 7, 3448
total = tn + fp + fn + tp

st.markdown("""
<div class="hero">
  <div class="eyebrow">MODEL INTELLIGENCE · EVALUATION</div>
  <h1>Evaluating entity matching performance.</h1>
  <p>
    A recorded evaluation of the separate V8 clean-split experiment,
    covering held-out test performance, classification errors, and
    the trade-offs involved in automated entity resolution.
  </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    f"V8 clean-split TEST snapshot · {total:,} held-out pairs · Not live predictions"
)

# Summary metrics
st.markdown('<div class="section-title">Performance overview</div>',
            unsafe_allow_html=True)

metrics = [
    ("Accuracy", accuracy, "Correct predictions overall"),
    ("Precision", precision, "How often predicted matches were correct"),
    ("Recall", recall, "How many benchmark matches were detected"),
    ("F1-score", f1_score, "Balance of precision and recall"),
]

cols = st.columns(4)
for col, (label, value, description) in zip(cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value:.2%}</div>
              <div class="metric-label">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



st.markdown('<div class="section-title">Confusion matrix</div>',
            unsafe_allow_html=True)
st.caption("Rows show actual classes; columns show predicted classes.")

st.markdown("""
<style>
div[data-testid="stVerticalBlockBorderWrapper"]:has(.matrix-correct) {
    background: #F0F5F0;
    border: 1px solid #C8D9CC;
    border-radius: 14px;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.matrix-error) {
    background: #FBF0E8;
    border: 1px solid #E7CBB7;
    border-radius: 14px;
}
.matrix-label {
    color: #77766D;
    font-size: 0.78rem;
    font-weight: 600;
}
.matrix-correct { color: #294D3B; }
.matrix-error { color: #9A4F32; }
</style>
""", unsafe_allow_html=True)

header = st.columns([1, 1, 1])
with header[0]:
    st.caption("ACTUAL CLASS")
with header[1]:
    st.caption("PREDICTED: NON-MATCH")
with header[2]:
    st.caption("PREDICTED: MATCH")

row1 = st.columns([1, 1, 1])
with row1[0]:
    st.markdown("**Actual non-match**")
with row1[1]:
    with st.container(border=True):
        st.markdown('<div class="matrix-correct">TRUE NEGATIVE</div>',
                    unsafe_allow_html=True)
        st.metric("Correctly identified", f"{tn:,}")
with row1[2]:
    with st.container(border=True):
        st.markdown('<div class="matrix-error">FALSE POSITIVE</div>',
                    unsafe_allow_html=True)
        st.metric("Incorrect match", f"{fp:,}")

row2 = st.columns([1, 1, 1])
with row2[0]:
    st.markdown("**Actual match**")
with row2[1]:
    with st.container(border=True):
        st.markdown('<div class="matrix-error">FALSE NEGATIVE</div>',
                    unsafe_allow_html=True)
        st.metric("Missed match", f"{fn:,}")
with row2[2]:
    with st.container(border=True):
        st.markdown('<div class="matrix-correct">TRUE POSITIVE</div>',
                    unsafe_allow_html=True)
        st.metric("Correctly identified", f"{tp:,}")

left, right = st.columns(2)
with left:
    st.markdown("#### Correct predictions")
    st.metric("True negatives", f"{tn:,}",
              help="Non-matches correctly identified.")
    st.metric("True positives", f"{tp:,}",
              help="Matches correctly identified.")

with right:
    st.markdown("#### Errors to understand")
    st.metric("False positives", f"{fp:,}",
              help="Non-matches incorrectly predicted as matches.")
    st.metric("False negatives", f"{fn:,}",
              help="Matches incorrectly predicted as non-matches.")

st.markdown('<div class="section-title">What these results mean</div>',
            unsafe_allow_html=True)

st.markdown("""
- **High precision:** very few benchmark non-matches were incorrectly
  predicted as matches in this evaluation.
- **High recall:** most benchmark-positive pairs were detected, although
  some matches were missed.
- **Errors still matter:** false matches can merge distinct businesses;
  missed matches can leave duplicate records unresolved.
- **A review path is important:** uncertain or conflicting records should
  not be treated as verified matches just because a model score is high.
""")

st.markdown('<div class="section-title">Evaluation limitations</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="note">
<strong>Interpret these metrics in context.</strong><br>
These are recorded V8 clean-split TEST results, not results from the V3 model currently running in the app. They are not a
guarantee of production performance. The dataset's negative labels use
an unlisted-ID assumption, which can introduce label noise. Real-world
performance may differ when names, addresses, countries, and data
quality differ from the benchmark. The model score is not a calibrated
probability of a match.
</div>
""", unsafe_allow_html=True)

st.caption(
    "Evaluation: V8 clean-split TEST snapshot · This page does not retrain "
    "the model, change thresholds, or modify decision rules."
)
