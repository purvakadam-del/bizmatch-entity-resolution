from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from train_model_v3 import make_features, normalize_text
from decision_rules import decide_match

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "business_match_model_v3.joblib"

REQUIRED_BUNDLE_KEYS = {"model", "feature_names", "threshold"}
REQUIRED_FEATURES = [
    "name_similarity",
    "name_token_similarity",
    "address_similarity",
    "address_token_similarity",
    "address_missing",
    "country_match",
]

st.set_page_config(
    page_title="BizMatch | Intelligent Entity Resolution",
    page_icon="🏢",
    layout="wide",
)

# ENTITY_INTELLIGENCE_IVORY_GOLD
st.markdown("""
<style>
/* Fix expander icon rendering */
[data-testid="stExpander"] [data-testid="stIconMaterial"] {
    display: none !important;
}
/* Fix sidebar expand arrow icon */
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
    display: inline-flex !important;
    font-size: 0 !important;
}

[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]::after {
    content: "›";
    font-family: Arial, sans-serif;
    font-size: 32px;
    line-height: 1;
}

:root {
    --ivory: #F7F5EF;
    --paper: #FFFEFA;
    --ink: #292A25;
    --muted: #77766D;
    --gold: #A8874D;
    --gold-soft: #E6D9BD;
    --line: #E8E2D6;
}

/* Overall canvas */
.stApp {
    background: #F7F5EF !important;
    color: #292A25;
}

[data-testid="stAppViewContainer"] {
    background: #F7F5EF !important;
}

[data-testid="stHeader"] {
    background: rgba(247, 245, 239, 0.96) !important;
}

[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Typography */
html, body, [class*="st-"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 sans-serif;
}
h1, h2, h3 {
    color: var(--ink) !important;
    letter-spacing: -0.035em;
}
p, label, [data-testid="stMarkdownContainer"] {
    color: var(--ink);
}
[data-testid="stCaptionContainer"] {
    color: var(--muted);
}

/* Hero */
.entity-hero {
    background: linear-gradient(120deg, #FFFEFA 0%, #F1EBDD 100%);
    border: 1px solid #E5DCCB;
    border-radius: 22px;
    padding: 34px 38px;
    margin: 0 0 28px 0;
    position: relative;
    overflow: hidden;
}
.entity-eyebrow {
    color: #917344;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.19em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.entity-hero h1 {
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2.15rem, 4vw, 3.35rem);
    font-weight: 500;
    line-height: 1.08;
    letter-spacing: -0.045em;
    margin: 0 0 14px 0;
    color: #282923;
}
.entity-hero p {
    color: #6D6B61;
    max-width: 680px;
    line-height: 1.75;
    font-size: 15px;
    margin: 0;
}
.entity-hero .hero-tag {
    display: inline-block;
    border: 1px solid #D9C8A5;
    color: #80663A;
    border-radius: 30px;
    padding: 7px 12px;
    margin-top: 22px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.13em;
}

/* Section headings and separators */
h2, h3 {
    letter-spacing: -0.025em;
}
hr {
    border-color: var(--line) !important;
}


/* Premium entity comparison workspace */
.workspace-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    margin: 28px 0 18px;
}
.workspace-heading h2 {
    color: #292A25;
    font-family: Georgia, serif;
    font-size: 28px;
    font-weight: 500;
    letter-spacing: -0.7px;
    margin: 5px 0 4px;
}
.workspace-heading p {
    color: #77766D;
    font-size: 13px;
    margin: 0;
}
.section-kicker, .entity-card-kicker {
    color: #A8874D;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1.8px;
}
.workspace-status {
    align-items: center;
    border: 1px solid #E6D9BD;
    border-radius: 999px;
    color: #74613D;
    display: flex;
    flex-shrink: 0;
    font-size: 9px;
    font-weight: 750;
    gap: 8px;
    letter-spacing: 1px;
    padding: 9px 12px;
}
.workspace-status span, .footnote-dot {
    background: #A8874D;
    border-radius: 50%;
    display: inline-block;
    height: 6px;
    width: 6px;
}
.entity-card-heading {
    align-items: center;
    background: #F1EDE3;
    border: 1px solid #E8E2D6;
    border-bottom: 0;
    border-radius: 14px 14px 0 0;
    display: flex;
    gap: 14px;
    margin-bottom: 0;
    padding: 19px 20px 16px;
}
.entity-number {
    align-items: center;
    background: #FFFEFA;
    border: 1px solid #DCCBA9;
    border-radius: 11px;
    color: #967542;
    display: flex;
    flex-shrink: 0;
    font-family: Georgia, serif;
    font-size: 20px;
    height: 45px;
    justify-content: center;
    width: 45px;
}
.entity-card-title {
    color: #292A25;
    font-family: Georgia, serif;
    font-size: 20px;
    margin-top: 4px;
}
.entity-card-caption {
    color: #77766D;
    font-size: 12px;
    margin-top: 3px;
}
[data-testid="stVerticalBlock"]:has(> .entity-card-heading) {
    gap: 0.65rem;
}
.entity-card-heading + [data-testid="stTextInput"] {
    padding-top: 15px;
}
.workspace-footnote {
    align-items: center;
    color: #77766D;
    display: flex;
    font-size: 11px;
    gap: 8px;
    margin: 15px 0 24px;
}
.footnote-dot {
    height: 5px;
    width: 5px;
}
@media (max-width: 700px) {
    .workspace-heading {
        align-items: flex-start;
        flex-direction: column;
    }
}


/* Premium form controls */
[data-testid="stTextInput"],
[data-testid="stSelectbox"] {
    margin-bottom: 0.35rem;
}

[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #FFFEFA !important;
    border: 1px solid #DED6C5 !important;
    border-radius: 11px !important;
    color: #292A25 !important;
    box-shadow: 0 1px 2px rgba(41, 42, 37, 0.025) !important;
    min-height: 44px;
    transition: border-color 160ms ease, box-shadow 160ms ease;
}

[data-testid="stTextInput"] input:hover,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border-color: #B9A57E !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: #536B54 !important;
    box-shadow: 0 0 0 3px rgba(83, 107, 84, 0.12) !important;
    outline: none !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #A19A8C !important;
    opacity: 1;
}

[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label {
    color: #55564D !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    margin-bottom: 5px !important;
}

[data-testid="stTextInput"] input:disabled {
    background: #F0EDE5 !important;
    color: #77766D !important;
    opacity: 1 !important;
}



/* Comparison panels */
[data-testid="stColumn"] > div > div[data-testid="stVerticalBlock"] {
    gap: 0.7rem;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--paper);
    border-color: var(--line) !important;
    border-radius: 16px;
}

/* Buttons */
.stButton > button,
[data-testid="stFormSubmitButton"] button {
    background: #292A25;
    color: #FDFBF6;
    border: 1px solid #292A25;
    border-radius: 10px;
    padding: 0.65rem 1.15rem;
    font-weight: 600;
    transition: all 160ms ease;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #A8874D;
    border-color: #A8874D;
    color: white;
}

/* Result metrics */
[data-testid="stMetric"] {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 18px 20px;
}
[data-testid="stMetricLabel"] {
    color: var(--muted);
}
[data-testid="stMetricValue"] {
    color: var(--ink);
    font-weight: 650;
}
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Tables and diagnostics */
[data-testid="stTable"] {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stExpander"] {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 12px;
}
[data-testid="stCode"] {
    border: 1px solid var(--line);
    border-radius: 10px;
}

/* Keep the design usable on narrow screens */
@media (max-width: 700px) {
    [data-testid="stAppViewContainer"] > .main .block-container {
        padding: 1.1rem 1rem 2.5rem;
    }
    .entity-hero {
        padding: 25px 22px;
        border-radius: 16px;
    }
}
/* PREMIUM DECISION PANEL */
.decision-panel {
    background: linear-gradient(135deg, #fffefa 0%, #f1eee4 100%);
    border: 1px solid #e5ddcc;
    border-radius: 22px;
    padding: 30px 32px;
    margin: 14px 0 22px 0;
    box-shadow: 0 8px 28px rgba(53, 47, 33, 0.055);
}

.decision-kicker {
    color: #8d774e;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.19em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.decision-title {
    color: #292a25;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2rem, 4vw, 3.1rem);
    font-weight: 500;
    line-height: 1.1;
    margin: 0 0 12px 0;
}

.decision-description {
    color: #6e6b60;
    font-size: 0.98rem;
    line-height: 1.75;
    margin: 0;
}

.decision-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 7px 13px;
    margin-bottom: 18px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.13em;
}

.decision-match {
    background: #e2eee4;
    color: #285b3b;
    border: 1px solid #c5ddca;
}

.decision-review {
    background: #f6ecd5;
    color: #805d20;
    border: 1px solid #e8d6a8;
}

.decision-nonmatch {
    background: #f3e4df;
    color: #873f32;
    border: 1px solid #e6c8bf;
}

.decision-metric {
    background: #fffefa;
    border: 1px solid #e8e2d6;
    border-radius: 16px;
    padding: 20px;
    height: 100%;
}

.decision-metric-label {
    color: #77766d;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.decision-metric-value {
    color: #292a25;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.85rem;
    line-height: 1.2;
}

.decision-note {
    color: #77766d;
    font-size: 0.8rem;
    line-height: 1.65;
    border-top: 1px solid #e8e2d6;
    margin-top: 20px;
    padding-top: 14px;
}

.section-kicker {
    color: #8d774e;
    font-size: 0.69rem;
    font-weight: 700;
    letter-spacing: 0.17em;
    text-transform: uppercase;
}

@media (max-width: 640px) {
    .decision-panel {
        padding: 23px 20px;
        border-radius: 17px;
    }
}
/* Normalized text diagnostics */
.section-eyebrow {
    color: var(--gold);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 24px;
    margin-bottom: 8px;
}

.diagnostics-heading {
    color: var(--ink);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 27px;
    font-weight: 500;
    letter-spacing: -0.035em;
    margin-bottom: 6px;
}

.diagnostics-subtitle {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 18px;
}

[data-testid="stCodeBlock"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stCodeBlock"] pre {
    background: var(--paper);
    color: var(--ink);
    font-size: 13px;
    line-height: 1.8;
}

/* Premium preset selector */
[data-testid="stSelectbox"] label {
    color: #77766D !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #FFFEFA !important;
    border: 1px solid #DED2BB !important;
    border-radius: 14px !important;
    min-height: 50px !important;
    box-shadow: 0 2px 8px rgba(41, 42, 37, 0.025) !important;
    transition: border-color 180ms ease, box-shadow 180ms ease;
}

/* Keep the selected value readable */
[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] div,
[data-testid="stSelectbox"] [data-baseweb="select"] span {
    font-size: 14px;
}

/* Elegant hover and keyboard focus */
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    border-color: #A8874D !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: #A8874D !important;
    box-shadow: 0 0 0 3px rgba(168, 135, 77, 0.12) !important;
}



/* Consistent sidebar arrows across all pages */
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





/* BIZMATCH PREMIUM REFINEMENT */
:root {
    --ivory: #F4F0E8;
    --paper: #FFFEFA;
    --ink: #20231F;
    --muted: #77766D;
    --gold: #B69A62;
    --gold-soft: #E8DCC3;
    --line: #E7E0D3;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background: #F4F0E8 !important;
}

[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 1320px;
    padding-top: 2.5rem;
    padding-bottom: 5rem;
}

h1, h2, h3 {
    color: #20231F !important;
    letter-spacing: -0.04em;
}

.entity-hero {
    background:
        radial-gradient(ellipse at 95% 5%, rgba(182,154,98,.16), transparent 38%),
        linear-gradient(125deg, #FFFEFA 0%, #F0E9DA 100%);
    border: 1px solid #E2D7C3;
    border-radius: 24px;
    padding: 42px 44px;
    box-shadow: 0 12px 35px rgba(45, 39, 27, .045);
}

.entity-hero h1 {
    font-family: Georgia, "Times New Roman", serif;
    font-weight: 500;
    letter-spacing: -0.05em;
}

.entity-card-heading {
    background: rgba(255, 254, 250, .82);
    border: 1px solid #E7E0D3;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 6px 18px rgba(45, 39, 27, .035);
}

.workspace-heading {
    margin-top: 1.5rem;
    margin-bottom: 1.25rem;
}

[data-testid="stMetric"] {
    background: #FFFEFA;
    border: 1px solid #E7E0D3;
    border-radius: 15px;
    padding: 18px 20px;
    box-shadow: 0 5px 16px rgba(45, 39, 27, .035);
}

[data-testid="stTextInput"] input {
    background: #FFFEFA !important;
    border-color: #DED6C7 !important;
    border-radius: 10px !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: #B69A62 !important;
    box-shadow: 0 0 0 2px rgba(182,154,98,.15) !important;
}

[data-testid="stSelectbox"] > div > div {
    border-radius: 10px;
}

hr {
    border-color: #E7E0D3 !important;
}

@media (max-width: 768px) {
    .entity-hero {
        padding: 28px 24px;
        border-radius: 18px;
    }
    [data-testid="stMetric"] {
        padding: 14px;
    }
}
/* END BIZMATCH PREMIUM REFINEMENT */


/* BIZMATCH DISTINCTIVE HERO */
.entity-hero {
    background: linear-gradient(120deg, #20231F 0%, #34382F 68%, #514632 100%) !important;
    border: 1px solid #6D5B3B !important;
    border-radius: 24px !important;
    padding: 52px 48px !important;
    box-shadow: 0 20px 48px rgba(32, 35, 31, 0.14) !important;
}

.entity-hero .entity-eyebrow {
    color: #D6BC83 !important;
    letter-spacing: 0.22em !important;
}

.entity-hero h1 {
    color: #FFFEFA !important;
    max-width: 800px;
    font-size: clamp(2.4rem, 5vw, 4.2rem) !important;
    line-height: 1.08 !important;
}

.entity-hero p {
    color: #E2DED4 !important;
    max-width: 680px;
    font-size: 1.05rem !important;
    line-height: 1.8 !important;
}

.entity-hero .hero-tag {
    display: inline-block;
    margin-top: 16px;
    padding: 9px 13px;
    border: 1px solid rgba(214, 188, 131, .4);
    border-radius: 999px;
    color: #E7D3A6 !important;
    font-size: 10px;
    letter-spacing: .13em;
}

.entity-card-heading {
    border-top: 3px solid #B69A62 !important;
}
/* END BIZMATCH DISTINCTIVE HERO */


/* BIZMATCH PREMIUM DECISION PANEL */
.decision-panel {
    background: linear-gradient(125deg, #20231F, #34382F) !important;
    border: 1px solid #655638 !important;
    border-radius: 22px !important;
    padding: 32px 36px !important;
    margin: 18px 0 22px !important;
    box-shadow: 0 16px 38px rgba(32, 35, 31, .12) !important;
}

.decision-kicker {
    color: #D6BC83 !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: .19em !important;
    margin-bottom: 16px !important;
}

.decision-title {
    color: #FFFEFA !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: clamp(1.7rem, 3vw, 2.4rem) !important;
    font-weight: 500 !important;
    letter-spacing: -.035em !important;
    margin: 18px 0 10px !important;
}

.decision-description {
    color: #E2DED4 !important;
    line-height: 1.75 !important;
}

.decision-pill {
    display: inline-block !important;
    border-radius: 999px !important;
    padding: 7px 13px !important;
    font-size: 10px !important;
    font-weight: 750 !important;
    letter-spacing: .12em !important;
}

.decision-match {
    background: #DCE9D9 !important;
    color: #244D31 !important;
}

.decision-review {
    background: #F2E4C5 !important;
    color: #76551C !important;
}

.decision-nonmatch {
    background: #F0DAD5 !important;
    color: #803C32 !important;
}

.decision-metric {
    background: #FFFEFA !important;
    border: 1px solid #E7E0D3 !important;
    border-radius: 16px !important;
    padding: 22px 24px !important;
    min-height: 112px !important;
    box-shadow: 0 6px 18px rgba(45, 39, 27, .04) !important;
}

.decision-metric-label {
    color: #77766D !important;
    font-size: 11px !important;
    letter-spacing: .09em !important;
    text-transform: uppercase !important;
}

.decision-metric-value {
    color: #20231F !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: 2rem !important;
    margin-top: 9px !important;
    font-variant-numeric: tabular-nums;
}

.decision-note {
    background: #F0E9DA !important;
    border-left: 3px solid #B69A62 !important;
    border-radius: 0 12px 12px 0 !important;
    color: #555348 !important;
    padding: 16px 20px !important;
    line-height: 1.75 !important;
    margin: 18px 0 28px !important;
}

@media (max-width: 768px) {
    .decision-panel {
        padding: 26px 22px !important;
    }
    .decision-metric {
        padding: 18px !important;
    }
}
/* END BIZMATCH PREMIUM DECISION PANEL */


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


@st.cache_resource
def load_bundle():
    """Load and validate the V3 model bundle."""
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: `{MODEL_PATH}`")
        return None

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model file `{MODEL_PATH}`: {e}")
        return None

    if not isinstance(bundle, dict):
        st.error(f"Model bundle format invalid: expected dict, got {type(bundle).__name__}.")
        return None

    missing_keys = REQUIRED_BUNDLE_KEYS - set(bundle.keys())
    if missing_keys:
        st.error(f"Model bundle is missing required keys: {sorted(missing_keys)}")
        return None

    bundle_features = bundle.get("feature_names", [])
    missing_features = [f for f in REQUIRED_FEATURES if f not in bundle_features]
    if missing_features:
        st.error(f"Model bundle is missing required feature definitions: {missing_features}")
        return None

    model = bundle.get("model")
    if not hasattr(model, "predict_proba"):
        st.error("Trained model pipeline does not support probability score estimation (`predict_proba`).")
        return None

    return bundle


PRESETS = {
    "-- Custom Input --": {
        "name1": "", "addr1": "", "ctry1": "US",
        "name2": "", "addr2": "", "ctry2": "US",
    },
    "Example 1: True Match (Suffix & Abbreviation Variation)": {
        "name1": "Acme Industrial Solutions Pvt. Ltd.",
        "addr1": "100 Innovation Way, Suite 400, Austin, TX",
        "ctry1": "US",
        "name2": "Acme Industrial Solutions Private Limited",
        "addr2": "100 Innovation Way, Austin, TX",
        "ctry2": "US",
    },
    "Example 2: Hard Negative (Same Address, Different Business)": {
        "name1": "Starbucks Coffee",
        "addr1": "750 North Highland Mall, Los Angeles, CA",
        "ctry1": "US",
        "name2": "Subway Sandwiches",
        "addr2": "750 North Highland Mall, Los Angeles, CA",
        "ctry2": "US",
    },
    "Example 3: Branch / Country Conflict (Same Name, Different Country)": {
        "name1": "Apex Global Logistics Inc.",
        "addr1": "12 Harbour Road, San Francisco, CA",
        "ctry1": "US",
        "name2": "Apex Global Logistics Inc.",
        "addr2": "45 Cyber City Phase 2, Gurgaon, HR",
        "ctry2": "India",
    },
}


def on_preset_change():
    """Callback to sync input fields when preset selection changes."""
    selected = st.session_state.get("preset_selector", "-- Custom Input --")
    preset = PRESETS.get(selected, PRESETS["-- Custom Input --"])
    st.session_state["n1"] = preset["name1"]
    st.session_state["a1"] = preset["addr1"]
    st.session_state["c1"] = preset["ctry1"]
    st.session_state["n2"] = preset["name2"]
    st.session_state["a2"] = preset["addr2"]
    st.session_state["c2"] = preset["ctry2"]


# Initialize session state keys once
if "initialized" not in st.session_state:
    init_preset = PRESETS["-- Custom Input --"]
    st.session_state["n1"] = init_preset["name1"]
    st.session_state["a1"] = init_preset["addr1"]
    st.session_state["c1"] = init_preset["ctry1"]
    st.session_state["n2"] = init_preset["name2"]
    st.session_state["a2"] = init_preset["addr2"]
    st.session_state["c2"] = init_preset["ctry2"]
    st.session_state["initialized"] = True

st.markdown("""
<div class="entity-hero">
  <div class="entity-eyebrow">Intelligent entity resolution</div>
  <h1>Know when business records truly match.</h1>
  <p>Resolve duplicate business records with explainable similarity signals,
  transparent decisions, and an interactive entity comparison workspace.</p>
  <span class="hero-tag">BIZMATCH &nbsp; / &nbsp; INTELLIGENT ENTITY RESOLUTION</span>
</div>
""", unsafe_allow_html=True)

bundle = load_bundle()
if bundle is None:
    st.stop()

model = bundle["model"]
feature_names = bundle["feature_names"]
threshold = float(bundle.get("threshold", 0.5))

st.selectbox(
    "Load Example Preset:",
    list(PRESETS.keys()),
    key="preset_selector",
    on_change=on_preset_change,
)

st.markdown("""
<div class="workspace-heading">
  <div>
    <div class="section-kicker">WORKSPACE / 01</div>
    <h2>Entity comparison</h2>
    <p>Review both business records before evaluating their identity.</p>
  </div>
  <div class="workspace-status"><span></span> READY TO ANALYSE</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="entity-card-heading">
      <span class="entity-number">01</span>
      <div>
        <div class="entity-card-kicker">SOURCE RECORD</div>
        <div class="entity-card-title">Reference entity</div>
        <div class="entity-card-caption">The original business record</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    name1 = st.text_input("Business name", key="n1", placeholder="e.g. Acme Industries Pvt. Ltd.")
    addr1 = st.text_input("Registered or business address", key="a1", placeholder="Enter the business address")
    ctry1 = st.text_input("Country", key="c1", placeholder="e.g. India")

with col2:
    st.markdown("""
    <div class="entity-card-heading">
      <span class="entity-number">02</span>
      <div>
        <div class="entity-card-kicker">CANDIDATE RECORD</div>
        <div class="entity-card-title">Potential match</div>
        <div class="entity-card-caption">The record being evaluated</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    name2 = st.text_input("Business name", key="n2", placeholder="Enter the candidate business name")
    addr2 = st.text_input("Registered or business address", key="a2", placeholder="Enter the candidate address")
    ctry2 = st.text_input("Country", key="c2", placeholder="e.g. India")

st.markdown("""
<div class="workspace-footnote">
  <span class="footnote-dot"></span>
  Names are required. Addresses and countries help provide additional context.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Graceful input validation
if not name1.strip() or not name2.strip():
    st.info("ℹ️ Please enter business names for both Record 1 and Record 2 to evaluate matching.")
    st.stop()

# Build single-pair DataFrame and run existing V3 feature extraction
pair_df = pd.DataFrame([{
    "source1_name": name1,
    "source1_address": addr1,
    "source1_country": ctry1,
    "candidate_name": name2,
    "candidate_address": addr2,
    "candidate_country": ctry2,
}])

X = make_features(pair_df)[feature_names]
model_score = float(model.predict_proba(X)[0, 1])

# Conservative demo safeguard.
# These rules are heuristics, not independently validated production rules.

name_similarity = float(X["name_similarity"].iloc[0])
name_token_similarity = float(X["name_token_similarity"].iloc[0])

decision, decision_reason = decide_match(
    name1=name1,
    name2=name2,
    country1=ctry1,
    country2=ctry2,
    name_similarity=name_similarity,
    name_token_similarity=name_token_similarity,
    model_score=model_score,
    threshold=threshold,
)


# LIVE MATCHING ANALYTICS
st.markdown("""
<div class="workspace-heading">
  <div>
    <div class="section-kicker">LIVE MATCHING ANALYTICS</div>
    <h2>Signals at a glance</h2>
    <p>Live measurements for the pair currently being evaluated.</p>
  </div>
</div>
""", unsafe_allow_html=True)

metric_cols = st.columns(4, gap="medium")
metric_cols[0].metric("Name similarity", f"{name_similarity:.1%}")
metric_cols[1].metric(
    "Name token similarity", f"{name_token_similarity:.1%}"
)
metric_cols[2].metric(
    "Address similarity", f"{float(X['address_similarity'].iloc[0]):.1%}"
)
metric_cols[3].metric(
    "Model score · uncalibrated", f"{model_score:.3f}"
)

similarity_chart = pd.DataFrame({
    "Signal": [
        "Name",
        "Name tokens",
        "Address",
        "Address tokens",
    ],
    "Similarity": [
        float(X["name_similarity"].iloc[0]),
        float(X["name_token_similarity"].iloc[0]),
        float(X["address_similarity"].iloc[0]),
        float(X["address_token_similarity"].iloc[0]),
    ],
}).set_index("Signal")

with st.expander("Explore similarity signals", expanded=True):
    st.bar_chart(
        similarity_chart,
        horizontal=True,
        x_label="Similarity score",
        y_label="Signal",
    )


# Result display — premium decision panel
st.markdown("""
<div class="workspace-heading">
    <div>
        <div class="section-kicker">ANALYSIS / 02</div>
        <h2>Matching assessment</h2>
        <p>A transparent view of the model output and decision safeguards.</p>
    </div>
</div>
""", unsafe_allow_html=True)

if decision == "MATCH":
    result_class = "decision-match"
    result_label = "MATCH"
    result_title = "Likely the same entity."
    result_description = (
        "The records satisfy the current matching decision rules. "
        "Review the underlying attributes before relying on this result."
    )
elif decision == "REVIEW":
    result_class = "decision-review"
    result_label = "MANUAL REVIEW"
    result_title = "A closer look is needed."
    result_description = (
        "The records triggered a review safeguard. Their identity "
        "should be checked before making a final determination."
    )
else:
    result_class = "decision-nonmatch"
    result_label = "NON-MATCH"
    result_title = "The match criteria were not met."
    result_description = (
        "The current decision rules did not accept these records as a "
        "match. This result is not definitive proof of different entities."
    )

st.markdown(f"""
<div class="decision-panel">
    <div class="decision-kicker">ENTITY RESOLUTION · DECISION SUMMARY</div>
    <span class="decision-pill {result_class}">{result_label}</span>
    <h2 class="decision-title">{result_title}</h2>
    <p class="decision-description">{result_description}</p>
</div>
""", unsafe_allow_html=True)

res_col1, res_col2 = st.columns(2, gap="large")

with res_col1:
    st.markdown(f"""
    <div class="decision-metric">
        <div class="decision-metric-label">Uncalibrated model score</div>
        <div class="decision-metric-value">{model_score:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with res_col2:
    st.markdown(f"""
    <div class="decision-metric">
        <div class="decision-metric-label">Model threshold</div>
        <div class="decision-metric-value">{threshold:.2f}</div>
        <div class="decision-description">
            Score margin: {model_score - threshold:+.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("#### Decision rationale")
st.write(decision_reason)

st.markdown("""
<div class="decision-note">
    <strong>Interpretation note:</strong> The model score is an
    uncalibrated diagnostic score, not a verified probability of identity.
    The review safeguard uses heuristics that have not been independently
    validated for production use.
</div>
""", unsafe_allow_html=True)


# Premium similarity diagnostics
st.markdown("""
<div class="workspace-heading">
    <div>
        <div class="section-kicker">DIAGNOSTICS / 03</div>
        <h2>Similarity diagnostics</h2>
        <p>A closer look at the signals used to compare these records.</p>
    </div>
</div>
""", unsafe_allow_html=True)

similarity_features = [
    (
        "Name similarity",
        float(X["name_similarity"].iloc[0]),
        "Character-level edit similarity between normalized names.",
    ),
    (
        "Name token similarity",
        float(X["name_token_similarity"].iloc[0]),
        "Word-level similarity that tolerates differences in word order.",
    ),
    (
        "Address similarity",
        float(X["address_similarity"].iloc[0]),
        "Character-level edit similarity between normalized addresses.",
    ),
    (
        "Address token similarity",
        float(X["address_token_similarity"].iloc[0]),
        "Token overlap that can tolerate suite numbers or added landmarks.",
    ),
]

for start in range(0, len(similarity_features), 2):
    feature_cols = st.columns(2, gap="large")

    for col, feature in zip(feature_cols, similarity_features[start:start + 2]):
        feature_name, feature_value, feature_description = feature
        display_value = max(0.0, min(1.0, feature_value))

        with col:
            st.markdown(f"""
            <div style="
                background: #FFFEFA;
                border: 1px solid #E8E2D6;
                border-radius: 16px;
                padding: 21px 22px;
                margin-bottom: 14px;
            ">
                <div style="
                    color: #77766D;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 1.2px;
                    text-transform: uppercase;
                    margin-bottom: 10px;
                ">{feature_name}</div>
                <div style="
                    color: #292A25;
                    font-family: Georgia, serif;
                    font-size: 30px;
                    line-height: 1.2;
                    margin-bottom: 15px;
                ">{feature_value:.2%}</div>
                <div style="
                    background: #EDE9DF;
                    height: 6px;
                    border-radius: 99px;
                    overflow: hidden;
                    margin-bottom: 14px;
                ">
                    <div style="
                        background: #607B61;
                        height: 6px;
                        width: {display_value:.2%};
                        border-radius: 99px;
                    "></div>
                </div>
                <div style="
                    color: #77766D;
                    font-size: 12px;
                    line-height: 1.6;
                ">{feature_description}</div>
            </div>
            """, unsafe_allow_html=True)

# Context flags
address_missing = int(X["address_missing"].iloc[0])
country_match = int(X["country_match"].iloc[0])

flag_col1, flag_col2 = st.columns(2, gap="large")

with flag_col1:
    st.markdown(f"""
    <div style="
        background: #F1EDE3;
        border: 1px solid #E8E2D6;
        border-radius: 14px;
        padding: 18px 20px;
    ">
        <div style="color:#77766D;font-size:11px;font-weight:700;
                    letter-spacing:1.1px;text-transform:uppercase;">
            Address availability
        </div>
        <div style="color:#292A25;font-size:17px;font-weight:600;
                    margin-top:8px;">
            {"Address missing" if address_missing else "Both addresses provided"}
        </div>
        <div style="color:#77766D;font-size:12px;margin-top:5px;">
            {"At least one address is blank." if address_missing else "Neither address is blank."}
        </div>
    </div>
    """, unsafe_allow_html=True)

with flag_col2:
    st.markdown(f"""
    <div style="
        background: #F1EDE3;
        border: 1px solid #E8E2D6;
        border-radius: 14px;
        padding: 18px 20px;
    ">
        <div style="color:#77766D;font-size:11px;font-weight:700;
                    letter-spacing:1.1px;text-transform:uppercase;">
            Country comparison
        </div>
        <div style="color:#292A25;font-size:17px;font-weight:600;
                    margin-top:8px;">
            {"Countries match" if country_match else "Countries do not match"}
        </div>
        <div style="color:#77766D;font-size:12px;margin-top:5px;">
            {"The supplied country fields agree." if country_match else "Check the country fields before relying on the result."}
        </div>
    </div>
    """, unsafe_allow_html=True)


# Text normalization diagnostics
st.markdown(
    """
    <div class="section-eyebrow">TECHNICAL DETAILS</div>
    <div class="diagnostics-heading">Normalized Text Diagnostics</div>
    <div class="diagnostics-subtitle">
        Compare the cleaned values used by the matching pipeline.
    </div>
    """,
    unsafe_allow_html=True,
)

diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    st.markdown("**Record 1 · Normalized**")
    st.code(
        f"Name:    {normalize_text(name1)}\n"
        f"Address: {normalize_text(addr1)}\n"
        f"Country: {ctry1.strip().lower()}",
        language=None,
    )

with diag_col2:
    st.markdown("**Record 2 · Normalized**")
    st.code(
        f"Name:    {normalize_text(name2)}\n"
        f"Address: {normalize_text(addr2)}\n"
        f"Country: {ctry2.strip().lower()}",
        language=None,
    )
