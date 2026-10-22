# BizMatch — Business Entity Resolution

**An ML-powered application for identifying whether two business records refer to the same real-world business.**

BizMatch compares business names, addresses, and country information to help identify duplicate business records. It combines machine-learning predictions with a decision layer that can classify record pairs as a match, non-match, or requiring manual review.

## Live Demo

- **Primary demo:** https://bizmatch-entity-resolution.onrender.com
- **Alternative demo:** https://bizmatch-entity-resolution.streamlit.app/

## The Problem

Business datasets often contain multiple records for the same company because of differences in spelling, formatting, abbreviations, or address representation.

Traditional exact-string matching can miss these duplicates. BizMatch uses text-similarity features and a machine-learning model to assess pairs of business records.

## Features

- **Individual matching:** Compare two business records.
- **Batch matching:** Process multiple record pairs.
- **Model evaluation:** Explore the model evaluation interface.
- **Decision outcomes:** MATCH, NON-MATCH, or REVIEW.
- **Text normalization:** Standardize text before calculating similarities.
- **Similarity-based features:** Compare business names and addresses.
- **Interactive web interface:** Explore matching results through a Streamlit application.

## How It Works

1. **Input:** Provide business records containing relevant fields.
2. **Normalize:** Standardize text to reduce differences caused by formatting.
3. **Extract features:** Calculate similarity measures for names and addresses, alongside relevant indicators.
4. **Predict:** Use a trained Logistic Regression model to estimate the likelihood of a match.
5. **Apply decision rules:** Convert model outputs into a matching decision, including a review option where appropriate.

The machine-learning model and the decision-policy layer are separate components.

## Technology Stack

- **Language:** Python
- **Machine Learning:** scikit-learn, Logistic Regression
- **Data Processing:** pandas, NumPy
- **Text Similarity:** RapidFuzz
- **Model Persistence:** joblib
- **Web Application:** Streamlit
- **Version Control:** Git and GitHub

## Project Structure

bizmatch-entity-resolution/
├── app.py
├── decision_rules.py
├── train_model_v3.py
├── business_match_model_v3.joblib
├── requirements.txt
└── pages/
    ├── 2_Batch_Matching.py
    └── 3_Model_Evaluation.py
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/purvakadam-del/bizmatch-entity-resolution.git
cd bizmatch-entity-resolution
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## Important Notes

- Matching predictions are decision-support outputs, not definitive proof that two records represent the same business.
- REVIEW outcomes allow uncertain cases to be examined rather than forcing every pair into an automatic decision.
- Model performance depends on the dataset and evaluation methodology. No performance figure is claimed here without a verified evaluation report.

## Future Improvements

- Expand evaluation across diverse business-record variations.
- Improve handling of difficult non-matching pairs.
- Add more transparent explanations for individual decisions.
- Strengthen automated testing and deployment reliability.

## Author

**Purva Kadam**

GitHub: https://github.com/purvakadam-del

---

*Built as a machine-learning project exploring business entity resolution, text similarity, and decision-support systems.*
