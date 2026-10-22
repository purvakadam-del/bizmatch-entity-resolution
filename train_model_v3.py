import re
import unicodedata
import joblib
import pandas as pd

from unidecode import unidecode
from rapidfuzz import fuzz
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = "labeled_pairs_v2.csv"
MODEL_PATH = "business_match_model_v3.joblib"


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    value = unidecode(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def make_features(df):
    features = pd.DataFrame(index=df.index)

    left_name = df["source1_name"].map(normalize_text)
    right_name = df["candidate_name"].map(normalize_text)
    left_address = df["source1_address"].map(normalize_text)
    right_address = df["candidate_address"].map(normalize_text)

    features["name_similarity"] = [
        fuzz.ratio(a, b) / 100.0 if a and b else 0.0
        for a, b in zip(left_name, right_name)
    ]

    features["name_token_similarity"] = [
        fuzz.token_sort_ratio(a, b) / 100.0
        for a, b in zip(left_name, right_name)
    ]

    features["address_similarity"] = [
        fuzz.ratio(a, b) / 100.0 if a and b else 0.0
        for a, b in zip(left_address, right_address)
    ]

    features["address_token_similarity"] = [
        fuzz.token_set_ratio(a, b) / 100.0
        for a, b in zip(left_address, right_address)
    ]

    features["address_missing"] = (
        (left_address == "") | (right_address == "")
    ).astype(int)

    left_country = df["source1_country"].fillna("").astype(str).str.strip().str.lower()
    right_country = df["candidate_country"].fillna("").astype(str).str.strip().str.lower()

    features["country_match"] = (
        (left_country != "") &
        (right_country != "") &
        (left_country == right_country)
    ).astype(int)

    return features.fillna(0)


def main():
    print("Loading V2 dataset...")
    df = pd.read_csv(DATA_PATH).fillna("")

    required = [
        "source1_id", "source1_name", "source1_address",
        "source1_country", "candidate_id", "candidate_name",
        "candidate_address", "candidate_country", "label",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["label"] = df["label"].astype(int)

    print("Dataset shape:", df.shape)
    print("Extracting features...")
    X = make_features(df)
    y = df["label"]

    feature_names = list(X.columns)

    # All pairs belonging to one reference stay in one split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=df["source1_id"])
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("Training pairs:", len(train_idx))
    print("Testing pairs:", len(test_idx))
    print("Training references:", df.iloc[train_idx]["source1_id"].nunique())
    print("Testing references:", df.iloc[test_idx]["source1_id"].nunique())

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n===== MODEL V3 RESULTS =====")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
    print("F1 score:", round(f1_score(y_test, predictions, zero_division=0), 4))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Non-match", "Match"],
        zero_division=0,
    ))

    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "normalization": "NFKC + Unidecode + lowercase + alphanumeric tokens",
            "training_dataset": DATA_PATH,
            "threshold": 0.5,
        },
        MODEL_PATH,
    )

    print(f"\nSaved new model to {MODEL_PATH}")
    print("Note: predicted scores are not calibrated probabilities.")
    print("Evaluation labels use the dataset's unlisted-ID assumption.")


if __name__ == "__main__":
    main()
