import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


LANGUAGES = ["de", "es", "fr"]
RANDOM_STATE = 42


def build_model_text(df: pd.DataFrame) -> pd.Series:
    title_fallback = df["review_body"].astype(str).str.split().str[:5].str.join(" ") + "..."
    review_title = df["review_title"].fillna(title_fallback).astype(str)
    review_body = df["review_body"].astype(str)
    return (review_title + " " + review_body).str.strip()


def select_language(df: pd.DataFrame, language: str) -> pd.DataFrame:
    subset = df[df["language"] == language].copy()
    subset["model_text"] = build_model_text(subset)
    subset["stars"] = subset["stars"].astype(int)
    return subset


def evaluate_split(model: Pipeline, split_df: pd.DataFrame) -> dict:
    y_true = split_df["stars"].to_numpy(dtype=np.int64)
    X_eval = split_df["model_text"].astype(str).tolist()
    y_pred = model.predict(X_eval)

    accuracy = float(accuracy_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    within_1 = float((np.abs(y_pred - y_true) <= 1).mean())

    return {
        "accuracy": accuracy,
        "mae": mae,
        "within_1_star_accuracy": within_1,
    }


def train_one_language(
    language: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    lang_train = select_language(train_df, language)
    lang_val = select_language(val_df, language)
    lang_test = select_language(test_df, language)

    if len(lang_train) == 0 or len(lang_val) == 0 or len(lang_test) == 0:
        raise ValueError(f"Missing rows for language '{language}' in one or more splits.")

    X_train: list[str] = lang_train["model_text"].astype(str).tolist()
    y_train = lang_train["stars"].to_numpy(dtype=np.int64)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "tfidf__analyzer": ["word"],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [15000, 25000, 40000],
        "tfidf__min_df": [1, 2, 3],
        "tfidf__sublinear_tf": [True],
        "clf__C": [0.5, 1.0, 2.0, 3.0],
        "clf__solver": ["lbfgs"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    print(f"\n=== Language: {language} ===")
    print(f"Train rows: {len(lang_train)} | Val rows: {len(lang_val)} | Test rows: {len(lang_test)}")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    val_metrics = evaluate_split(best_model, lang_val)
    test_metrics = evaluate_split(best_model, lang_test)

    model_path = output_dir / f"model_a_{language}.pkl"
    metadata_path = output_dir / f"model_a_{language}_metadata.json"

    joblib.dump(best_model, model_path)

    metadata = {
        "language": language,
        "cv_best_accuracy": float(search.best_score_),
        "best_params": search.best_params_,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_rows": int(len(lang_train)),
        "validation_rows": int(len(lang_val)),
        "test_rows": int(len(lang_test)),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Best CV accuracy:", metadata["cv_best_accuracy"])
    print("Best params:", metadata["best_params"])
    print("Validation metrics:", metadata["validation_metrics"])
    print("Test metrics:", metadata["test_metrics"])
    print("Saved model to:", model_path)
    print("Saved metadata to:", metadata_path)

    return metadata


def main() -> None:
    # Resolve paths from the repository root so the script works from any cwd.
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    output_dir = project_root / "models" / "saved"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for language in LANGUAGES:
        summary[language] = train_one_language(
            language=language,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_dir=output_dir,
        )

    summary_path = output_dir / "model_a_multilang_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved summary to:", summary_path)


if __name__ == "__main__":
    main()
