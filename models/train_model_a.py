import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline


def prepare_english_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["language"] == "en"].copy()
    title_fallback = df["review_body"].astype(str).str.split().str[:5].str.join(" ") + "..."
    df["review_title"] = df["review_title"].fillna(title_fallback)
    df["model_text"] = (df["review_title"].astype(str) + " " + df["review_body"].astype(str)).str.strip()
    return df


def main():
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/validation.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df = prepare_english_split(train_df)
    val_df = prepare_english_split(val_df)
    test_df = prepare_english_split(test_df)

    X_train = train_df["model_text"]
    y_train = train_df["stars"]
    X_val = val_df["model_text"]
    y_val = val_df["stars"]
    X_test = test_df["model_text"]
    y_test = test_df["stars"]

    candidates = [
        {"max_features": 15000, "min_df": 2, "C": 1.0},
        {"max_features": 20000, "min_df": 2, "C": 2.0},
        {"max_features": 30000, "min_df": 2, "C": 2.0},
        {"max_features": 30000, "min_df": 3, "C": 3.0},
    ]

    best_score = -1.0
    best_params = None

    for params in candidates:
        candidate_pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=params["max_features"],
                        ngram_range=(1, 2),
                        min_df=params["min_df"],
                        sublinear_tf=True,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1500, C=params["C"])),
            ]
        )
        candidate_pipeline.fit(X_train, y_train)
        val_pred = candidate_pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print("Validation candidate:", params, "accuracy:", val_acc)

        if val_acc > best_score:
            best_score = val_acc
            best_params = params

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    X_final_train = combined_df["model_text"]
    y_final_train = combined_df["stars"]

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=best_params["max_features"],
                    ngram_range=(1, 2),
                    min_df=best_params["min_df"],
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1500, C=best_params["C"])),
        ]
    )

    print("Best validation params:", best_params)
    print("Best validation accuracy:", best_score)

    pipeline.fit(X_final_train, y_final_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    within_1 = (abs(y_pred - y_test) <= 1).mean()

    print(f"MAE: {mae:.3f}")
    print(f"Within-1-Star Accuracy: {within_1:.3f}")
    print(f"Baseline MAE (always predict mode): "
          f"{mean_absolute_error(y_test, [y_train.mode()[0]]*len(y_test)):.3f}")

    joblib.dump(pipeline, "models/saved/model_a.pkl")


if __name__ == "__main__":
    main()
