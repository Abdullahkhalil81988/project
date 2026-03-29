import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


QUICK_MODE = False
MODEL_B_MAX_LENGTH = 128


def prepare_text(df: pd.DataFrame) -> pd.DataFrame:
    title_fallback = df["review_body"].astype(str).str.split().str[:5].str.join(" ") + "..."
    df["review_title"] = df["review_title"].fillna(title_fallback)
    df["model_text"] = (df["review_title"].astype(str) + " " + df["review_body"].astype(str)).str.strip()
    return df


def get_base_probabilities(df: pd.DataFrame, model_a, tokenizer_b, model_b, device: str, batch_size: int = 64) -> np.ndarray:
    probs = np.zeros((len(df), 5), dtype=np.float32)

    en_mask = df["language"] == "en"
    if en_mask.any():
        en_texts = df.loc[en_mask, "model_text"].tolist()
        probs[en_mask.to_numpy()] = model_a.predict_proba(en_texts)

    non_en_idx = np.where((~en_mask).to_numpy())[0]
    total_batches = (len(non_en_idx) + batch_size - 1) // batch_size
    for i, start in enumerate(range(0, len(non_en_idx), batch_size), start=1):
        batch_idx = non_en_idx[start:start + batch_size]
        texts = df.iloc[batch_idx]["model_text"].tolist()
        inputs = tokenizer_b(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MODEL_B_MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            logits = model_b(**inputs).logits
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs[batch_idx] = batch_probs

        if i % 50 == 0 or i == total_batches:
            print(f"Processed non-en batches: {i}/{total_batches}")

    return probs


def main():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df = prepare_text(train_df)
    test_df = prepare_text(test_df)

    if QUICK_MODE:
        train_df = train_df.sample(n=min(30000, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(8000, len(test_df)), random_state=42)

    model_a = joblib.load("models/saved/model_a.pkl")
    tokenizer_b = AutoTokenizer.from_pretrained("models/saved/model_b")
    model_b = AutoModelForSequenceClassification.from_pretrained("models/saved/model_b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_b.to(device)
    model_b.eval()

    print("Generating train base probabilities...")
    train_probs = get_base_probabilities(train_df, model_a, tokenizer_b, model_b, device=device)
    print("Generating test base probabilities...")
    test_probs = get_base_probabilities(test_df, model_a, tokenizer_b, model_b, device=device)

    prob_cols = [f"prob_{i}" for i in range(1, 6)]
    train_prob_df = pd.DataFrame(train_probs, columns=prob_cols)
    test_prob_df = pd.DataFrame(test_probs, columns=prob_cols)

    train_category_df = pd.get_dummies(train_df["product_category"], prefix="category")
    test_category_df = pd.get_dummies(test_df["product_category"], prefix="category")
    test_category_df = test_category_df.reindex(columns=train_category_df.columns, fill_value=0)

    X_train = pd.concat([train_prob_df, train_category_df.reset_index(drop=True)], axis=1)
    y_train = train_df["stars"].astype(int)

    X_test = pd.concat([test_prob_df, test_category_df.reset_index(drop=True)], axis=1)
    y_test = test_df["stars"].astype(int)

    meta_model = LogisticRegression(max_iter=500)
    meta_model.fit(X_train, y_train)

    y_pred = meta_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(meta_model, "models/saved/model_c.pkl")
    joblib.dump(list(train_category_df.columns), "models/saved/model_c_categories.pkl")


if __name__ == "__main__":
    main()
