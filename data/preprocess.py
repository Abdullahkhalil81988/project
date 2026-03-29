import pandas as pd


def main():
    df = pd.read_csv("data/train.csv")

    print("Before cleaning shape:", df.shape)
    print("Before missing values:")
    print(df[["stars", "review_body", "review_title", "language", "product_category"]].isnull().sum())

    cleaned = df[["stars", "review_body", "review_title", "language", "product_category"]].copy()
    cleaned = cleaned.dropna(subset=["review_body"])

    title_fallback = (
        cleaned["review_body"]
        .astype(str)
        .str.split()
        .str[:5]
        .str.join(" ")
        + "..."
    )
    cleaned["review_title"] = cleaned["review_title"].fillna(title_fallback)

    cleaned["text_length"] = cleaned["review_body"].astype(str).str.split().str.len()

    print("After cleaning shape:", cleaned.shape)
    print("After missing values:")
    print(cleaned[["stars", "review_body", "review_title", "language", "product_category", "text_length"]].isnull().sum())

    cleaned.to_csv("data/cleaned.csv", index=False)


if __name__ == "__main__":
    main()
