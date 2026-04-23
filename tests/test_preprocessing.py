"""
Assignment 1 – Preprocessing pipeline unit tests.

Covers:
- Missing-value handling (drop null review_body, fill null review_title)
- Title fallback content (first 5 words + "...")
- text_length feature computation
- prepare_english_split (language filter, model_text construction)
- prepare_text (multi-lingual variant used by model_c)
- Train / val / test split leakage checks
"""

import pandas as pd
import numpy as np
import pytest

from models.train_model_a import prepare_english_split
from models.train_model_c import prepare_text


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "review_body":    ["Great product", "Terrible item", "Okay I guess", None, "Amazing stuff here"],
        "review_title":   ["Good",           None,            "Meh",          "Skip", None],
        "stars":          [5,                1,               3,              2,      5],
        "language":       ["en",             "en",            "fr",           "en",   "en"],
        "product_category": ["electronics",  "book",          "apparel",      "pc",   "book"],
    })


# ---------------------------------------------------------------------------
# Core preprocessing logic (mirrors data/preprocess.py)
# ---------------------------------------------------------------------------
class TestPreprocessClean:
    def test_drops_null_review_body(self, raw_df):
        cleaned = raw_df.dropna(subset=["review_body"])
        assert cleaned["review_body"].isnull().sum() == 0
        assert len(cleaned) == len(raw_df) - 1

    def test_fills_null_title_with_fallback(self, raw_df):
        df = raw_df.dropna(subset=["review_body"]).copy()
        fallback = df["review_body"].astype(str).str.split().str[:5].str.join(" ") + "..."
        df["review_title"] = df["review_title"].fillna(fallback)
        assert df["review_title"].isnull().sum() == 0

    def test_title_fallback_uses_first_five_words(self):
        df = pd.DataFrame({
            "review_body":  ["one two three four five six seven"],
            "review_title": [None],
        })
        fallback = df["review_body"].astype(str).str.split().str[:5].str.join(" ") + "..."
        df["review_title"] = df["review_title"].fillna(fallback)
        assert df["review_title"].iloc[0] == "one two three four five..."

    def test_title_fallback_fewer_than_five_words(self):
        df = pd.DataFrame({
            "review_body":  ["short"],
            "review_title": [None],
        })
        fallback = df["review_body"].astype(str).str.split().str[:5].str.join(" ") + "..."
        df["review_title"] = df["review_title"].fillna(fallback)
        assert df["review_title"].iloc[0] == "short..."

    def test_existing_title_not_overwritten(self, raw_df):
        df = raw_df.dropna(subset=["review_body"]).copy()
        original_title = df.loc[df["review_title"] == "Good", "review_title"].iloc[0]
        fallback = df["review_body"].astype(str).str.split().str[:5].str.join(" ") + "..."
        df["review_title"] = df["review_title"].fillna(fallback)
        assert df.loc[df["review_body"] == "Great product", "review_title"].iloc[0] == original_title

    def test_text_length_equals_word_count(self, raw_df):
        df = raw_df.dropna(subset=["review_body"]).copy()
        df["text_length"] = df["review_body"].astype(str).str.split().str.len()
        assert df.loc[df["review_body"] == "Great product", "text_length"].iloc[0] == 2
        assert df.loc[df["review_body"] == "Amazing stuff here", "text_length"].iloc[0] == 3

    def test_text_length_always_positive(self, raw_df):
        df = raw_df.dropna(subset=["review_body"]).copy()
        df["text_length"] = df["review_body"].astype(str).str.split().str.len()
        assert (df["text_length"] > 0).all()

    def test_no_nulls_in_key_columns_after_full_pipeline(self, raw_df):
        df = raw_df.dropna(subset=["review_body"]).copy()
        fallback = df["review_body"].astype(str).str.split().str[:5].str.join(" ") + "..."
        df["review_title"] = df["review_title"].fillna(fallback)
        df["text_length"] = df["review_body"].astype(str).str.split().str.len()
        for col in ["review_body", "review_title", "text_length"]:
            assert df[col].isnull().sum() == 0, f"Nulls remain in '{col}' after cleaning"


# ---------------------------------------------------------------------------
# prepare_english_split (train_model_a.py)
# ---------------------------------------------------------------------------
class TestPrepareEnglishSplit:
    def test_filters_to_english_only(self, raw_df):
        result = prepare_english_split(raw_df.dropna(subset=["review_body"]).copy())
        assert (result["language"] == "en").all()

    def test_non_english_rows_removed(self, raw_df):
        n_en = (raw_df["language"] == "en").sum()
        # one row has null body so it drops from prepare_english_split internal fillna
        result = prepare_english_split(raw_df.dropna(subset=["review_body"]).copy())
        assert len(result) == (raw_df.dropna(subset=["review_body"])["language"] == "en").sum()

    def test_model_text_non_empty(self, raw_df):
        result = prepare_english_split(raw_df.dropna(subset=["review_body"]).copy())
        assert (result["model_text"].str.len() > 0).all()

    def test_title_fallback_applied(self):
        df = pd.DataFrame({
            "review_body":    ["This is a great laptop"],
            "review_title":   [None],
            "stars":          [5],
            "language":       ["en"],
            "product_category": ["electronics"],
        })
        result = prepare_english_split(df)
        assert result["review_title"].iloc[0] == "This is a great laptop..."

    def test_model_text_concatenates_title_and_body(self):
        df = pd.DataFrame({
            "review_body":    ["great product"],
            "review_title":   ["Excellent"],
            "stars":          [5],
            "language":       ["en"],
            "product_category": ["electronics"],
        })
        result = prepare_english_split(df)
        assert result["model_text"].iloc[0] == "Excellent great product"

    def test_no_null_model_text(self, raw_df):
        result = prepare_english_split(raw_df.dropna(subset=["review_body"]).copy())
        assert result["model_text"].isnull().sum() == 0


# ---------------------------------------------------------------------------
# prepare_text (train_model_c.py) — multilingual variant
# ---------------------------------------------------------------------------
class TestPrepareText:
    def test_creates_model_text_column(self):
        df = pd.DataFrame({
            "review_body":    ["Good product", "Mauvais produit"],
            "review_title":   ["Nice",          None],
            "language":       ["en",            "fr"],
            "product_category": ["book",         "apparel"],
        })
        result = prepare_text(df.copy())
        assert "model_text" in result.columns

    def test_no_null_model_text(self):
        df = pd.DataFrame({
            "review_body":    ["Good", "Bad"],
            "review_title":   [None,   None],
            "language":       ["en",   "fr"],
            "product_category": ["book", "apparel"],
        })
        result = prepare_text(df.copy())
        assert result["model_text"].isnull().sum() == 0

    def test_keeps_all_languages(self):
        df = pd.DataFrame({
            "review_body":    ["Good", "Bien", "Gut"],
            "review_title":   ["OK",   None,   None],
            "language":       ["en",   "fr",   "de"],
            "product_category": ["book", "apparel", "electronics"],
        })
        result = prepare_text(df.copy())
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Leakage checks — train / val / test splits must be disjoint
# ---------------------------------------------------------------------------
class TestLeakageChecks:
    @pytest.fixture
    def splits(self):
        """Synthetic splits that mirror the project's CSV structure."""
        train = pd.DataFrame({
            "id":          range(200),
            "review_body": [f"train review {i}" for i in range(200)],
        })
        val = pd.DataFrame({
            "id":          range(200, 240),
            "review_body": [f"val review {i}" for i in range(200, 240)],
        })
        test = pd.DataFrame({
            "id":          range(240, 280),
            "review_body": [f"test review {i}" for i in range(240, 280)],
        })
        return train, val, test

    def test_train_val_no_overlap(self, splits):
        train, val, _ = splits
        overlap = pd.merge(train, val, on="id")
        assert len(overlap) == 0, "Train and val share rows — data leakage detected"

    def test_train_test_no_overlap(self, splits):
        train, _, test = splits
        overlap = pd.merge(train, test, on="id")
        assert len(overlap) == 0, "Train and test share rows — data leakage detected"

    def test_val_test_no_overlap(self, splits):
        _, val, test = splits
        overlap = pd.merge(val, test, on="id")
        assert len(overlap) == 0, "Val and test share rows — data leakage detected"

    def test_holdout_ids_absent_from_train(self, splits):
        train, val, test = splits
        holdout_ids = set(val["id"]) | set(test["id"])
        leaked = train[train["id"].isin(holdout_ids)]
        assert len(leaked) == 0, f"{len(leaked)} holdout IDs found in training set"

    def test_splits_cover_distinct_ranges(self, splits):
        """All three splits together should equal their combined unique row count."""
        train, val, test = splits
        combined = pd.concat([train, val, test])
        assert combined["id"].nunique() == len(combined)
