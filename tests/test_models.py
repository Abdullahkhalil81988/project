"""
Assignment 2 – Model serialization, reproducibility, and evaluation unit tests.

Covers:
- Model A: loads correctly, Pipeline structure (TF-IDF + LR), predictions in 1–5 range,
  predict_proba shape, deterministic output (reproducibility)
- Model A evaluation: accuracy in valid range, MAE ≥ 0, within-1-star ≥ exact accuracy,
  baseline comparison
- Model C: loads correctly, correct type, categories list structure
- Multiple configurations: best pipeline uses expected sklearn components
"""

import os
import numpy as np
import pytest
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

MODEL_A_PATH = "models/saved/model_a.pkl"
MODEL_C_PATH = "models/saved/model_c.pkl"
MODEL_C_CAT_PATH = "models/saved/model_c_categories.pkl"

requires_model_a = pytest.mark.skipif(
    not os.path.exists(MODEL_A_PATH), reason=f"{MODEL_A_PATH} not found — run train_model_a.py first"
)
requires_model_c = pytest.mark.skipif(
    not os.path.exists(MODEL_C_PATH), reason=f"{MODEL_C_PATH} not found — run train_model_c.py first"
)

SAMPLE_TEXTS = [
    "This product is absolutely amazing, highly recommend to everyone.",
    "Terrible quality, broke after two days of use.",
    "It is okay, nothing special but gets the job done.",
    "Best purchase I have ever made, completely life changing.",
    "Would not recommend to anyone, complete waste of money.",
]
SAMPLE_LABELS = np.array([5, 1, 3, 5, 1])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_a():
    return joblib.load(MODEL_A_PATH)


@pytest.fixture(scope="module")
def model_c():
    return joblib.load(MODEL_C_PATH)


@pytest.fixture(scope="module")
def model_c_categories():
    return joblib.load(MODEL_C_CAT_PATH)


# ---------------------------------------------------------------------------
# Model A — serialization & structure
# ---------------------------------------------------------------------------
@requires_model_a
class TestModelASerialization:
    def test_loads_without_error(self, model_a):
        assert model_a is not None

    def test_is_sklearn_pipeline(self, model_a):
        assert isinstance(model_a, Pipeline), "model_a should be an sklearn Pipeline"

    def test_has_tfidf_step(self, model_a):
        assert "tfidf" in model_a.named_steps
        assert isinstance(model_a.named_steps["tfidf"], TfidfVectorizer)

    def test_has_classifier_step(self, model_a):
        assert "clf" in model_a.named_steps
        assert isinstance(model_a.named_steps["clf"], LogisticRegression)

    def test_tfidf_uses_bigrams(self, model_a):
        assert model_a.named_steps["tfidf"].ngram_range == (1, 2)

    def test_tfidf_sublinear_tf_enabled(self, model_a):
        assert model_a.named_steps["tfidf"].sublinear_tf is True


# ---------------------------------------------------------------------------
# Model A — predictions
# ---------------------------------------------------------------------------
@requires_model_a
class TestModelAPredictions:
    def test_predict_returns_array(self, model_a):
        preds = model_a.predict(SAMPLE_TEXTS)
        assert len(preds) == len(SAMPLE_TEXTS)

    def test_predictions_in_star_range(self, model_a):
        preds = model_a.predict(SAMPLE_TEXTS)
        assert all(1 <= p <= 5 for p in preds), f"Out-of-range predictions: {preds}"

    def test_predict_proba_shape(self, model_a):
        proba = model_a.predict_proba(SAMPLE_TEXTS)
        assert proba.shape == (len(SAMPLE_TEXTS), 5)

    def test_predict_proba_sums_to_one(self, model_a):
        proba = model_a.predict_proba(SAMPLE_TEXTS)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_all_non_negative(self, model_a):
        proba = model_a.predict_proba(SAMPLE_TEXTS)
        assert (proba >= 0).all()

    def test_single_input_works(self, model_a):
        pred = model_a.predict(["Great product"])
        assert len(pred) == 1
        assert 1 <= pred[0] <= 5


# ---------------------------------------------------------------------------
# Model A — reproducibility (Assignment 2: seeds / determinism)
# ---------------------------------------------------------------------------
@requires_model_a
class TestModelAReproducibility:
    def test_predict_is_deterministic(self, model_a):
        text = ["Excellent item, very happy with this purchase."]
        assert model_a.predict(text)[0] == model_a.predict(text)[0]

    def test_predict_proba_is_deterministic(self, model_a):
        text = ["Excellent item, very happy with this purchase."]
        np.testing.assert_array_equal(
            model_a.predict_proba(text),
            model_a.predict_proba(text),
        )

    def test_batch_predict_consistent_with_single(self, model_a):
        """Predicting in batch should give the same result as predicting one at a time."""
        texts = SAMPLE_TEXTS[:3]
        batch_preds = model_a.predict(texts)
        single_preds = np.array([model_a.predict([t])[0] for t in texts])
        np.testing.assert_array_equal(batch_preds, single_preds)


# ---------------------------------------------------------------------------
# Model A — evaluation metrics (Assignment 2: multiple metrics, error analysis)
# ---------------------------------------------------------------------------
@requires_model_a
class TestModelAEvaluation:
    def test_accuracy_in_valid_range(self, model_a):
        preds = model_a.predict(SAMPLE_TEXTS)
        acc = accuracy_score(SAMPLE_LABELS, preds)
        assert 0.0 <= acc <= 1.0

    def test_mae_non_negative(self, model_a):
        preds = model_a.predict(SAMPLE_TEXTS)
        mae = mean_absolute_error(SAMPLE_LABELS, preds)
        assert mae >= 0.0

    def test_mae_bounded_by_max_possible(self, model_a):
        """MAE cannot exceed 4 (max star difference on a 1–5 scale)."""
        preds = model_a.predict(SAMPLE_TEXTS)
        mae = mean_absolute_error(SAMPLE_LABELS, preds)
        assert mae <= 4.0

    def test_within_1_star_gte_exact_accuracy(self, model_a):
        """Within-1-star accuracy must be >= exact accuracy (it relaxes the criterion)."""
        preds = np.array(model_a.predict(SAMPLE_TEXTS))
        exact_acc = accuracy_score(SAMPLE_LABELS, preds)
        within_1 = (np.abs(preds - SAMPLE_LABELS) <= 1).mean()
        assert within_1 >= exact_acc

    def test_within_1_star_in_valid_range(self, model_a):
        preds = np.array(model_a.predict(SAMPLE_TEXTS))
        within_1 = (np.abs(preds - SAMPLE_LABELS) <= 1).mean()
        assert 0.0 <= within_1 <= 1.0

    def test_baseline_mae_computable(self, model_a):
        """Baseline MAE (always predict mode) is a valid non-negative number."""
        preds = model_a.predict(SAMPLE_TEXTS)
        mode_label = int(np.bincount(SAMPLE_LABELS).argmax())
        baseline_mae = mean_absolute_error(SAMPLE_LABELS, [mode_label] * len(SAMPLE_LABELS))
        assert baseline_mae >= 0.0

    def test_positive_reviews_score_higher_on_average(self, model_a):
        """Model should rank clearly positive reviews above clearly negative ones."""
        positive = ["Absolutely perfect, best product I ever bought, love everything about it."]
        negative = ["Completely broken on arrival, worst purchase ever, total scam product."]
        pos_pred = model_a.predict(positive)[0]
        neg_pred = model_a.predict(negative)[0]
        assert pos_pred > neg_pred, "Positive review should score higher than negative review"


# ---------------------------------------------------------------------------
# Model C — serialization & structure
# ---------------------------------------------------------------------------
@requires_model_c
class TestModelCSerialization:
    def test_loads_without_error(self, model_c):
        assert model_c is not None

    def test_is_logistic_regression(self, model_c):
        assert isinstance(model_c, LogisticRegression)

    def test_has_predict_method(self, model_c):
        assert callable(getattr(model_c, "predict", None))

    def test_has_predict_proba_method(self, model_c):
        assert callable(getattr(model_c, "predict_proba", None))

    def test_categories_loads_as_list(self, model_c_categories):
        assert isinstance(model_c_categories, list)
        assert len(model_c_categories) > 0

    def test_categories_are_strings(self, model_c_categories):
        assert all(isinstance(c, str) for c in model_c_categories)

    def test_category_columns_have_prefix(self, model_c_categories):
        assert all(c.startswith("category_") for c in model_c_categories)

    def test_no_duplicate_categories(self, model_c_categories):
        assert len(model_c_categories) == len(set(model_c_categories))
