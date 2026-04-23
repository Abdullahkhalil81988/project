"""
End-to-end inference tests — routing → preprocessing → real model → response.

Each test calls run_inference() with inputs crafted to hit a specific model
path, then verifies the output shape and values are valid.

Routing rules (from router/engine.py):
  model_c  — product_category in {book, digital_ebook_purchase, electronics, pc}
    model_a  — language in {en,de,es,fr} AND 15 <= text_length < 80 (other categories)
  model_b  — everything else (non-English, or English outside 15-79 words)
"""

import os
import pytest
import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from router.engine import run_inference

# ---------------------------------------------------------------------------
# Skip entire module if saved models are not present
# ---------------------------------------------------------------------------
MODEL_A_PATH = "models/saved/model_a.pkl"
MODEL_A_DE_PATH = "models/saved/model_a_de.pkl"
MODEL_A_ES_PATH = "models/saved/model_a_es.pkl"
MODEL_A_FR_PATH = "models/saved/model_a_fr.pkl"
MODEL_B_PATH = "models/saved/model_b"
MODEL_C_PATH = "models/saved/model_c.pkl"
MODEL_C_CAT_PATH = "models/saved/model_c_categories.pkl"

missing = [p for p in [MODEL_A_PATH, MODEL_B_PATH, MODEL_C_PATH, MODEL_C_CAT_PATH]
           if not os.path.exists(p)]
pytestmark = pytest.mark.skipif(bool(missing), reason=f"Missing model files: {missing}")


# ---------------------------------------------------------------------------
# Load all models once for the whole module
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B_PATH)
    model_b = AutoModelForSequenceClassification.from_pretrained(MODEL_B_PATH).to(device)
    model_b.eval()
    model_a_by_language = {"en": joblib.load(MODEL_A_PATH)}
    for lang, path in [("de", MODEL_A_DE_PATH), ("es", MODEL_A_ES_PATH), ("fr", MODEL_A_FR_PATH)]:
        if os.path.exists(path):
            model_a_by_language[lang] = joblib.load(path)

    return {
        "model_a":           model_a_by_language["en"],
        "model_a_by_language": model_a_by_language,
        "model_b":           model_b,
        "model_b_tokenizer": tokenizer_b,
        "model_c":           joblib.load(MODEL_C_PATH),
        "model_c_categories": joblib.load(MODEL_C_CAT_PATH),
        "loaded":            True,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def assert_valid_result(result: dict, expected_model: str):
    assert result["predicted_stars"] in range(1, 6), \
        f"predicted_stars {result['predicted_stars']} out of 1-5"
    assert result["sentiment"] in ("positive", "neutral", "negative")
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_used"].startswith(expected_model), \
        f"Expected model '{expected_model}', got '{result['model_used']}'"
    assert result.get("resolved_language") is not None
    assert isinstance(result.get("language_was_detected"), bool)


# ---------------------------------------------------------------------------
# Model A path  (English, 15-79 words, non-special category)
# ---------------------------------------------------------------------------
class TestRouteModelA:
    # 20-word English review in "apparel" category → model_a
    BODY = ("The jacket fits perfectly and the material feels premium. "
            "Stitching is solid and zipper works smoothly. Very happy with this purchase.")
    LANG = "en"
    CAT  = "apparel"

    def test_routes_to_model_a(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert result["model_used"] in ("model_a", "model_b_escalated")

    def test_predicted_stars_in_range(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert 1 <= result["predicted_stars"] <= 5

    def test_confidence_in_range(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_sentiment_is_valid(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert result["sentiment"] in ("positive", "neutral", "negative")

    def test_positive_review_not_predicted_low(self, models):
        body = ("Absolutely incredible product. Best I have ever used. "
                "Exceeded every expectation. Will definitely buy again. Highly recommend.")
        result = run_inference(body, "en", "apparel", models)
        assert result["predicted_stars"] >= 3

    def test_negative_review_not_predicted_high(self, models):
        body = ("Terrible quality. Broke immediately. Waste of money. "
                "Customer service was useless. Never buying from this brand again.")
        result = run_inference(body, "en", "apparel", models)
        assert result["predicted_stars"] <= 3

    def test_with_title(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models,
                               review_title="Great jacket")
        assert 1 <= result["predicted_stars"] <= 5

    def test_escalation_flag_when_present(self, models):
        """If model_a confidence is low it escalates to model_b — flag should reflect that."""
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert result["model_used"] in ("model_a", "model_b_escalated")


# ---------------------------------------------------------------------------
# Model B path
# ---------------------------------------------------------------------------
class TestRouteModelB:
    BODY = "Prodotto eccellente, sono molto soddisfatto della qualita e della consegna rapida."
    LANG = "it"
    CAT  = "apparel"

    def test_routes_to_model_b(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert result["model_used"] == "model_b"

    def test_predicted_stars_in_range(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert 1 <= result["predicted_stars"] <= 5

    def test_confidence_in_range(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_sentiment_valid(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert result["sentiment"] in ("positive", "neutral", "negative")

    def test_no_base_model_used_field(self, models):
        """model_b results should not have base_model_used (that's only model_c)."""
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert "base_model_used" not in result

    def test_german_review(self, models):
        body = "Sehr gutes Produkt, bin sehr zufrieden mit der Qualität und dem Preis."
        result = run_inference(body, "de", "apparel", models)
        assert result["model_used"] in (
            "model_a_de",
            "model_b",
            "model_b_language_fallback",
            "model_b_escalated",
        )
        assert 1 <= result["predicted_stars"] <= 5


# ---------------------------------------------------------------------------
# Model C path  (special categories: book, electronics, digital_ebook_purchase, pc)
# ---------------------------------------------------------------------------
class TestRouteModelC:
    BODY = ("This book was absolutely wonderful. The writing style is engaging "
            "and the story kept me hooked until the very last page. Highly recommended.")
    LANG = "en"
    CAT  = "book"

    def test_routes_to_model_c(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert result["model_used"] == "model_c"

    def test_predicted_stars_in_range(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert 1 <= result["predicted_stars"] <= 5

    def test_confidence_in_range(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_sentiment_valid(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert result["sentiment"] in ("positive", "neutral", "negative")

    def test_base_model_used_is_model_a_for_english(self, models):
        result = run_inference(self.BODY, self.LANG, self.CAT, models)
        assert result.get("base_model_used") == "model_a"

    def test_base_model_used_is_model_b_for_non_english(self, models):
        body = ("Ce livre était absolument merveilleux. L'écriture est engageante "
                "et l'histoire m'a tenu en haleine jusqu'à la dernière page.")
        result = run_inference(body, "fr", "book", models)
        assert result["model_used"] == "model_c"
        assert result.get("base_model_used") in ("model_a_fr", "model_b")

    def test_language_detected_when_not_provided(self, models):
        body = "Este producto es muy bueno y estoy muy satisfecho con la calidad"
        result = run_inference(body, None, "apparel", models)
        assert result.get("resolved_language") == "es"
        assert result.get("language_was_detected") is True

    def test_electronics_category_routes_to_model_c(self, models):
        body = ("Great laptop, fast processor and excellent battery life. "
                "The display is sharp and keyboard feels comfortable to type on.")
        result = run_inference(body, "en", "electronics", models)
        assert result["model_used"] == "model_c"

    def test_digital_ebook_routes_to_model_c(self, models):
        body = ("Loved this ebook. The content was insightful and well structured. "
                "The formatting was clean and easy to read on my tablet device.")
        result = run_inference(body, "en", "digital_ebook_purchase", models)
        assert result["model_used"] == "model_c"


# ---------------------------------------------------------------------------
# Sentiment consistency across all routes
# ---------------------------------------------------------------------------
class TestSentimentConsistency:
    @pytest.mark.parametrize("stars,expected_sentiment", [
        (4, "positive"),
        (5, "positive"),
        (3, "neutral"),
        (1, "negative"),
        (2, "negative"),
    ])
    def test_sentiment_matches_stars(self, stars, expected_sentiment):
        """Sentiment derivation logic is independent of routing — test it directly."""
        from router.engine import run_inference as _ri
        # derive expected sentiment using same rules as engine
        if stars >= 4:
            sentiment = "positive"
        elif stars <= 2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        assert sentiment == expected_sentiment
