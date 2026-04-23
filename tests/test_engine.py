"""
Assignment 1 & 2 – Routing engine unit tests.

Covers:
- preprocess_incoming_review: body cleaning, title fallback, model_text, text_length
- select_model: all routing branches and boundary conditions
"""

import pytest
from router.engine import detect_language, preprocess_incoming_review, select_model


# ---------------------------------------------------------------------------
# preprocess_incoming_review
# ---------------------------------------------------------------------------
class TestPreprocessIncomingReview:
    def test_returns_all_required_keys(self):
        result = preprocess_incoming_review("Good product", review_title="Nice")
        assert set(result.keys()) == {"review_body", "review_title", "model_text", "text_length"}

    def test_body_stripped(self):
        result = preprocess_incoming_review("  Great product  ")
        assert result["review_body"] == "Great product"

    def test_text_length_word_count(self):
        result = preprocess_incoming_review("one two three four")
        assert result["text_length"] == 4

    def test_text_length_single_word(self):
        result = preprocess_incoming_review("excellent")
        assert result["text_length"] == 1

    def test_title_preserved_when_provided(self):
        result = preprocess_incoming_review("Good product", review_title="My Review")
        assert result["review_title"] == "My Review"

    def test_title_fallback_when_none(self):
        result = preprocess_incoming_review("one two three four five six")
        assert result["review_title"] == "one two three four five..."

    def test_title_fallback_when_empty_string(self):
        result = preprocess_incoming_review("one two three", review_title="")
        assert result["review_title"] == "one two three..."

    def test_title_fallback_when_whitespace_string(self):
        result = preprocess_incoming_review("alpha beta gamma", review_title="   ")
        assert result["review_title"] == "alpha beta gamma..."

    def test_title_fallback_short_body(self):
        result = preprocess_incoming_review("hi")
        assert result["review_title"] == "hi..."

    def test_title_fallback_exactly_five_words(self):
        result = preprocess_incoming_review("a b c d e")
        assert result["review_title"] == "a b c d e..."

    def test_model_text_title_plus_body(self):
        result = preprocess_incoming_review("good product", review_title="Nice item")
        assert result["model_text"] == "Nice item good product"

    def test_model_text_with_fallback_title(self):
        result = preprocess_incoming_review("great product here")
        # fallback title is "great product here..." + " " + "great product here"
        assert result["model_text"].startswith("great product here...")


# ---------------------------------------------------------------------------
# select_model — routing logic
# ---------------------------------------------------------------------------
class TestSelectModel:
    # --- Model C: special product categories ---
    @pytest.mark.parametrize("category", [
        "book", "digital_ebook_purchase", "electronics", "pc",
    ])
    def test_special_categories_route_to_model_c(self, category):
        assert select_model("en", 40, category) == "model_c"

    def test_model_c_overrides_language(self):
        """Special categories take precedence — non-English still goes to model_c."""
        assert select_model("fr", 40, "book") == "model_c"
        assert select_model("de", 40, "electronics") == "model_c"

    def test_model_c_overrides_text_length(self):
        """Special categories take precedence regardless of text length."""
        assert select_model("en", 0, "electronics") == "model_c"
        assert select_model("en", 500, "electronics") == "model_c"

    # --- Model A: English with 15 ≤ length < 80 ---
    def test_english_mid_length_routes_to_model_a(self):
        assert select_model("en", 15, "apparel") == "model_a"
        assert select_model("en", 40, "apparel") == "model_a"
        assert select_model("en", 79, "apparel") == "model_a"

    def test_lower_boundary_15_is_model_a(self):
        assert select_model("en", 15, "apparel") == "model_a"

    def test_upper_boundary_79_is_model_a(self):
        assert select_model("en", 79, "apparel") == "model_a"

    # --- Model B fallbacks ---
    def test_english_below_15_routes_to_model_b(self):
        assert select_model("en", 14, "apparel") == "model_b"
        assert select_model("en", 0, "apparel") == "model_b"

    def test_english_at_80_routes_to_model_b(self):
        assert select_model("en", 80, "apparel") == "model_b"

    def test_english_above_80_routes_to_model_b(self):
        assert select_model("en", 200, "apparel") == "model_b"

    @pytest.mark.parametrize("lang", ["ja", "ar", "ru", "it"])
    def test_non_english_routes_to_model_b(self, lang):
        assert select_model(lang, 40, "apparel") == "model_b"

    def test_non_english_mid_length_not_model_a(self):
        """Even with ideal text length, non-English should not go to model_a."""
        assert select_model("it", 40, "kitchen") != "model_a"

    @pytest.mark.parametrize("lang", ["de", "es", "fr"])
    def test_model_a_supported_multilingual_routes_to_model_a(self, lang):
        assert select_model(lang, 40, "apparel") == "model_a"


class TestLanguageDetection:
    def test_detect_language_german(self):
        detected = detect_language("Das Produkt ist sehr gut und ich bin sehr zufrieden mit der Qualitat")
        assert detected == "de"

    def test_detect_language_spanish(self):
        detected = detect_language("Este producto es muy bueno y estoy muy satisfecho con la calidad")
        assert detected == "es"

    def test_detect_language_french(self):
        detected = detect_language("Ce produit est tres bon et je suis tres satisfait de la qualite")
        assert detected == "fr"
