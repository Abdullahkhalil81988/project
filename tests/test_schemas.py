"""
Assignment 2 & 3 – API schema validation unit tests.

Covers:
- ReviewRequest: required fields, empty/whitespace rejection, length limits,
  language/category normalisation, title stripping
- PredictionResponse: star range, sentiment literals, confidence bounds
- HealthResponse: status literals
"""

import pytest
from pydantic import ValidationError

from api.schemas import ReviewRequest, PredictionResponse, HealthResponse


# ---------------------------------------------------------------------------
# ReviewRequest
# ---------------------------------------------------------------------------
class TestReviewRequestValid:
    def test_minimal_valid_request(self):
        req = ReviewRequest(review_body="Great product", language="en", product_category="electronics")
        assert req.review_body == "Great product"
        assert req.language == "en"
        assert req.product_category == "electronics"
        assert req.review_title is None

    def test_full_valid_request(self):
        req = ReviewRequest(
            review_body="Exceptional quality.",
            review_title="Best purchase ever",
            language="en",
            product_category="electronics",
        )
        assert req.review_title == "Best purchase ever"

    def test_language_uppercased_normalised(self):
        req = ReviewRequest(review_body="Good", language="EN", product_category="book")
        assert req.language == "en"

    def test_language_with_spaces_stripped(self):
        req = ReviewRequest(review_body="Good", language=" en ", product_category="book")
        assert req.language == "en"

    def test_category_uppercased_normalised(self):
        req = ReviewRequest(review_body="Good", language="en", product_category="Electronics")
        assert req.product_category == "electronics"

    def test_category_mixed_case_normalised(self):
        req = ReviewRequest(review_body="Good", language="en", product_category="BOOK")
        assert req.product_category == "book"

    def test_title_whitespace_stripped(self):
        req = ReviewRequest(
            review_body="Good", review_title="  My Title  ",
            language="en", product_category="electronics",
        )
        assert req.review_title == "My Title"

    def test_title_all_whitespace_becomes_none(self):
        req = ReviewRequest(
            review_body="Good", review_title="   ",
            language="en", product_category="electronics",
        )
        assert req.review_title is None

    def test_title_none_accepted(self):
        req = ReviewRequest(review_body="Good", language="en", product_category="electronics")
        assert req.review_title is None

    def test_multilingual_language_code(self):
        req = ReviewRequest(review_body="Très bien", language="fr", product_category="apparel")
        assert req.language == "fr"


class TestReviewRequestInvalid:
    def test_missing_review_body_rejected(self):
        with pytest.raises(ValidationError):
            ReviewRequest(language="en", product_category="electronics")

    def test_missing_language_rejected(self):
        with pytest.raises(ValidationError):
            ReviewRequest(review_body="Good", product_category="electronics")

    def test_missing_product_category_rejected(self):
        with pytest.raises(ValidationError):
            ReviewRequest(review_body="Good", language="en")

    def test_empty_body_rejected(self):
        with pytest.raises(ValidationError):
            ReviewRequest(review_body="", language="en", product_category="electronics")

    def test_whitespace_only_body_rejected(self):
        with pytest.raises(ValidationError):
            ReviewRequest(review_body="   ", language="en", product_category="electronics")

    def test_tab_only_body_rejected(self):
        with pytest.raises(ValidationError):
            ReviewRequest(review_body="\t\n  ", language="en", product_category="electronics")

    def test_body_exceeds_max_length_rejected(self):
        with pytest.raises(ValidationError):
            ReviewRequest(
                review_body="x" * 10001,
                language="en",
                product_category="electronics",
            )

    def test_body_at_max_length_accepted(self):
        req = ReviewRequest(
            review_body="x" * 10000,
            language="en",
            product_category="electronics",
        )
        assert len(req.review_body) == 10000

    def test_language_too_short_rejected(self):
        with pytest.raises(ValidationError):
            ReviewRequest(review_body="Good", language="e", product_category="electronics")

    def test_empty_product_category_rejected(self):
        with pytest.raises(ValidationError):
            ReviewRequest(review_body="Good", language="en", product_category="")


# ---------------------------------------------------------------------------
# PredictionResponse
# ---------------------------------------------------------------------------
class TestPredictionResponse:
    def _base(self, **kwargs):
        defaults = dict(
            predicted_stars=4,
            sentiment="positive",
            confidence=0.87,
            model_used="model_a",
        )
        defaults.update(kwargs)
        return PredictionResponse(**defaults)

    def test_valid_response_constructed(self):
        resp = self._base()
        assert resp.predicted_stars == 4
        assert resp.sentiment == "positive"

    @pytest.mark.parametrize("stars", [1, 2, 3, 4, 5])
    def test_all_valid_star_ratings_accepted(self, stars):
        resp = self._base(predicted_stars=stars)
        assert resp.predicted_stars == stars

    def test_stars_zero_rejected(self):
        with pytest.raises(ValidationError):
            self._base(predicted_stars=0)

    def test_stars_six_rejected(self):
        with pytest.raises(ValidationError):
            self._base(predicted_stars=6)

    @pytest.mark.parametrize("sentiment", ["positive", "neutral", "negative"])
    def test_all_valid_sentiments_accepted(self, sentiment):
        resp = self._base(sentiment=sentiment)
        assert resp.sentiment == sentiment

    def test_invalid_sentiment_rejected(self):
        with pytest.raises(ValidationError):
            self._base(sentiment="great")

    def test_confidence_zero_accepted(self):
        resp = self._base(confidence=0.0)
        assert resp.confidence == 0.0

    def test_confidence_one_accepted(self):
        resp = self._base(confidence=1.0)
        assert resp.confidence == 1.0

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            self._base(confidence=1.01)

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            self._base(confidence=-0.01)

    def test_base_model_used_optional(self):
        resp = self._base(model_used="model_c", base_model_used="model_a")
        assert resp.base_model_used == "model_a"

    def test_base_model_used_defaults_none(self):
        resp = self._base()
        assert resp.base_model_used is None


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------
class TestHealthResponse:
    def test_ok_status(self):
        h = HealthResponse(status="ok", models_loaded=True)
        assert h.status == "ok"
        assert h.models_loaded is True
        assert h.detail is None

    def test_degraded_status_with_detail(self):
        h = HealthResponse(status="degraded", models_loaded=False, detail="model_b load failed")
        assert h.status == "degraded"
        assert h.detail == "model_b load failed"

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="unknown", models_loaded=True)
