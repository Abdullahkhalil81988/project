from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ReviewRequest(BaseModel):
    """Input schema for a single review prediction request."""

    review_body: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The main body text of the review.",
        examples=["This product is absolutely amazing, works perfectly out of the box!"],
    )
    review_title: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional title of the review. If omitted, the first 5 words of the body are used.",
        examples=["Great product"],
    )
    language: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description="ISO 639-1 language code of the review (e.g. 'en', 'fr', 'de').",
        examples=["en"],
    )
    product_category: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Product category slug (e.g. 'electronics', 'book', 'apparel').",
        examples=["electronics"],
    )

    @field_validator("review_body", mode="before")
    @classmethod
    def body_not_whitespace(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise ValueError("review_body cannot be empty or whitespace only.")
        return v

    @field_validator("review_title")
    @classmethod
    def title_strip(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            stripped = v.strip()
            return stripped if stripped else None
        return v

    @field_validator("language")
    @classmethod
    def language_normalise(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("product_category")
    @classmethod
    def category_normalise(cls, v: str) -> str:
        return v.lower().strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "review_body": "Exceptional build quality and battery life. Highly recommended.",
                    "review_title": "Best laptop I've owned",
                    "language": "en",
                    "product_category": "electronics",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema returned by the /predict endpoint."""

    predicted_stars: int = Field(..., ge=1, le=5, description="Predicted star rating (1–5).")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        ..., description="Derived sentiment bucket."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score (0–1).")
    model_used: str = Field(..., description="Which model handled this request.")
    base_model_used: Optional[str] = Field(
        None, description="For Model C: which base model generated the feature vector."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_stars": 5,
                    "sentiment": "positive",
                    "confidence": 0.87,
                    "model_used": "model_a",
                    "base_model_used": None,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: Literal["ok", "degraded"] = Field(..., description="'ok' if all models loaded.")
    models_loaded: bool = Field(..., description="True when all models are in memory.")
    detail: Optional[str] = Field(None, description="Error detail when models failed to load.")


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
