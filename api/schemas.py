from datetime import datetime
from typing import Any, Optional, Literal
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
    language: Optional[str] = Field(
        None,
        min_length=2,
        max_length=10,
        description="Optional ISO 639-1 language code (e.g. 'en', 'fr', 'de'). If omitted, API auto-detects language.",
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

    @field_validator("language", mode="before")
    @classmethod
    def language_normalise(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = str(v).lower().strip()
        return cleaned or None

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
    inference_id: Optional[str] = Field(
        None,
        description="Firestore inference record ID when persistence is enabled.",
    )
    queued_for_review: bool = Field(
        False,
        description="True when this prediction was queued for human review.",
    )
    review_reasons: list[str] = Field(
        default_factory=list,
        description="Reasons this item was added to human review queue.",
    )
    resolved_language: Optional[str] = Field(
        None,
        description="Final language used by routing/inference (provided or auto-detected).",
    )
    language_was_detected: bool = Field(
        False,
        description="True when language was auto-detected because request language was omitted.",
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
    firestore_connected: Optional[bool] = Field(
        None,
        description="True when Firestore client is available.",
    )


class HumanReviewQueueItem(BaseModel):
    """A queue item requiring human label verification."""

    id: str
    inference_id: str
    reasons: list[str] = Field(default_factory=list)
    priority: int = Field(3, ge=1, le=5)
    status: str
    assigned_to: Optional[str] = None
    created_at: Optional[datetime] = None
    inference: Optional[dict[str, Any]] = None


class HumanLabelRequest(BaseModel):
    """Payload submitted by reviewer for a queued prediction."""

    human_stars: int = Field(..., ge=1, le=5)
    reviewer_id: str = Field(..., min_length=1, max_length=100)
    notes: Optional[str] = Field(None, max_length=2000)


class HumanLabelResponse(BaseModel):
    """Result of a successful human labeling action."""

    queue_id: str
    inference_id: str
    status: str
    human_stars: int = Field(..., ge=1, le=5)


class DriftMetricResponse(BaseModel):
    """Single drift metric persisted in Firestore."""

    id: Optional[str] = None
    metric_name: str
    metric_value: float
    warn_threshold: float
    threshold: float
    status: Literal["ok", "warn", "alert"]
    baseline_count: int
    current_count: int
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    created_at: Optional[datetime] = None


class DriftRunResponse(BaseModel):
    """Summary from drift detection execution."""

    status: Literal["ok", "warn", "alert", "insufficient_data"]
    baseline_count: int
    current_count: int
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    metrics: list[DriftMetricResponse] = Field(default_factory=list)
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
