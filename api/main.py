"""
ReviewRoute API
---------------
FastAPI service that wraps the three-model inference engine.

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
    http://localhost:8000/openapi.json
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import joblib
import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure project root is on the path so `router` can be imported when the
# module is launched from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.engine import run_inference  # noqa: E402

from api.schemas import ErrorResponse, HealthResponse, PredictionResponse, ReviewRequest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model registry populated at startup
# ---------------------------------------------------------------------------
MODELS: dict[str, Any] = {}

MODEL_A_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved", "model_a.pkl")
MODEL_B_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved", "model_b")
MODEL_C_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved", "model_c.pkl")
MODEL_C_CAT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved", "model_c_categories.pkl")


# ---------------------------------------------------------------------------
# Lifespan: load / unload models
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading models…")
    try:
        MODELS["model_a"] = joblib.load(MODEL_A_PATH)
        log.info("Model A loaded.")

        MODELS["model_c"] = joblib.load(MODEL_C_PATH)
        MODELS["model_c_categories"] = joblib.load(MODEL_C_CAT_PATH)
        log.info("Model C loaded.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B_PATH)
        model_b = AutoModelForSequenceClassification.from_pretrained(MODEL_B_PATH).to(device)
        model_b.eval()
        MODELS["model_b_tokenizer"] = tokenizer_b
        MODELS["model_b"] = model_b
        log.info("Model B loaded on %s.", device)

        MODELS["loaded"] = True
        log.info("All models ready.")
    except Exception as exc:
        MODELS["loaded"] = False
        MODELS["load_error"] = str(exc)
        log.error("Model loading failed: %s", exc)

    yield

    MODELS.clear()
    log.info("Models unloaded.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ReviewRoute API",
    description=(
        "REST API for the ReviewRoute multi-model star-rating predictor.\n\n"
        "The service routes each review to one of three models:\n"
        "- **Model A** – TF-IDF + Logistic Regression (fast, English-only)\n"
        "- **Model B** – XLM-RoBERTa transformer (multilingual)\n"
        "- **Model C** – Stacking ensemble (selected product categories)\n\n"
        "Confidence-based escalation: if Model A confidence < 0.55, Model B is used automatically."
    ),
    version="1.0.0",
    contact={"name": "ReviewRoute"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a structured 422 with field-level error details."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service health check",
    description="Returns whether the API is running and all models are loaded.",
)
async def health() -> HealthResponse:
    loaded = MODELS.get("loaded", False)
    return HealthResponse(
        status="ok" if loaded else "degraded",
        models_loaded=loaded,
        detail=MODELS.get("load_error") if not loaded else None,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict star rating for a review",
    description=(
        "Submit a review and receive a 1–5 star prediction, a sentiment label, "
        "a confidence score, and information about which model was used."
    ),
    responses={
        200: {"description": "Successful prediction", "model": PredictionResponse},
        422: {"description": "Validation error – check request fields", "model": ErrorResponse},
        503: {"description": "Models not yet loaded", "model": ErrorResponse},
        500: {"description": "Inference error", "model": ErrorResponse},
    },
)
async def predict(review: ReviewRequest) -> PredictionResponse:
    if not MODELS.get("loaded"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded. The service may still be starting up.",
        )

    try:
        result = run_inference(
            review_body=review.review_body,
            language=review.language,
            product_category=review.product_category,
            models=MODELS,
            review_title=review.review_title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        log.exception("Inference failed for request: %s", review.model_dump())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed. Please try again.",
        )

    return PredictionResponse(**result)
