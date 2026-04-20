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
from time import perf_counter
from typing import Any

import joblib
import torch
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure project root is on the path so `router` can be imported when the
# module is launched from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.engine import compute_text_signals, preprocess_incoming_review, run_inference  # noqa: E402

from api.firestore_service import (  # noqa: E402
    get_firestore_client,
    get_latest_drift_metrics,
    list_human_review_queue,
    log_inference_and_maybe_enqueue,
    run_drift_detection,
    submit_human_label,
)

from api.schemas import (  # noqa: E402
    DriftMetricResponse,
    DriftRunResponse,
    ErrorResponse,
    HealthResponse,
    HumanLabelRequest,
    HumanLabelResponse,
    HumanReviewQueueItem,
    PredictionResponse,
    ReviewRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model registry populated at startup
# ---------------------------------------------------------------------------
MODELS: dict[str, Any] = {}
FIRESTORE_STATE: dict[str, Any] = {
    "client": None,
    "connected": False,
    "error": None,
}

MODEL_A_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved", "model_a.pkl")
MODEL_A_LANG_PATHS = {
    "de": os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved", "model_a_de.pkl"),
    "es": os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved", "model_a_es.pkl"),
    "fr": os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved", "model_a_fr.pkl"),
}
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

        MODELS["model_a_by_language"] = {"en": MODELS["model_a"]}
        for language, model_path in MODEL_A_LANG_PATHS.items():
            if not os.path.exists(model_path):
                continue
            try:
                MODELS["model_a_by_language"][language] = joblib.load(model_path)
                log.info("Model A (%s) loaded.", language)
            except Exception as exc:
                log.warning("Model A (%s) failed to load from %s: %s", language, model_path, exc)

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

    firestore_client, firestore_error = get_firestore_client()
    if firestore_client is not None:
        FIRESTORE_STATE["client"] = firestore_client
        FIRESTORE_STATE["connected"] = True
        FIRESTORE_STATE["error"] = None
        log.info("Firestore connected.")
    else:
        FIRESTORE_STATE["client"] = None
        FIRESTORE_STATE["connected"] = False
        FIRESTORE_STATE["error"] = firestore_error
        if firestore_error:
            log.warning("Firestore unavailable: %s", firestore_error)

    yield

    MODELS.clear()
    FIRESTORE_STATE["client"] = None
    FIRESTORE_STATE["connected"] = False
    FIRESTORE_STATE["error"] = None
    log.info("Models unloaded.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ReviewRoute API",
    description=(
        "REST API for the ReviewRoute multi-model star-rating predictor.\n\n"
        "The service routes each review to one of three models:\n"
        "- **Model A** – TF-IDF + Logistic Regression (fast, language-specific variants)\n"
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
    firestore_connected = bool(FIRESTORE_STATE.get("connected"))
    firestore_enabled = os.getenv("FIRESTORE_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }

    detail = MODELS.get("load_error") if not loaded else None
    if loaded and firestore_enabled and not firestore_connected:
        detail = FIRESTORE_STATE.get("error") or "Firestore is not connected."

    return HealthResponse(
        status="ok" if loaded else "degraded",
        models_loaded=loaded,
        detail=detail,
        firestore_connected=firestore_connected,
    )


def _require_firestore_client() -> Any:
    client = FIRESTORE_STATE.get("client")
    if client is None or not FIRESTORE_STATE.get("connected"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firestore is not configured or unavailable.",
        )
    return client


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

    start_time = perf_counter()

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

    latency_ms = (perf_counter() - start_time) * 1000.0

    result.setdefault("inference_id", None)
    result.setdefault("queued_for_review", False)
    result.setdefault("review_reasons", [])
    result.setdefault("resolved_language", review.language or "en")
    result.setdefault("language_was_detected", review.language is None)

    firestore_client = FIRESTORE_STATE.get("client")
    if firestore_client is not None and FIRESTORE_STATE.get("connected"):
        try:
            prepared = preprocess_incoming_review(review.review_body, review.review_title)
            signals = compute_text_signals(prepared["review_body"])
            review_data_for_storage = review.model_dump()
            review_data_for_storage["language"] = result.get("resolved_language")

            persist_result = log_inference_and_maybe_enqueue(
                client=firestore_client,
                review_data=review_data_for_storage,
                prediction=result,
                text_length=prepared["text_length"],
                non_ascii_ratio=float(signals["non_ascii_ratio"]),
                latency_ms=latency_ms,
            )
            result["inference_id"] = persist_result.get("inference_id")
            result["queued_for_review"] = bool(persist_result.get("queued_for_review", False))
            result["review_reasons"] = persist_result.get("review_reasons", [])
        except Exception as exc:
            log.warning("Failed to persist inference in Firestore: %s", exc)

    return PredictionResponse(**result)


@app.get(
    "/human-review/queue",
    response_model=list[HumanReviewQueueItem],
    tags=["Human Review"],
    summary="List queued reviews for human verification",
)
async def human_review_queue(
    status_filter: str = Query("pending", alias="status"),
    limit: int = Query(50, ge=1, le=200),
) -> list[HumanReviewQueueItem]:
    client = _require_firestore_client()
    try:
        rows = list_human_review_queue(client=client, status=status_filter, limit=limit)
    except Exception as exc:
        log.exception("Failed to list human review queue")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list review queue: {exc}",
        )

    return [HumanReviewQueueItem(**row) for row in rows]


@app.post(
    "/human-review/{queue_id}/label",
    response_model=HumanLabelResponse,
    tags=["Human Review"],
    summary="Submit a human label for a queued review",
)
async def label_human_review(queue_id: str, payload: HumanLabelRequest) -> HumanLabelResponse:
    client = _require_firestore_client()
    try:
        row = submit_human_label(
            client=client,
            queue_id=queue_id,
            human_stars=payload.human_stars,
            reviewer_id=payload.reviewer_id,
            notes=payload.notes,
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        log.exception("Failed to submit human label")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit human label: {exc}",
        )

    return HumanLabelResponse(**row)


@app.post(
    "/drift/run",
    response_model=DriftRunResponse,
    tags=["Drift"],
    summary="Run drift detection against Firestore inference logs",
)
async def drift_run(
    lookback_hours: int = Query(24, ge=1, le=168),
    baseline_days: int = Query(30, ge=1, le=365),
    min_samples: int = Query(200, ge=20, le=100000),
) -> DriftRunResponse:
    client = _require_firestore_client()
    try:
        summary = run_drift_detection(
            client=client,
            lookback_hours=lookback_hours,
            baseline_days=baseline_days,
            min_samples=min_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        log.exception("Failed to run drift detection")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {exc}",
        )

    return DriftRunResponse(**summary)


@app.get(
    "/drift/latest",
    response_model=list[DriftMetricResponse],
    tags=["Drift"],
    summary="Get latest persisted drift metrics",
)
async def drift_latest(limit: int = Query(20, ge=1, le=200)) -> list[DriftMetricResponse]:
    client = _require_firestore_client()
    try:
        rows = get_latest_drift_metrics(client=client, limit=limit)
    except Exception as exc:
        log.exception("Failed to read latest drift metrics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read drift metrics: {exc}",
        )

    return [DriftMetricResponse(**row) for row in rows]
