from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from api import main


@pytest.fixture(autouse=True)
def reset_app_state(monkeypatch):
    original_models = dict(main.MODELS)
    original_firestore_state = dict(main.FIRESTORE_STATE)
    original_lifespan_context = main.app.router.lifespan_context

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    # Keep tests fast and deterministic by bypassing heavy model startup.
    main.app.router.lifespan_context = noop_lifespan
    main.MODELS.clear()
    main.MODELS["loaded"] = True
    main.FIRESTORE_STATE.clear()
    main.FIRESTORE_STATE.update(
        {
            "client": object(),
            "connected": True,
            "error": None,
        }
    )

    yield

    main.app.router.lifespan_context = original_lifespan_context
    main.MODELS.clear()
    main.MODELS.update(original_models)
    main.FIRESTORE_STATE.clear()
    main.FIRESTORE_STATE.update(original_firestore_state)


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_endpoint_returns_expected_shape(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["models_loaded"] is True
    assert payload["firestore_connected"] is True


def test_predict_endpoint_returns_enriched_firestore_fields(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "run_inference",
        lambda **_kwargs: {
            "predicted_stars": 5,
            "sentiment": "positive",
            "confidence": 0.91,
            "model_used": "model_a",
        },
    )
    monkeypatch.setattr(
        main,
        "preprocess_incoming_review",
        lambda review_body, review_title=None: {
            "review_body": review_body,
            "review_title": review_title,
            "model_text": review_body,
            "text_length": 4,
        },
    )
    monkeypatch.setattr(main, "compute_text_signals", lambda _text: {"non_ascii_ratio": 0.0})
    monkeypatch.setattr(
        main,
        "log_inference_and_maybe_enqueue",
        lambda **_kwargs: {
            "inference_id": "inf_http_1",
            "queued_for_review": True,
            "review_reasons": ["low_confidence"],
        },
    )

    response = client.post(
        "/predict",
        json={
            "review_body": "Great value and quality overall",
            "review_title": "Impressive",
            "language": "en",
            "product_category": "electronics",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_stars"] == 5
    assert payload["model_used"] == "model_a"
    assert payload["inference_id"] == "inf_http_1"
    assert payload["queued_for_review"] is True
    assert payload["review_reasons"] == ["low_confidence"]


def test_predict_endpoint_detects_language_when_missing(client, monkeypatch):
    def _fake_run_inference(**kwargs):
        assert kwargs["language"] is None
        return {
            "predicted_stars": 5,
            "sentiment": "positive",
            "confidence": 0.88,
            "model_used": "model_a_fr",
            "resolved_language": "fr",
            "language_was_detected": True,
        }

    monkeypatch.setattr(main, "run_inference", _fake_run_inference)
    monkeypatch.setattr(
        main,
        "preprocess_incoming_review",
        lambda review_body, review_title=None: {
            "review_body": review_body,
            "review_title": review_title,
            "model_text": review_body,
            "text_length": 4,
        },
    )
    monkeypatch.setattr(main, "compute_text_signals", lambda _text: {"non_ascii_ratio": 0.0})
    monkeypatch.setattr(
        main,
        "log_inference_and_maybe_enqueue",
        lambda **_kwargs: {
            "inference_id": "inf_http_detect_1",
            "queued_for_review": False,
            "review_reasons": [],
        },
    )

    response = client.post(
        "/predict",
        json={
            "review_body": "Ce produit est tres bon et je suis tres satisfait.",
            "review_title": "Excellent",
            "product_category": "apparel",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resolved_language"] == "fr"
    assert payload["language_was_detected"] is True


def test_predict_endpoint_returns_503_when_models_unloaded(client):
    main.MODELS["loaded"] = False

    response = client.post(
        "/predict",
        json={
            "review_body": "Good product",
            "review_title": "Good",
            "language": "en",
            "product_category": "electronics",
        },
    )

    assert response.status_code == 503


def test_human_review_queue_endpoint_returns_items(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "list_human_review_queue",
        lambda **_kwargs: [
            {
                "id": "queue_http_1",
                "inference_id": "inf_1",
                "reasons": ["low_confidence"],
                "priority": 1,
                "status": "pending",
                "assigned_to": None,
            }
        ],
    )

    response = client.get("/human-review/queue", params={"status": "pending", "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["id"] == "queue_http_1"


def test_human_review_queue_endpoint_requires_firestore(client):
    main.FIRESTORE_STATE["client"] = None
    main.FIRESTORE_STATE["connected"] = False

    response = client.get("/human-review/queue")

    assert response.status_code == 503


def test_human_review_label_endpoint_returns_response(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "submit_human_label",
        lambda **_kwargs: {
            "queue_id": "queue_http_1",
            "inference_id": "inf_1",
            "status": "resolved",
            "human_stars": 4,
        },
    )

    response = client.post(
        "/human-review/queue_http_1/label",
        json={
            "human_stars": 4,
            "reviewer_id": "reviewer_http",
            "notes": "looks right",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "resolved"
    assert payload["human_stars"] == 4


def test_drift_endpoints_return_expected_payloads(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "run_drift_detection",
        lambda **_kwargs: {
            "status": "warn",
            "baseline_count": 42,
            "current_count": 21,
            "window_start": None,
            "window_end": None,
            "message": "Drift metrics computed and stored.",
            "metrics": [
                {
                    "id": "dr1",
                    "metric_name": "confidence_psi",
                    "metric_value": 0.21,
                    "warn_threshold": 0.2,
                    "threshold": 0.3,
                    "status": "warn",
                    "baseline_count": 42,
                    "current_count": 21,
                    "window_start": None,
                    "window_end": None,
                    "created_at": None,
                }
            ],
        },
    )
    monkeypatch.setattr(
        main,
        "get_latest_drift_metrics",
        lambda **_kwargs: [
            {
                "id": "dr2",
                "metric_name": "route_mix_jsd",
                "metric_value": 0.05,
                "warn_threshold": 0.1,
                "threshold": 0.2,
                "status": "ok",
                "baseline_count": 42,
                "current_count": 21,
                "window_start": None,
                "window_end": None,
                "created_at": None,
            }
        ],
    )

    run_response = client.post(
        "/drift/run",
        params={"lookback_hours": 24, "baseline_days": 30, "min_samples": 20},
    )
    latest_response = client.get("/drift/latest", params={"limit": 5})

    assert run_response.status_code == 200
    assert latest_response.status_code == 200

    run_payload = run_response.json()
    latest_payload = latest_response.json()

    assert run_payload["status"] == "warn"
    assert run_payload["current_count"] == 21
    assert len(run_payload["metrics"]) == 1

    assert len(latest_payload) == 1
    assert latest_payload[0]["metric_name"] == "route_mix_jsd"
