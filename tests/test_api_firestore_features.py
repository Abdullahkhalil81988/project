import asyncio

import pytest
from fastapi import HTTPException, status

from api import main
from api.schemas import HumanLabelRequest, ReviewRequest


@pytest.fixture(autouse=True)
def restore_global_state(monkeypatch):
    original_models = dict(main.MODELS)
    original_firestore_state = dict(main.FIRESTORE_STATE)
    monkeypatch.delenv("FIRESTORE_ENABLED", raising=False)

    yield

    main.MODELS.clear()
    main.MODELS.update(original_models)
    main.FIRESTORE_STATE.clear()
    main.FIRESTORE_STATE.update(original_firestore_state)


def _run(coro):
    return asyncio.run(coro)


def test_health_reports_firestore_error_when_models_loaded_but_firestore_disconnected(monkeypatch):
    monkeypatch.setenv("FIRESTORE_ENABLED", "true")
    main.MODELS.clear()
    main.MODELS["loaded"] = True
    main.FIRESTORE_STATE.update(
        {
            "client": None,
            "connected": False,
            "error": "credentials missing",
        }
    )

    resp = _run(main.health())

    assert resp.status == "ok"
    assert resp.models_loaded is True
    assert resp.firestore_connected is False
    assert resp.detail == "credentials missing"


def test_require_firestore_client_raises_when_disconnected():
    main.FIRESTORE_STATE.update(
        {
            "client": None,
            "connected": False,
            "error": "down",
        }
    )

    with pytest.raises(HTTPException) as exc:
        main._require_firestore_client()

    assert exc.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_predict_returns_503_when_models_not_loaded():
    main.MODELS.clear()
    main.MODELS["loaded"] = False

    review = ReviewRequest(
        review_body="great product",
        review_title=None,
        language="en",
        product_category="electronics",
    )

    with pytest.raises(HTTPException) as exc:
        _run(main.predict(review))

    assert exc.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_predict_includes_firestore_metadata_when_persistence_succeeds(monkeypatch):
    main.MODELS.clear()
    main.MODELS["loaded"] = True
    main.FIRESTORE_STATE.update(
        {
            "client": object(),
            "connected": True,
            "error": None,
        }
    )

    monkeypatch.setattr(
        main,
        "run_inference",
        lambda **_kwargs: {
            "predicted_stars": 5,
            "sentiment": "positive",
            "confidence": 0.9,
            "model_used": "model_a",
        },
    )
    monkeypatch.setattr(
        main,
        "preprocess_incoming_review",
        lambda review_body, review_title=None: {
            "review_body": review_body,
            "text_length": 3,
            "review_title": review_title,
            "model_text": review_body,
        },
    )
    monkeypatch.setattr(main, "compute_text_signals", lambda _text: {"non_ascii_ratio": 0.0})
    monkeypatch.setattr(
        main,
        "log_inference_and_maybe_enqueue",
        lambda **_kwargs: {
            "inference_id": "inf_test_1",
            "queued_for_review": True,
            "review_reasons": ["low_confidence"],
        },
    )

    review = ReviewRequest(
        review_body="great product overall",
        review_title=None,
        language="en",
        product_category="electronics",
    )
    resp = _run(main.predict(review))

    assert resp.predicted_stars == 5
    assert resp.model_used == "model_a"
    assert resp.inference_id == "inf_test_1"
    assert resp.queued_for_review is True
    assert resp.review_reasons == ["low_confidence"]


def test_human_review_queue_returns_items(monkeypatch):
    main.FIRESTORE_STATE.update(
        {
            "client": object(),
            "connected": True,
            "error": None,
        }
    )
    monkeypatch.setattr(
        main,
        "list_human_review_queue",
        lambda **_kwargs: [
            {
                "id": "queue_1",
                "inference_id": "inf_1",
                "reasons": ["low_confidence"],
                "priority": 1,
                "status": "pending",
                "assigned_to": None,
            }
        ],
    )

    items = _run(main.human_review_queue(status_filter="pending", limit=10))

    assert len(items) == 1
    assert items[0].id == "queue_1"
    assert items[0].inference_id == "inf_1"


def test_label_human_review_translates_lookup_error_to_404(monkeypatch):
    main.FIRESTORE_STATE.update(
        {
            "client": object(),
            "connected": True,
            "error": None,
        }
    )

    def _raise_lookup(**_kwargs):
        raise LookupError("Queue item not found")

    monkeypatch.setattr(main, "submit_human_label", _raise_lookup)

    payload = HumanLabelRequest(human_stars=4, reviewer_id="rev_1", notes="checked")
    with pytest.raises(HTTPException) as exc:
        _run(main.label_human_review("missing", payload))

    assert exc.value.status_code == status.HTTP_404_NOT_FOUND


def test_drift_endpoints_return_expected_shapes(monkeypatch):
    main.FIRESTORE_STATE.update(
        {
            "client": object(),
            "connected": True,
            "error": None,
        }
    )

    monkeypatch.setattr(
        main,
        "run_drift_detection",
        lambda **_kwargs: {
            "status": "warn",
            "baseline_count": 50,
            "current_count": 25,
            "window_start": None,
            "window_end": None,
            "message": "Drift metrics computed and stored.",
            "metrics": [
                {
                    "id": "m1",
                    "metric_name": "confidence_psi",
                    "metric_value": 0.22,
                    "warn_threshold": 0.2,
                    "threshold": 0.3,
                    "status": "warn",
                    "baseline_count": 50,
                    "current_count": 25,
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
                "id": "m2",
                "metric_name": "route_mix_jsd",
                "metric_value": 0.08,
                "warn_threshold": 0.1,
                "threshold": 0.2,
                "status": "ok",
                "baseline_count": 50,
                "current_count": 25,
                "window_start": None,
                "window_end": None,
                "created_at": None,
            }
        ],
    )

    run_resp = _run(main.drift_run(lookback_hours=24, baseline_days=30, min_samples=20))
    latest_resp = _run(main.drift_latest(limit=5))

    assert run_resp.status == "warn"
    assert run_resp.current_count == 25
    assert len(run_resp.metrics) == 1

    assert len(latest_resp) == 1
    assert latest_resp[0].metric_name == "route_mix_jsd"
