import os
import uuid

import pytest

import firebase_admin
from api import firestore_service as fs


@pytest.fixture(scope="module")
def emulator_client():
    if os.getenv("FIRESTORE_E2E_RUN") != "1":
        pytest.skip("Set FIRESTORE_E2E_RUN=1 to run Firestore emulator e2e tests.")

    if not os.getenv("FIRESTORE_EMULATOR_HOST"):
        pytest.skip("FIRESTORE_EMULATOR_HOST is not set.")

    os.environ["FIRESTORE_ENABLED"] = "true"
    os.environ.setdefault("FIREBASE_PROJECT_ID", "demo-firestore-e2e")
    os.environ.pop("FIREBASE_CREDENTIALS_PATH", None)
    os.environ.pop("FIREBASE_CREDENTIALS_JSON", None)
    os.environ["FIREBASE_REQUIRE_LOCAL_CREDENTIALS"] = "false"

    try:
        firebase_admin.delete_app(firebase_admin.get_app())
    except Exception:
        pass

    client, error = fs.get_firestore_client()
    if client is None:
        pytest.skip(f"Could not connect to Firestore emulator: {error}")

    return client


@pytest.fixture
def isolated_collections(emulator_client):
    suffix = uuid.uuid4().hex[:10]
    original_names = (
        fs.COLLECTION_INFERENCE,
        fs.COLLECTION_HUMAN_QUEUE,
        fs.COLLECTION_HUMAN_LABELS,
        fs.COLLECTION_DRIFT_METRICS,
    )

    fs.COLLECTION_INFERENCE = f"e2e_inference_{suffix}"
    fs.COLLECTION_HUMAN_QUEUE = f"e2e_human_queue_{suffix}"
    fs.COLLECTION_HUMAN_LABELS = f"e2e_human_labels_{suffix}"
    fs.COLLECTION_DRIFT_METRICS = f"e2e_drift_metrics_{suffix}"

    yield emulator_client

    for collection_name in [
        fs.COLLECTION_INFERENCE,
        fs.COLLECTION_HUMAN_QUEUE,
        fs.COLLECTION_HUMAN_LABELS,
        fs.COLLECTION_DRIFT_METRICS,
    ]:
        for doc in emulator_client.collection(collection_name).stream():
            doc.reference.delete()

    (
        fs.COLLECTION_INFERENCE,
        fs.COLLECTION_HUMAN_QUEUE,
        fs.COLLECTION_HUMAN_LABELS,
        fs.COLLECTION_DRIFT_METRICS,
    ) = original_names


def test_emulator_e2e_log_queue_and_label_roundtrip(isolated_collections, monkeypatch):
    client = isolated_collections

    monkeypatch.setenv("HITL_CONFIDENCE_THRESHOLD", "0.99")
    monkeypatch.setenv("HITL_RANDOM_SAMPLE_RATE", "0")
    monkeypatch.setenv("HITL_INCLUDE_ESCALATIONS", "true")

    result = fs.log_inference_and_maybe_enqueue(
        client=client,
        review_data={
            "review_body": "Great quality with minor issues",
            "review_title": "Good overall",
            "language": "en",
            "product_category": "electronics",
        },
        prediction={
            "predicted_stars": 3,
            "sentiment": "neutral",
            "confidence": 0.30,
            "model_used": "model_b_escalated",
        },
        text_length=5,
        non_ascii_ratio=0.0,
        latency_ms=8.5,
    )

    assert result["logged"] is True
    assert result["queued_for_review"] is True
    assert result["inference_id"] is not None

    queue_items = fs.list_human_review_queue(client=client, status="pending", limit=10)
    assert queue_items

    queue_id = queue_items[0]["id"]
    labeled = fs.submit_human_label(
        client=client,
        queue_id=queue_id,
        human_stars=4,
        reviewer_id="emulator_reviewer",
        notes="e2e check",
    )

    assert labeled["status"] == "resolved"
    assert labeled["human_stars"] == 4

    inference_snapshot = (
        client.collection(fs.COLLECTION_INFERENCE)
        .document(result["inference_id"])
        .get()
    )
    assert inference_snapshot.exists
    inference_doc = inference_snapshot.to_dict()
    assert inference_doc.get("human_stars") == 4


def test_emulator_e2e_drift_insufficient_data(isolated_collections):
    summary = fs.run_drift_detection(
        client=isolated_collections,
        lookback_hours=24,
        baseline_days=30,
        min_samples=9999,
    )

    assert summary["status"] == "insufficient_data"
