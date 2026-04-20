import types

import pytest

from api import firestore_service as fs


class DummySnapshot:
    def __init__(self, doc_id: str, data: dict | None):
        self.id = doc_id
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict:
        return dict(self._data) if self._data is not None else {}


class DummyDocumentRef:
    def __init__(self, collection: "DummyCollection", doc_id: str):
        self._collection = collection
        self._doc_id = doc_id

    def set(self, data: dict) -> None:
        self._collection.docs[self._doc_id] = dict(data)

    def get(self) -> DummySnapshot:
        return DummySnapshot(self._doc_id, self._collection.docs.get(self._doc_id))

    def update(self, data: dict) -> None:
        if self._doc_id not in self._collection.docs:
            raise KeyError(self._doc_id)
        self._collection.docs[self._doc_id].update(data)


class DummyCollection:
    def __init__(self):
        self.docs: dict[str, dict] = {}

    def document(self, doc_id: str) -> DummyDocumentRef:
        return DummyDocumentRef(self, doc_id)


class DummyClient:
    def __init__(self):
        self._collections: dict[str, DummyCollection] = {}

    def collection(self, name: str) -> DummyCollection:
        if name not in self._collections:
            self._collections[name] = DummyCollection()
        return self._collections[name]


def test_get_firestore_client_disabled(monkeypatch):
    monkeypatch.setenv("FIRESTORE_ENABLED", "false")
    client, error = fs.get_firestore_client()
    assert client is None
    assert error is not None
    assert "disabled" in error


def test_log_inference_and_maybe_enqueue_creates_queue_for_low_confidence(monkeypatch):
    monkeypatch.setenv("HITL_CONFIDENCE_THRESHOLD", "0.95")
    monkeypatch.setenv("HITL_RANDOM_SAMPLE_RATE", "0")
    monkeypatch.setenv("HITL_INCLUDE_ESCALATIONS", "true")
    monkeypatch.setattr(fs.random, "random", lambda: 1.0)
    monkeypatch.setattr(fs, "firestore", types.SimpleNamespace(SERVER_TIMESTAMP="ts"))

    client = DummyClient()
    result = fs.log_inference_and_maybe_enqueue(
        client=client,
        review_data={
            "review_body": "Great product",
            "review_title": "Good",
            "language": "en",
            "product_category": "electronics",
        },
        prediction={
            "predicted_stars": 4,
            "sentiment": "positive",
            "confidence": 0.4,
            "model_used": "model_a_escalated",
        },
        text_length=2,
        non_ascii_ratio=0.0,
        latency_ms=12.0,
    )

    assert result["logged"] is True
    assert result["queued_for_review"] is True
    assert "low_confidence" in result["review_reasons"]
    assert "escalated_path" in result["review_reasons"]

    inference_docs = client.collection(fs.COLLECTION_INFERENCE).docs
    queue_docs = client.collection(fs.COLLECTION_HUMAN_QUEUE).docs
    assert len(inference_docs) == 1
    assert len(queue_docs) == 1


def test_submit_human_label_resolves_queue_and_updates_inference(monkeypatch):
    monkeypatch.setattr(fs, "firestore", types.SimpleNamespace(SERVER_TIMESTAMP="ts"))
    client = DummyClient()

    queue_id = "queue_1"
    inference_id = "inf_1"

    client.collection(fs.COLLECTION_HUMAN_QUEUE).document(queue_id).set(
        {
            "inference_id": inference_id,
            "status": "pending",
            "reasons": ["low_confidence"],
        }
    )
    client.collection(fs.COLLECTION_INFERENCE).document(inference_id).set(
        {
            "predicted_stars": 2,
            "confidence": 0.42,
        }
    )

    result = fs.submit_human_label(
        client=client,
        queue_id=queue_id,
        human_stars=4,
        reviewer_id="reviewer_demo",
        notes="looks closer to 4",
    )

    assert result["status"] == "resolved"
    assert result["queue_id"] == queue_id
    assert result["inference_id"] == inference_id
    assert result["human_stars"] == 4

    queue_doc = client.collection(fs.COLLECTION_HUMAN_QUEUE).docs[queue_id]
    assert queue_doc["status"] == "resolved"
    assert queue_doc["resolved_by"] == "reviewer_demo"

    label_doc = client.collection(fs.COLLECTION_HUMAN_LABELS).docs[inference_id]
    assert label_doc["human_stars"] == 4
    assert label_doc["reviewer_id"] == "reviewer_demo"

    inference_doc = client.collection(fs.COLLECTION_INFERENCE).docs[inference_id]
    assert inference_doc["human_stars"] == 4
    assert inference_doc["human_reviewer_id"] == "reviewer_demo"


def test_submit_human_label_rejects_invalid_rating(monkeypatch):
    monkeypatch.setattr(fs, "firestore", types.SimpleNamespace(SERVER_TIMESTAMP="ts"))
    client = DummyClient()

    with pytest.raises(ValueError, match="human_stars"):
        fs.submit_human_label(
            client=client,
            queue_id="missing",
            human_stars=6,
            reviewer_id="reviewer_demo",
        )


def test_run_drift_detection_insufficient_data(monkeypatch):
    monkeypatch.setattr(fs, "firestore", types.SimpleNamespace(SERVER_TIMESTAMP="ts"))
    monkeypatch.setattr(fs, "_fetch_inference_logs", lambda *_args, **_kwargs: [])

    summary = fs.run_drift_detection(
        client=object(),
        lookback_hours=24,
        baseline_days=30,
        min_samples=10,
    )

    assert summary["status"] == "insufficient_data"
    assert summary["baseline_count"] == 0
    assert summary["current_count"] == 0
