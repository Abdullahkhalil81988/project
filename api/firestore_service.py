from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud.firestore_v1 import SERVER_TIMESTAMP as GOOGLE_SERVER_TIMESTAMP
except Exception:  # pragma: no cover - import availability differs per environment
    firebase_admin = None
    credentials = None
    firestore = None
    GOOGLE_SERVER_TIMESTAMP = None


log = logging.getLogger(__name__)

COLLECTION_INFERENCE = os.getenv("FIRESTORE_INFERENCE_COLLECTION", "inference_log")
COLLECTION_HUMAN_QUEUE = os.getenv("FIRESTORE_HUMAN_QUEUE_COLLECTION", "human_review_queue")
COLLECTION_HUMAN_LABELS = os.getenv("FIRESTORE_HUMAN_LABELS_COLLECTION", "human_labels")
COLLECTION_DRIFT_METRICS = os.getenv("FIRESTORE_DRIFT_COLLECTION", "drift_metrics")


def _server_timestamp() -> Any:
    # Keep runtime compatibility with firebase_admin while avoiding static
    # type-checker false positives for firestore.SERVER_TIMESTAMP.
    if firestore is not None:
        runtime_timestamp = getattr(firestore, "SERVER_TIMESTAMP", None)
        if runtime_timestamp is not None:
            return runtime_timestamp
    return GOOGLE_SERVER_TIMESTAMP


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _hash_review(review_body: str, review_title: Optional[str]) -> str:
    source = f"{review_title or ''}\n{review_body}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def get_firestore_client() -> tuple[Optional[Any], Optional[str]]:
    """
    Create and return a Firestore client.

    Uses one of the following authentication methods:
    1. FIREBASE_CREDENTIALS_PATH -> path to service account JSON
    2. FIREBASE_CREDENTIALS_JSON -> service account JSON string
    3. Project-local firebase-service-account.json
    4. Application Default Credentials
    """
    if not _env_bool("FIRESTORE_ENABLED", True):
        return None, "Firestore disabled via FIRESTORE_ENABLED=false"

    if firebase_admin is None or credentials is None or firestore is None:
        return None, "firebase-admin is not installed"

    try:
        try:
            firebase_admin.get_app()
        except ValueError:
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
            cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
            project_id = os.getenv("FIREBASE_PROJECT_ID")

            project_root = Path(__file__).resolve().parents[1]
            if cred_path:
                path_obj = Path(cred_path)
                if not path_obj.is_absolute():
                    cred_path = str((project_root / path_obj).resolve())

            if not cred_path and not cred_json:
                # Default local credential file in project root.
                default_path = (project_root / "firebase-service-account.json").resolve()
                cred_path = str(default_path)
                log.info("Using project Firebase credentials file: %s", cred_path)

            init_options: dict[str, Any] = {}
            if project_id:
                init_options["projectId"] = project_id

            if cred_path:
                cred = credentials.Certificate(cred_path)
            elif cred_json:
                cred_data = json.loads(cred_json)
                cred = credentials.Certificate(cred_data)
            else:
                cred = credentials.ApplicationDefault()

            if init_options:
                firebase_admin.initialize_app(cred, init_options)
            else:
                firebase_admin.initialize_app(cred)

        return firestore.client(), None
    except Exception as exc:  # pragma: no cover - depends on local credential setup
        return None, str(exc)


def log_inference_and_maybe_enqueue(
    client: Any,
    review_data: dict[str, Any],
    prediction: dict[str, Any],
    text_length: int,
    non_ascii_ratio: float,
    latency_ms: Optional[float] = None,
) -> dict[str, Any]:
    """Persist inference metadata and queue uncertain cases for human review."""
    if client is None or firestore is None:
        return {
            "logged": False,
            "inference_id": None,
            "queued_for_review": False,
            "review_reasons": [],
        }

    confidence_threshold = _env_float("HITL_CONFIDENCE_THRESHOLD", 0.60)
    random_sample_rate = min(max(_env_float("HITL_RANDOM_SAMPLE_RATE", 0.02), 0.0), 1.0)
    include_escalations = _env_bool("HITL_INCLUDE_ESCALATIONS", True)

    review_body = str(review_data.get("review_body", ""))
    review_title = review_data.get("review_title")

    inference_id = str(uuid.uuid4())
    model_used = str(prediction.get("model_used", "unknown"))
    confidence = float(prediction.get("confidence", 0.0))

    inference_doc = {
        "created_at": _server_timestamp(),
        "language": str(review_data.get("language", "")),
        "product_category": str(review_data.get("product_category", "")),
        "text_length": int(text_length),
        "non_ascii_ratio": float(non_ascii_ratio),
        "model_used": model_used,
        "base_model_used": prediction.get("base_model_used"),
        "predicted_stars": int(prediction.get("predicted_stars", 0)),
        "sentiment": str(prediction.get("sentiment", "")),
        "confidence": confidence,
        "latency_ms": float(latency_ms) if latency_ms is not None else None,
        "review_hash": _hash_review(review_body, review_title),
    }
    client.collection(COLLECTION_INFERENCE).document(inference_id).set(inference_doc)

    reasons: list[str] = []
    if confidence < confidence_threshold:
        reasons.append("low_confidence")
    if include_escalations and "escalated" in model_used:
        reasons.append("escalated_path")
    if random.random() < random_sample_rate:
        reasons.append("random_audit")

    queued = bool(reasons)
    if queued:
        queue_id = str(uuid.uuid4())
        priority = 1 if "low_confidence" in reasons else (2 if "escalated_path" in reasons else 3)
        queue_doc = {
            "inference_id": inference_id,
            "reasons": reasons,
            "priority": priority,
            "status": "pending",
            "assigned_to": None,
            "created_at": _server_timestamp(),
        }
        client.collection(COLLECTION_HUMAN_QUEUE).document(queue_id).set(queue_doc)

    return {
        "logged": True,
        "inference_id": inference_id,
        "queued_for_review": queued,
        "review_reasons": reasons,
    }


def list_human_review_queue(client: Any, status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
    """Return queued reviews for reviewer workflows."""
    if client is None:
        return []

    safe_limit = max(1, min(limit, 200))
    query = client.collection(COLLECTION_HUMAN_QUEUE).where("status", "==", status).limit(safe_limit)

    items: list[dict[str, Any]] = []
    for doc in query.stream():
        row = doc.to_dict() or {}
        row["id"] = doc.id

        inference_id = row.get("inference_id")
        if inference_id:
            inference_doc = client.collection(COLLECTION_INFERENCE).document(str(inference_id)).get()
            if inference_doc.exists:
                row["inference"] = inference_doc.to_dict()

        items.append(row)

    return items


def submit_human_label(
    client: Any,
    queue_id: str,
    human_stars: int,
    reviewer_id: str,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Store reviewer label and resolve the queue item."""
    if client is None or firestore is None:
        raise ValueError("Firestore client is not available")

    if human_stars < 1 or human_stars > 5:
        raise ValueError("human_stars must be in [1, 5]")

    queue_ref = client.collection(COLLECTION_HUMAN_QUEUE).document(queue_id)
    queue_snapshot = queue_ref.get()
    if not queue_snapshot.exists:
        raise LookupError(f"Queue item not found: {queue_id}")

    queue_data = queue_snapshot.to_dict() or {}
    if queue_data.get("status") == "resolved":
        raise ValueError(f"Queue item already resolved: {queue_id}")

    inference_id = str(queue_data.get("inference_id", "")).strip()
    if not inference_id:
        raise ValueError("Queue item does not reference an inference record")

    label_doc = {
        "inference_id": inference_id,
        "human_stars": int(human_stars),
        "reviewer_id": reviewer_id,
        "notes": notes,
        "labeled_at": _server_timestamp(),
    }
    client.collection(COLLECTION_HUMAN_LABELS).document(inference_id).set(label_doc)

    queue_ref.update(
        {
            "status": "resolved",
            "resolved_at": _server_timestamp(),
            "resolved_by": reviewer_id,
        }
    )

    inference_ref = client.collection(COLLECTION_INFERENCE).document(inference_id)
    try:
        inference_ref.update(
            {
                "human_stars": int(human_stars),
                "human_label_at": _server_timestamp(),
                "human_reviewer_id": reviewer_id,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive update
        log.warning("Failed to update inference with human label: %s", exc)

    return {
        "queue_id": queue_id,
        "inference_id": inference_id,
        "status": "resolved",
        "human_stars": int(human_stars),
    }


def _to_numeric_array(records: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return np.array(values, dtype=np.float64)


def _to_string_list(records: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        values.append(str(value))
    return values


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)

    if len(edges) < 2:
        ref_mean = float(np.mean(reference))
        cur_mean = float(np.mean(current))
        denom = abs(ref_mean) + 1e-6
        return abs(cur_mean - ref_mean) / denom

    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)

    ref_ratio = ref_hist / max(ref_hist.sum(), 1)
    cur_ratio = cur_hist / max(cur_hist.sum(), 1)

    eps = 1e-6
    return float(np.sum((cur_ratio - ref_ratio) * np.log((cur_ratio + eps) / (ref_ratio + eps))))


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _js_divergence(reference_values: list[str], current_values: list[str]) -> float:
    if not reference_values or not current_values:
        return 0.0

    ref_count = Counter(reference_values)
    cur_count = Counter(current_values)

    keys = sorted(set(ref_count.keys()) | set(cur_count.keys()))
    ref_arr = np.array([ref_count.get(key, 0.0) for key in keys], dtype=np.float64)
    cur_arr = np.array([cur_count.get(key, 0.0) for key in keys], dtype=np.float64)

    ref_arr = ref_arr / max(ref_arr.sum(), 1.0)
    cur_arr = cur_arr / max(cur_arr.sum(), 1.0)

    mix = 0.5 * (ref_arr + cur_arr)
    return 0.5 * _kl_div(ref_arr, mix) + 0.5 * _kl_div(cur_arr, mix)


def _metric_status(value: float, warn_threshold: float, alert_threshold: float) -> str:
    if value >= alert_threshold:
        return "alert"
    if value >= warn_threshold:
        return "warn"
    return "ok"


def _fetch_inference_logs(client: Any, start: datetime, end: datetime, max_docs: int = 50000) -> list[dict[str, Any]]:
    query = (
        client.collection(COLLECTION_INFERENCE)
        .where("created_at", ">=", start)
        .where("created_at", "<", end)
        .limit(max_docs)
    )

    rows: list[dict[str, Any]] = []
    for doc in query.stream():
        data = doc.to_dict() or {}
        data["id"] = doc.id
        rows.append(data)

    return rows


def run_drift_detection(
    client: Any,
    lookback_hours: int = 24,
    baseline_days: int = 30,
    min_samples: int = 200,
) -> dict[str, Any]:
    """Compute and store drift metrics in Firestore."""
    if client is None or firestore is None:
        raise ValueError("Firestore client is not available")

    now = datetime.now(timezone.utc)
    current_start = now - timedelta(hours=max(1, lookback_hours))
    baseline_end = current_start
    baseline_start = baseline_end - timedelta(days=max(1, baseline_days))

    baseline_logs = _fetch_inference_logs(client, baseline_start, baseline_end)
    current_logs = _fetch_inference_logs(client, current_start, now)

    if len(baseline_logs) < min_samples or len(current_logs) < min_samples:
        return {
            "status": "insufficient_data",
            "baseline_count": len(baseline_logs),
            "current_count": len(current_logs),
            "window_start": current_start,
            "window_end": now,
            "metrics": [],
            "message": (
                "Not enough records for drift detection. "
                f"Need at least {min_samples} in baseline and current windows."
            ),
        }

    psi_warn = _env_float("DRIFT_PSI_WARN", 0.20)
    psi_alert = _env_float("DRIFT_PSI_ALERT", 0.30)
    js_warn = _env_float("DRIFT_JS_WARN", 0.10)
    js_alert = _env_float("DRIFT_JS_ALERT", 0.20)

    baseline_conf = _to_numeric_array(baseline_logs, "confidence")
    current_conf = _to_numeric_array(current_logs, "confidence")
    baseline_len = _to_numeric_array(baseline_logs, "text_length")
    current_len = _to_numeric_array(current_logs, "text_length")

    baseline_lang = _to_string_list(baseline_logs, "language")
    current_lang = _to_string_list(current_logs, "language")
    baseline_cat = _to_string_list(baseline_logs, "product_category")
    current_cat = _to_string_list(current_logs, "product_category")
    baseline_route = _to_string_list(baseline_logs, "model_used")
    current_route = _to_string_list(current_logs, "model_used")

    low_conf_threshold = _env_float("HITL_CONFIDENCE_THRESHOLD", 0.60)
    baseline_low_conf_rate = float(np.mean(baseline_conf < low_conf_threshold)) if len(baseline_conf) else 0.0
    current_low_conf_rate = float(np.mean(current_conf < low_conf_threshold)) if len(current_conf) else 0.0
    low_conf_delta = current_low_conf_rate - baseline_low_conf_rate

    metric_specs = [
        {
            "metric_name": "confidence_psi",
            "metric_value": _psi(baseline_conf, current_conf),
            "warn_threshold": psi_warn,
            "alert_threshold": psi_alert,
        },
        {
            "metric_name": "text_length_psi",
            "metric_value": _psi(baseline_len, current_len),
            "warn_threshold": psi_warn,
            "alert_threshold": psi_alert,
        },
        {
            "metric_name": "language_jsd",
            "metric_value": _js_divergence(baseline_lang, current_lang),
            "warn_threshold": js_warn,
            "alert_threshold": js_alert,
        },
        {
            "metric_name": "product_category_jsd",
            "metric_value": _js_divergence(baseline_cat, current_cat),
            "warn_threshold": js_warn,
            "alert_threshold": js_alert,
        },
        {
            "metric_name": "route_mix_jsd",
            "metric_value": _js_divergence(baseline_route, current_route),
            "warn_threshold": js_warn,
            "alert_threshold": js_alert,
        },
        {
            "metric_name": "low_confidence_rate_delta",
            "metric_value": float(abs(low_conf_delta)),
            "warn_threshold": _env_float("DRIFT_LOW_CONF_DELTA_WARN", 0.05),
            "alert_threshold": _env_float("DRIFT_LOW_CONF_DELTA_ALERT", 0.10),
        },
    ]

    metrics: list[dict[str, Any]] = []
    for spec in metric_specs:
        status = _metric_status(spec["metric_value"], spec["warn_threshold"], spec["alert_threshold"])
        metric_row = {
            "metric_name": spec["metric_name"],
            "metric_value": float(spec["metric_value"]),
            "warn_threshold": float(spec["warn_threshold"]),
            "threshold": float(spec["alert_threshold"]),
            "status": status,
            "window_start": current_start,
            "window_end": now,
            "baseline_start": baseline_start,
            "baseline_end": baseline_end,
            "baseline_count": len(baseline_logs),
            "current_count": len(current_logs),
            "created_at": _server_timestamp(),
        }
        client.collection(COLLECTION_DRIFT_METRICS).document(str(uuid.uuid4())).set(metric_row)
        metrics.append(
            {
                "metric_name": metric_row["metric_name"],
                "metric_value": metric_row["metric_value"],
                "warn_threshold": metric_row["warn_threshold"],
                "threshold": metric_row["threshold"],
                "status": metric_row["status"],
                "baseline_count": metric_row["baseline_count"],
                "current_count": metric_row["current_count"],
                "window_start": metric_row["window_start"],
                "window_end": metric_row["window_end"],
                "created_at": None,
            }
        )

    severity_rank = {"ok": 0, "warn": 1, "alert": 2}
    worst_status = "ok"
    for metric in metrics:
        metric_status = metric["status"]
        if severity_rank[metric_status] > severity_rank[worst_status]:
            worst_status = metric_status

    return {
        "status": worst_status,
        "baseline_count": len(baseline_logs),
        "current_count": len(current_logs),
        "window_start": current_start,
        "window_end": now,
        "metrics": metrics,
        "message": "Drift metrics computed and stored.",
    }


def get_latest_drift_metrics(client: Any, limit: int = 20) -> list[dict[str, Any]]:
    """Fetch most recent drift metrics for dashboards and API reads."""
    if client is None or firestore is None:
        return []

    safe_limit = max(1, min(limit, 200))
    query = (
        client.collection(COLLECTION_DRIFT_METRICS)
        .order_by("created_at", direction="DESCENDING")
        .limit(safe_limit)
    )

    rows: list[dict[str, Any]] = []
    for doc in query.stream():
        row = doc.to_dict() or {}
        row["id"] = doc.id
        rows.append(row)

    return rows
