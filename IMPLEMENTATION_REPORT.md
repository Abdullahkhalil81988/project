# Implementation Report

Date: 2026-04-20
Project: Programming for AI - Review Sentiment/Star Prediction API

## 1. What We Were Asked To Do

The work evolved in phases:

1. Create a CV-friendly summary of the project.
2. Explain the complete workflow and identify room for innovation.
3. Add human-in-the-loop (HITL) review support and drift detection.
4. Use Firebase Firestore as the persistence layer.
5. Make credential setup easier by auto-picking local credentials where possible.
6. Protect credential files and runtime noise with .gitignore updates.
7. Add automated tests for all new behavior.
8. Fix test errors in the Firestore API test file.
9. Finish remaining todo items and validate results.

## 2. High-Level Outcome

The API moved from inference-only behavior to a more production-ready MLOps flow:

- Prediction persistence in Firestore.
- Automatic queueing of uncertain cases for human review.
- Human label submission endpoint to close the feedback loop.
- Drift detection pipeline with persisted metrics.
- Better service health visibility (including Firestore status).
- Expanded automated tests (unit, endpoint-level, HTTP integration, emulator e2e).
- CI workflow for regression protection.

## 3. Architecture Changes

## 3.1 Before

- FastAPI service accepted review input and returned prediction.
- No persistent store for inference logs.
- No structured HITL queue.
- No drift metrics or drift endpoint.

## 3.2 After

End-to-end flow now supports:

1. Client calls POST /predict.
2. Inference runs through existing model-routing logic.
3. Inference metadata is persisted to Firestore.
4. Case is optionally queued for human review based on policy rules.
5. Reviewer fetches queue and submits human label.
6. Label is persisted and linked back to inference.
7. Drift job computes metrics from historical vs current windows.
8. Drift metrics are stored and served by API.

## 4. File-by-File Change Log

## 4.1 API Core

### api/main.py

Main integration point for Firestore and new operational endpoints.

Key updates:

- Added Firestore lifecycle state:
  - FIRESTORE_STATE with client, connected, error.
- During startup (lifespan):
  - Attempts Firestore connection using service helper.
  - Stores connection status and error details.
- During shutdown:
  - Clears model and Firestore state safely.
- Health endpoint:
  - Added firestore_connected field in response.
  - Reports Firestore issue details when models are loaded but Firestore is unavailable.
- Added _require_firestore_client helper:
  - Standardized 503 response when Firestore is unavailable.
- Predict endpoint enhancements:
  - Captures latency via perf_counter.
  - Keeps inference functional even if Firestore persistence fails.
  - Persists inference metadata and potential review queue outcome.
  - Returns additional response fields:
    - inference_id
    - queued_for_review
    - review_reasons
- New endpoints:
  - GET /human-review/queue
  - POST /human-review/{queue_id}/label
  - POST /drift/run
  - GET /drift/latest

### api/firestore_service.py

New service module implementing Firestore-related logic.

Main capabilities:

- Firestore client initialization and credential resolution.
- Local service-account auto-discovery.
- Inference logging and conditional HITL queueing.
- Human queue listing and label submission.
- Drift calculation and persistence.

Credential resolution order implemented:

1. FIREBASE_CREDENTIALS_PATH
2. FIREBASE_CREDENTIALS_JSON
3. Auto-discover local JSON in repo root:
   - firebase-service-account.json
   - *firebase-adminsdk-*.json
4. Application Default Credentials (ADC)

HITL decision rules:

- low_confidence if confidence < HITL_CONFIDENCE_THRESHOLD
- escalated_path if routing model indicates escalation and HITL_INCLUDE_ESCALATIONS=true
- random_audit by HITL_RANDOM_SAMPLE_RATE

Drift metrics implemented:

- confidence_psi
- text_length_psi
- language_jsd
- product_category_jsd
- route_mix_jsd
- low_confidence_rate_delta

Drift status policy:

- ok / warn / alert based on configurable thresholds.
- insufficient_data when baseline/current samples are below minimum.

### api/schemas.py

Expanded API contract to support Firestore/HITL/drift.

Added/updated models:

- PredictionResponse
  - inference_id
  - queued_for_review
  - review_reasons
- HealthResponse
  - firestore_connected
- HumanReviewQueueItem
- HumanLabelRequest
- HumanLabelResponse
- DriftMetricResponse
- DriftRunResponse

## 4.2 Dependency and Config

### requirements.txt

- Added firebase-admin dependency.

### .gitignore

Added ignores for sensitive and runtime files:

- firebase-service-account.json
- *firebase-adminsdk-*.json
- uvicorn.stdout.log
- uvicorn.stderr.log

## 4.3 Documentation

### FIRESTORE_SETUP.md

New setup guide covering:

- Dependency install.
- Firebase service account creation.
- Credential options (path/json/auto-discovery/ADC).
- PowerShell and CMD env syntax.
- Optional HITL and drift thresholds.
- Correct API startup commands (including explicit venv interpreter).
- Browser access guidance (localhost/127.0.0.1, not 0.0.0.0).
- New endpoint list and Firestore collections.

## 4.4 Test Suite

### tests/test_firestore_service.py

Unit tests for Firestore service logic with in-memory dummy client classes:

- Firestore disabled behavior.
- Low-confidence + escalation queueing behavior.
- Human-label flow updates queue, labels, and inference.
- Input validation for invalid rating.
- Drift insufficient-data path.

### tests/test_api_firestore_features.py

Direct async endpoint-function tests:

- Health reporting when Firestore disconnected.
- Firestore client requirement enforcement (503).
- Predict behavior when models unavailable (503).
- Predict response enrichment with Firestore metadata.
- Human queue and labeling error mapping.
- Drift endpoint response shape checks.

Compile issue fixed during session:

- Explicitly provided review_title=None where required.

### tests/test_api_testclient_integration.py

HTTP-level in-process integration tests using FastAPI TestClient:

- Health endpoint shape.
- Predict happy path and degraded path.
- Human review queue behavior and Firestore dependency checks.
- Human label endpoint response.
- Drift run + latest endpoint payload checks.

### tests/test_firestore_emulator_e2e.py

Guarded end-to-end tests for real Firestore interactions with emulator:

- Skips unless FIRESTORE_E2E_RUN=1 and FIRESTORE_EMULATOR_HOST is set.
- Uses isolated per-test collection names.
- Verifies log -> queue -> label roundtrip.
- Verifies drift insufficient-data behavior.

## 4.5 CI

### .github/workflows/python-tests.yml

New CI workflow:

- Triggers on push to main, pull_request, and manual dispatch.
- Uses Python 3.12.
- Installs requirements + pytest/httpx.
- Runs targeted core and new test files.
- Sets FIRESTORE_ENABLED=false and FIRESTORE_E2E_RUN=0 for safe default CI behavior.

## 5. Endpoint-Level Functional Additions

### POST /predict

Now returns:

- model prediction fields
- inference_id (if persisted)
- queued_for_review (bool)
- review_reasons (list)

### GET /human-review/queue

- Reads pending/relevant queue items from Firestore.
- Supports status and limit query params.

### POST /human-review/{queue_id}/label

- Accepts reviewer label payload.
- Stores label and resolves queue item.
- Updates linked inference record with human label metadata.

### POST /drift/run

- Executes drift analysis for baseline vs lookback windows.
- Writes metrics to drift collection.
- Returns summary status and metric set.

### GET /drift/latest

- Fetches latest persisted drift metrics.

### GET /health

- Includes firestore_connected status.
- Surfaces Firestore issue details when relevant.

## 6. Environment Variables and Controls

Firestore controls:

- FIRESTORE_ENABLED (default true)
- FIREBASE_PROJECT_ID
- FIREBASE_CREDENTIALS_PATH
- FIREBASE_CREDENTIALS_JSON
- FIREBASE_DEFAULT_CREDENTIALS_FILE
- FIREBASE_REQUIRE_LOCAL_CREDENTIALS

Collection overrides:

- FIRESTORE_INFERENCE_COLLECTION
- FIRESTORE_HUMAN_QUEUE_COLLECTION
- FIRESTORE_HUMAN_LABELS_COLLECTION
- FIRESTORE_DRIFT_COLLECTION

HITL policy:

- HITL_CONFIDENCE_THRESHOLD
- HITL_RANDOM_SAMPLE_RATE
- HITL_INCLUDE_ESCALATIONS

Drift thresholds:

- DRIFT_PSI_WARN
- DRIFT_PSI_ALERT
- DRIFT_JS_WARN
- DRIFT_JS_ALERT
- DRIFT_LOW_CONF_DELTA_WARN
- DRIFT_LOW_CONF_DELTA_ALERT

Emulator test controls:

- FIRESTORE_E2E_RUN
- FIRESTORE_EMULATOR_HOST

## 7. Issues Encountered and How They Were Resolved

1. PowerShell env var syntax issue:
- Problem: using shell syntax incompatible with PowerShell.
- Fix: documented and used $env:VAR="value".

2. Invalid browser URL with 0.0.0.0:
- Problem: browser cannot use 0.0.0.0 destination.
- Fix: documented localhost/127.0.0.1 usage.

3. Interpreter mismatch (global python vs venv):
- Problem: some starts used wrong interpreter context.
- Fix: documented explicit venv command for uvicorn.

4. Port binding conflict (8000 already in use):
- Problem: startup failures with WinError 10048.
- Fix: identified listener and cleared process before relaunch.

5. Test compile errors in Firestore API tests:
- Problem: schema-required fields missing in test payload.
- Fix: adjusted payloads and revalidated diagnostics/tests.

## 8. Validation and Test Results

Recent targeted suite execution completed successfully:

- Command run:
  - .\.venv\Scripts\python.exe -m pytest tests/test_schemas.py tests/test_engine.py tests/test_firestore_service.py tests/test_api_firestore_features.py tests/test_api_testclient_integration.py tests/test_firestore_emulator_e2e.py -q
- Result:
  - 90 passed
  - 2 skipped (expected emulator-gated tests when emulator is not enabled)

## 9. Completed Work Summary

Completed and verified:

- Firestore integration in API lifecycle and predict flow.
- HITL queue + label submission API.
- Drift detection run + retrieval API.
- Extended schemas and response contracts.
- Setup documentation and env guidance.
- Secret/runtime file ignore hardening.
- Unit + endpoint + integration + e2e (guarded) tests.
- CI workflow for new and core tests.
- Test file errors resolved.

## 10. Suggested Next Improvements

1. Add scripts for stable local operations on Windows:
   - scripts/start_api.ps1
   - scripts/run_tests.ps1
   - scripts/run_firestore_emulator_tests.ps1

2. Add a dedicated CI job for emulator tests:
   - Run only when emulator service is provided.

3. Add a lightweight dashboard for:
   - queue depth
   - reviewer throughput
   - drift trends over time

4. Add alerting for drift status transitions:
   - warn/alert notifications via email or chat webhook.

5. Add active-learning loop:
   - periodic retraining using high-confidence human-labeled data.

## 11. CV-Ready Impact Statement (Optional)

Designed and implemented a production-style MLOps extension for a FastAPI NLP inference service by integrating Firebase Firestore for inference logging, human-in-the-loop review workflows, and automated drift detection (PSI/JSD/quality-rate metrics), then hardened delivery with multi-layer automated testing and CI.
