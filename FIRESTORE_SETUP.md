# Firestore Integration Setup

This project now supports:
- Firestore inference logging
- Human-in-the-loop review queue
- Human labeling endpoint
- Drift detection and drift metric storage

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Create Firebase service account

1. Open Firebase Console and select your project.
2. Go to Project Settings -> Service accounts.
3. Generate a new private key JSON file.
4. Save it securely (do not commit it to git).

## 3) Configure environment variables

You can run without setting credential path manually.

### Option A: Auto-discovery from project root (default)

If your service account file is in project root with the exact filename below,
the API will automatically use it:

- `firebase-service-account.json`

### Option B: Explicit path in environment variable

Use PowerShell syntax:

```powershell
$env:FIRESTORE_ENABLED = "true"
$env:FIREBASE_PROJECT_ID = "your-project-id"
$env:FIREBASE_CREDENTIALS_PATH = "C:\path\to\service-account.json"
```

For CMD syntax:

```bat
set FIRESTORE_ENABLED=true
set FIREBASE_PROJECT_ID=your-project-id
set FIREBASE_CREDENTIALS_PATH=C:\path\to\service-account.json
```

### Option C: Inline service account JSON

```powershell
$env:FIRESTORE_ENABLED = "true"
$env:FIREBASE_PROJECT_ID = "your-project-id"
$env:FIREBASE_CREDENTIALS_JSON = "{...full_service_account_json...}"
```

If no env credentials are set and no local JSON is found, Application Default Credentials are used.

## 4) Optional HITL queue policy controls

```powershell
$env:HITL_CONFIDENCE_THRESHOLD = "0.60"
$env:HITL_RANDOM_SAMPLE_RATE = "0.02"
$env:HITL_INCLUDE_ESCALATIONS = "true"
```

## 5) Optional drift thresholds

```powershell
$env:DRIFT_PSI_WARN = "0.20"
$env:DRIFT_PSI_ALERT = "0.30"
$env:DRIFT_JS_WARN = "0.10"
$env:DRIFT_JS_ALERT = "0.20"
$env:DRIFT_LOW_CONF_DELTA_WARN = "0.05"
$env:DRIFT_LOW_CONF_DELTA_ALERT = "0.10"
```

## 6) Run the API

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

If PowerShell launches the wrong global Python, run with explicit venv interpreter:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open in browser using:

- http://localhost:8000/health
- http://127.0.0.1:8000/health

Do not use http://0.0.0.0:8000 in the browser. `0.0.0.0` is a server bind address, not a client destination URL.

## 7) New endpoints

- `GET /health`
  - Includes `firestore_connected`
- `POST /predict`
  - Logs inference to Firestore
  - Auto-enqueues uncertain cases for human review
  - Returns `inference_id`, `queued_for_review`, `review_reasons`
- `GET /human-review/queue?status=pending&limit=50`
  - Lists queue items
- `POST /human-review/{queue_id}/label`
  - Stores human label and resolves queue item
- `POST /drift/run`
  - Computes drift metrics from logged inference data
- `GET /drift/latest?limit=20`
  - Returns recent drift metrics

## 8) Firestore collections used

- `inference_log`
- `human_review_queue`
- `human_labels`
- `drift_metrics`

Collection names can be customized with:
- `FIRESTORE_INFERENCE_COLLECTION`
- `FIRESTORE_HUMAN_QUEUE_COLLECTION`
- `FIRESTORE_HUMAN_LABELS_COLLECTION`
- `FIRESTORE_DRIFT_COLLECTION`
