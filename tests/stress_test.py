"""
Stress Test – ReviewRoute API
==============================
Uses Locust to measure latency and throughput under load.

Install:
    pip install locust

Run (headless, 100 users, 10 spawned/s, 60 s duration):
    locust -f tests/stress_test.py --headless -u 100 -r 10 -t 60s --host http://localhost:8000

Run with web UI:
    locust -f tests/stress_test.py --host http://localhost:8000
    Then open http://localhost:8089

Key metrics reported by Locust:
    - Requests/s (throughput)
    - Median / 95th / 99th percentile latency (ms)
    - Failure rate
"""

import random
from locust import HttpUser, between, task

# ---------------------------------------------------------------------------
# Representative sample payloads
# ---------------------------------------------------------------------------
ENGLISH_REVIEWS = [
    {
        "review_body": "This laptop is fantastic. The battery lasts all day and performance is excellent.",
        "review_title": "Best laptop ever",
        "language": "en",
        "product_category": "electronics",
    },
    {
        "review_body": "Terrible product. Broke after two days and customer support was useless.",
        "review_title": "Complete waste of money",
        "language": "en",
        "product_category": "electronics",
    },
    {
        "review_body": "Average book. Some interesting chapters but overall it dragged on too long.",
        "review_title": "Decent read",
        "language": "en",
        "product_category": "book",
    },
    {
        "review_body": "Good quality shirt. Fits true to size and the fabric feels premium.",
        "language": "en",
        "product_category": "apparel",
    },
    {
        "review_body": "Not bad but not great either. Does what it says on the tin I suppose.",
        "language": "en",
        "product_category": "kitchen",
    },
]

MULTILINGUAL_REVIEWS = [
    {
        "review_body": "Produit excellent, livraison rapide. Je recommande vivement à tous.",
        "review_title": "Très satisfait",
        "language": "fr",
        "product_category": "apparel",
    },
    {
        "review_body": "Sehr gutes Produkt. Qualität ist hervorragend und der Preis ist fair.",
        "review_title": "Empfehlenswert",
        "language": "de",
        "product_category": "electronics",
    },
    {
        "review_body": "Producto terrible, llegó roto y el servicio al cliente no ayudó.",
        "language": "es",
        "product_category": "kitchen",
    },
]

ENSEMBLE_REVIEWS = [
    {
        "review_body": "Incredible novel, one of the best science fiction books I have read in years.",
        "review_title": "A masterpiece",
        "language": "en",
        "product_category": "book",
    },
    {
        "review_body": "The ebook format was great but the content was disappointing.",
        "language": "en",
        "product_category": "digital_ebook_purchase",
    },
]

# Validation-error payloads (expected to return 422).
# Use missing required fields — FastAPI enforces these before any validator runs,
# so they reliably produce 422 regardless of Pydantic version or validator order.
INVALID_REVIEWS = [
    {"language": "en", "product_category": "electronics"},               # missing review_body
    {"review_body": "Great product", "product_category": "electronics"}, # missing language
    {"review_body": "Great product", "language": "en"},                  # missing product_category
]

ALL_VALID = ENGLISH_REVIEWS + MULTILINGUAL_REVIEWS + ENSEMBLE_REVIEWS


# ---------------------------------------------------------------------------
# Locust user definition
# ---------------------------------------------------------------------------
class ReviewRouteUser(HttpUser):
    """Simulates a client hitting the ReviewRoute API."""

    # Wait between 0.5 s and 2 s between tasks (simulates realistic traffic)
    wait_time = between(0.5, 2)

    # ------------------------------------------------------------------
    # Tasks (weight = relative frequency)
    # ------------------------------------------------------------------
    @task(8)
    def predict_valid(self):
        """POST /predict with a valid review — main throughput task."""
        payload = random.choice(ALL_VALID)
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 503:
                resp.failure("Service unavailable – models not loaded")
            else:
                resp.failure(f"Unexpected status {resp.status_code}: {resp.text[:200]}")

    @task(1)
    def predict_invalid(self):
        """POST /predict with bad input — validates 422 handling."""
        payload = random.choice(INVALID_REVIEWS)
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code == 422:
                resp.success()
            else:
                resp.failure(f"Expected 422, got {resp.status_code}")

    @task(1)
    def health_check(self):
        """GET /health — lightweight liveness probe."""
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if body.get("models_loaded"):
                    resp.success()
                else:
                    resp.failure("Health check: models_loaded=False")
            else:
                resp.failure(f"Health returned {resp.status_code}")
