import os
import numpy as np
import pandas as pd
import torch


MODEL_B_MAX_LENGTH = 128
MODEL_A_ESCALATION_THRESHOLD = 0.55
MODEL_B_ESCALATION_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Configurable category sets (override via environment variables)
# ---------------------------------------------------------------------------
# Categories that always route to the stacking ensemble (model_c).
_default_c = "book,digital_ebook_purchase,electronics,pc"
MODEL_C_CATEGORIES = set(os.getenv("MODEL_C_CATEGORIES", _default_c).split(","))

# If model_b confidence falls below MODEL_B_ESCALATION_THRESHOLD AND the
# product category is in this set, escalate to model_c for a second opinion.
_default_b_esc = "software,video_games,music,movies"
MODEL_B_ESCALATION_CATEGORIES = set(
    os.getenv("MODEL_B_ESCALATION_CATEGORIES", _default_b_esc).split(",")
)


# ---------------------------------------------------------------------------
# Text signal extraction
# ---------------------------------------------------------------------------
def compute_text_signals(text: str) -> dict:
    """
    Derive script-level signals from the raw review body.

    Returns
    -------
    non_ascii_ratio : float  – fraction of characters with ord > 127
    has_cjk         : bool   – Chinese / Japanese / Korean script present
    has_arabic      : bool   – Arabic script present
    has_cyrillic    : bool   – Cyrillic script present
    is_low_quality  : bool   – all-caps, heavy repetition, or very short
    """
    if not text:
        return {
            "non_ascii_ratio": 0.0,
            "has_cjk": False,
            "has_arabic": False,
            "has_cyrillic": False,
            "is_low_quality": False,
        }

    total = len(text)
    non_ascii = sum(1 for c in text if ord(c) > 127)

    has_cjk = any(
        "\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff"
        for c in text
    )
    has_arabic   = any("\u0600" <= c <= "\u06ff" for c in text)
    has_cyrillic = any("\u0400" <= c <= "\u04ff" for c in text)

    # Quality signals
    words = text.split()
    alpha = [c for c in text if c.isalpha()]
    all_caps = bool(alpha) and (sum(1 for c in alpha if c.isupper()) / len(alpha)) > 0.8
    high_repetition = bool(words) and (max(words.count(w) for w in set(words)) / len(words)) > 0.5
    is_low_quality  = all_caps or high_repetition

    return {
        "non_ascii_ratio": non_ascii / total,
        "has_cjk":         has_cjk,
        "has_arabic":      has_arabic,
        "has_cyrillic":    has_cyrillic,
        "is_low_quality":  is_low_quality,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def select_model(language: str, text_length: int, product_category: str,
                 review_body: str = "") -> str:
    # 1. Special categories always use the stacking ensemble.
    if product_category in MODEL_C_CATEGORIES:
        return "model_c"

    # 2. Script-aware language check — override declared language when the
    #    actual text contains non-Latin scripts or heavy non-ASCII content.
    signals = compute_text_signals(review_body)

    if review_body and signals["is_low_quality"]:
        # Low-quality text (all-caps, spammy repetition) → robust multilingual model.
        return "model_b"

    NON_ASCII_THRESHOLD = 0.15
    text_is_latin_english = (
        language == "en"
        and signals["non_ascii_ratio"] < NON_ASCII_THRESHOLD
        and not signals["has_cjk"]
        and not signals["has_arabic"]
        and not signals["has_cyrillic"]
    )

    if text_is_latin_english and 15 <= text_length < 80:
        return "model_a"

    return "model_b"


def preprocess_incoming_review(review_body: str, review_title: str | None = None) -> dict:
    body = str(review_body).strip()
    if review_title is None or str(review_title).strip() == "":
        fallback_title = " ".join(body.split()[:5]) + "..."
        title = fallback_title
    else:
        title = str(review_title).strip()

    model_text = (title + " " + body).strip()
    text_length = len(body.split())

    return {
        "review_body": body,
        "review_title": title,
        "model_text": model_text,
        "text_length": text_length,
    }


def run_model(model_name: str, model_text: str, models: dict) -> dict:
    text = str(model_text)

    if model_name == "model_b":
        tokenizer_b = models["model_b_tokenizer"]
        model_b = models["model_b"]
        device = next(model_b.parameters()).device
        inputs = tokenizer_b(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MODEL_B_MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            logits = model_b(**inputs).logits
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_stars = int(np.argmax(proba) + 1)
        confidence = float(np.max(proba))
        sentiment = "positive" if predicted_stars >= 4 else ("negative" if predicted_stars <= 2 else "neutral")

        return {
            "predicted_stars": predicted_stars,
            "sentiment": sentiment,
            "confidence": confidence,
            "model_used": "model_b",
        }

    raise ValueError("Unsupported model")


def _run_model_c(text: str, base_proba: np.ndarray, base_model_used: str,
                 product_category: str, models: dict) -> dict:
    """Run the model_c meta-learner given pre-computed base probabilities."""
    model_c = models["model_c"]
    model_c_categories = models["model_c_categories"]

    category_df = pd.get_dummies(pd.Series([product_category]), prefix="category")
    category_df = category_df.reindex(columns=model_c_categories, fill_value=0)

    feature_row = pd.DataFrame([base_proba], columns=[f"prob_{i}" for i in range(1, 6)])
    X_meta = pd.concat([feature_row, category_df.reset_index(drop=True)], axis=1)

    meta_proba = model_c.predict_proba(X_meta)[0]
    predicted_stars = int(model_c.predict(X_meta)[0])
    confidence = float(np.max(meta_proba))
    sentiment = "positive" if predicted_stars >= 4 else ("negative" if predicted_stars <= 2 else "neutral")

    return {
        "predicted_stars": predicted_stars,
        "sentiment": sentiment,
        "confidence": confidence,
        "model_used": "model_c",
        "base_model_used": base_model_used,
    }


def run_inference(review_body: str, language: str, product_category: str,
                  models: dict, review_title: str | None = None) -> dict:
    prepared = preprocess_incoming_review(review_body, review_title)
    text_length = prepared["text_length"]
    text = prepared["model_text"]

    # Pass the actual body text so select_model can use script/quality signals.
    selected = select_model(language, text_length, product_category,
                            review_body=prepared["review_body"])

    # --- Model A ---
    if selected == "model_a":
        proba = models["model_a"].predict_proba([text])[0]
        predicted_stars = int(np.argmax(proba) + 1)
        confidence = float(np.max(proba))

        # Escalate up to model_b if confidence is too low.
        if confidence < MODEL_A_ESCALATION_THRESHOLD:
            result = run_model("model_b", text, models)
            result["model_used"] = "model_b_escalated"
            return result

        sentiment = "positive" if predicted_stars >= 4 else ("negative" if predicted_stars <= 2 else "neutral")
        return {
            "predicted_stars": predicted_stars,
            "sentiment": sentiment,
            "confidence": confidence,
            "model_used": "model_a",
        }

    # --- Model B ---
    if selected == "model_b":
        tokenizer_b = models["model_b_tokenizer"]
        model_b = models["model_b"]
        device = next(model_b.parameters()).device
        inputs = tokenizer_b(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MODEL_B_MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            logits = model_b(**inputs).logits
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_stars = int(np.argmax(proba) + 1)
        confidence = float(np.max(proba))

        # Escalate to model_c if uncertain and category qualifies.
        if (confidence < MODEL_B_ESCALATION_THRESHOLD
                and product_category in MODEL_B_ESCALATION_CATEGORIES):
            result = _run_model_c(text, proba, "model_b", product_category, models)
            result["model_used"] = "model_b_escalated_to_c"
            return result

        sentiment = "positive" if predicted_stars >= 4 else ("negative" if predicted_stars <= 2 else "neutral")
        return {
            "predicted_stars": predicted_stars,
            "sentiment": sentiment,
            "confidence": confidence,
            "model_used": "model_b",
        }

    # --- Model C ---
    if language == "en":
        base_proba = models["model_a"].predict_proba([text])[0]
        base_model_used = "model_a"
    else:
        device = next(models["model_b"].parameters()).device
        inputs = models["model_b_tokenizer"](
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MODEL_B_MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            logits = models["model_b"](**inputs).logits
            base_proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        base_model_used = "model_b"

    return _run_model_c(text, base_proba, base_model_used, product_category, models)
