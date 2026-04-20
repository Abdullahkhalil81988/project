import os
import re
from typing import Any
import numpy as np
import pandas as pd
import torch


MODEL_B_MAX_LENGTH = 128
MODEL_A_ESCALATION_THRESHOLD = 0.55
MODEL_B_ESCALATION_THRESHOLD = 0.35
_default_model_a_langs = "en,de,es,fr"
MODEL_A_SUPPORTED_LANGUAGES = {
    code.strip().lower()
    for code in os.getenv("MODEL_A_SUPPORTED_LANGUAGES", _default_model_a_langs).split(",")
    if code.strip()
}

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


def _normalize_language_code(language: str | None) -> str | None:
    if language is None:
        return None

    cleaned = str(language).strip().lower().replace("_", "-")
    if not cleaned:
        return None

    if "-" in cleaned:
        cleaned = cleaned.split("-", 1)[0]

    return cleaned if len(cleaned) >= 2 else None


def detect_language(review_body: str, review_title: str | None = None) -> str:
    """
    Best-effort language detection used when API input language is omitted.

    Priority:
    1) Script heuristics (Arabic/Cyrillic/CJK)
    2) langdetect package if present
    3) Lightweight token heuristic over en/de/es/fr
    """
    text = f"{review_title or ''} {review_body or ''}".strip()
    if not text:
        return "en"

    script_signals = compute_text_signals(text)
    if script_signals["has_arabic"]:
        return "ar"
    if script_signals["has_cyrillic"]:
        return "ru"
    if script_signals["has_cjk"]:
        return "zh"

    try:
        from langdetect import detect as langdetect_detect  # type: ignore

        detected = _normalize_language_code(langdetect_detect(text))
        if detected:
            return detected
    except Exception:
        pass

    token_re = re.compile(r"[a-zA-ZÀ-ÿ']+")
    tokens = token_re.findall(text.lower())
    if not tokens:
        return "en"

    stopwords = {
        "en": {
            "the", "and", "is", "are", "with", "this", "that", "for", "very", "not", "was",
            "were", "good", "bad", "product", "quality", "great", "excellent", "it",
        },
        "de": {
            "und", "ist", "sehr", "mit", "nicht", "das", "die", "der", "ein", "eine", "ich",
            "zu", "auf", "war", "sind", "gute", "produkt", "qualitat", "diese", "dieses",
        },
        "es": {
            "el", "la", "los", "las", "y", "es", "muy", "con", "no", "que", "una", "un",
            "este", "esta", "producto", "calidad", "para", "fue", "estoy", "bien",
        },
        "fr": {
            "le", "la", "les", "et", "est", "tres", "avec", "pas", "que", "une", "un", "ce",
            "cet", "cette", "produit", "qualite", "pour", "je", "suis", "bien",
        },
    }

    scores = {"en": 0.0, "de": 0.0, "es": 0.0, "fr": 0.0}
    for token in tokens:
        for language, words in stopwords.items():
            if token in words:
                scores[language] += 1.0

    text_lower = text.lower()
    if any(ch in text_lower for ch in "äöüß"):
        scores["de"] += 2.0
    if any(ch in text_lower for ch in "ñ¡¿"):
        scores["es"] += 2.0
    if any(ch in text_lower for ch in "àâæçéèêëîïôœùûÿ"):
        scores["fr"] += 2.0

    best_language = max(scores.items(), key=lambda pair: pair[1])[0]
    return best_language if scores[best_language] > 0 else "en"


def _resolve_model_a_variant(language: str, models: dict) -> tuple[Any | None, str | None]:
    resolved_language = _normalize_language_code(language) or "en"

    by_language = models.get("model_a_by_language")
    if isinstance(by_language, dict):
        if resolved_language in by_language:
            if resolved_language == "en":
                return by_language[resolved_language], "model_a"
            return by_language[resolved_language], f"model_a_{resolved_language}"
        if resolved_language == "en" and "model_a" in models:
            return models["model_a"], "model_a"
        return None, None

    explicit_key = f"model_a_{resolved_language}"
    if explicit_key in models:
        return models[explicit_key], explicit_key
    if resolved_language == "en" and "model_a" in models:
        return models["model_a"], "model_a"

    return None, None


def _attach_language_context(result: dict, resolved_language: str, language_was_detected: bool) -> dict:
    enriched = dict(result)
    enriched["resolved_language"] = resolved_language
    enriched["language_was_detected"] = language_was_detected
    return enriched


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
def select_model(language: str | None, text_length: int, product_category: str,
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

    resolved_language = _normalize_language_code(language) or "en"

    NON_ASCII_THRESHOLD = 0.30
    text_is_latin_script = (
        signals["non_ascii_ratio"] < NON_ASCII_THRESHOLD
        and not signals["has_cjk"]
        and not signals["has_arabic"]
        and not signals["has_cyrillic"]
    )
    language_supports_model_a = resolved_language in MODEL_A_SUPPORTED_LANGUAGES

    if text_is_latin_script and language_supports_model_a and 15 <= text_length < 80:
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


def run_inference(review_body: str, language: str | None, product_category: str,
                  models: dict, review_title: str | None = None) -> dict:
    prepared = preprocess_incoming_review(review_body, review_title)
    text_length = prepared["text_length"]
    text = prepared["model_text"]

    resolved_language = _normalize_language_code(language)
    language_was_detected = False
    if not resolved_language:
        resolved_language = detect_language(prepared["review_body"], prepared["review_title"])
        language_was_detected = True

    # Pass the actual body text so select_model can use script/quality signals.
    selected = select_model(resolved_language, text_length, product_category,
                            review_body=prepared["review_body"])

    # --- Model A ---
    if selected == "model_a":
        model_a_variant, model_a_name = _resolve_model_a_variant(resolved_language, models)

        if model_a_variant is None:
            fallback = run_model("model_b", text, models)
            fallback["model_used"] = "model_b_language_fallback"
            return _attach_language_context(fallback, resolved_language, language_was_detected)

        proba = model_a_variant.predict_proba([text])[0]
        predicted_stars = int(np.argmax(proba) + 1)
        confidence = float(np.max(proba))

        # Escalate up to model_b if confidence is too low.
        if confidence < MODEL_A_ESCALATION_THRESHOLD:
            result = run_model("model_b", text, models)
            result["model_used"] = "model_b_escalated"
            return _attach_language_context(result, resolved_language, language_was_detected)

        sentiment = "positive" if predicted_stars >= 4 else ("negative" if predicted_stars <= 2 else "neutral")
        return _attach_language_context({
            "predicted_stars": predicted_stars,
            "sentiment": sentiment,
            "confidence": confidence,
            "model_used": model_a_name or "model_a",
        }, resolved_language, language_was_detected)

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
            return _attach_language_context(result, resolved_language, language_was_detected)

        sentiment = "positive" if predicted_stars >= 4 else ("negative" if predicted_stars <= 2 else "neutral")
        return _attach_language_context({
            "predicted_stars": predicted_stars,
            "sentiment": sentiment,
            "confidence": confidence,
            "model_used": "model_b",
        }, resolved_language, language_was_detected)

    # --- Model C ---
    base_model, base_model_used = _resolve_model_a_variant(resolved_language, models)
    if base_model is not None:
        base_proba = base_model.predict_proba([text])[0]
        base_model_used_resolved = base_model_used or "model_a"
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
        base_model_used_resolved = "model_b"

    result = _run_model_c(text, base_proba, base_model_used_resolved, product_category, models)
    return _attach_language_context(result, resolved_language, language_was_detected)
