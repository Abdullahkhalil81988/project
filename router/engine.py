import numpy as np
import pandas as pd
import torch


MODEL_B_MAX_LENGTH = 128
MODEL_A_ESCALATION_THRESHOLD = 0.55


def select_model(language: str, text_length: int, product_category: str) -> str:
    if product_category in ["book", "digital_ebook_purchase", "electronics", "pc"]:
        return "model_c"
    if language == "en" and 15<=text_length < 80:
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
        if predicted_stars >= 4:
            sentiment = "positive"
        elif predicted_stars <= 2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "predicted_stars": predicted_stars,
            "sentiment": sentiment,
            "confidence": confidence,
            "model_used": "model_b",
        }

    raise ValueError("Unsupported model")


def run_inference(review_body: str, language: str, product_category: str, models: dict, review_title: str | None = None) -> dict:
    prepared = preprocess_incoming_review(review_body, review_title)
    text_length = prepared["text_length"]
    selected = select_model(language, text_length, product_category)

    text = prepared["model_text"]

    if selected == "model_a":
        model_a = models["model_a"]
        proba = model_a.predict_proba([text])[0]
        predicted_stars = int(np.argmax(proba) + 1)
        confidence = float(np.max(proba))

        result = {
            "predicted_stars": predicted_stars,
            "sentiment": "positive" if predicted_stars >= 4 else ("negative" if predicted_stars <= 2 else "neutral"),
            "confidence": confidence,
            "model_used": selected,
        }
        model_name = selected
        if model_name == "model_a" and result["confidence"] < MODEL_A_ESCALATION_THRESHOLD:
            result = run_model("model_b", text, models)
            result["model_used"] = "model_b_escalated"
        return result
    elif selected == "model_b":
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
    else:
        model_a = models["model_a"]
        tokenizer_b = models["model_b_tokenizer"]
        model_b = models["model_b"]
        model_c = models["model_c"]
        model_c_categories = models["model_c_categories"]

        if language == "en":
            base_proba = model_a.predict_proba([text])[0]
            base_model_used = "model_a"
        else:
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
                base_proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
            base_model_used = "model_b"

        category_df = pd.get_dummies(pd.Series([product_category]), prefix="category")
        category_df = category_df.reindex(columns=model_c_categories, fill_value=0)

        feature_row = pd.DataFrame([base_proba], columns=[f"prob_{i}" for i in range(1, 6)])
        X_meta = pd.concat([feature_row, category_df.reset_index(drop=True)], axis=1)

        meta_proba = model_c.predict_proba(X_meta)[0]
        predicted_stars = int(model_c.predict(X_meta)[0])
        confidence = float(np.max(meta_proba))

    if predicted_stars >= 4:
        sentiment = "positive"
    elif predicted_stars <= 2:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    result = {
        "predicted_stars": predicted_stars,
        "sentiment": sentiment,
        "confidence": confidence,
        "model_used": selected,
    }
    
    # Add base model info when Model C is the router
    if selected == "model_c":
        result["base_model_used"] = base_model_used
    
    return result
