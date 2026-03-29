import pandas as pd
import numpy as np
import os
from datasets import Dataset
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


QUICK_MODE = getenv_bool("MODEL_B_QUICK_MODE", False)
TEXT_SOURCE = os.getenv("MODEL_B_TEXT_SOURCE", "title_body").strip().lower()
MAX_LENGTH = int(os.getenv("MODEL_B_MAX_LENGTH", "128"))
NUM_EPOCHS = int(os.getenv("MODEL_B_NUM_EPOCHS", "4"))
TRAIN_BATCH_SIZE = int(os.getenv("MODEL_B_TRAIN_BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(os.getenv("MODEL_B_EVAL_BATCH_SIZE", "8"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("MODEL_B_GRAD_ACC_STEPS", "2"))
TRAIN_SAMPLE_CAP = int(os.getenv("MODEL_B_TRAIN_SAMPLE_CAP", "60000"))
OUTPUT_DIR = os.getenv("MODEL_B_OUTPUT_DIR", "models/saved/model_b")


def build_model_text(df: pd.DataFrame) -> pd.Series:
    title_fallback = df["review_body"].astype(str).str.split().str[:5].str.join(" ") + "..."
    review_title = df["review_title"].fillna(title_fallback).astype(str)
    review_body = df["review_body"].astype(str)
    return (review_title + " " + review_body).str.strip()


def build_text(df: pd.DataFrame) -> pd.Series:
    if TEXT_SOURCE == "review_body":
        return df["review_body"].astype(str)
    return build_model_text(df)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mapped_predictions = predictions + 1
    mapped_labels = labels + 1
    return {
        "accuracy": accuracy_score(labels, predictions),
        "mae": mean_absolute_error(mapped_labels, mapped_predictions),
    }


def stratified_cap(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap is None or len(df) <= cap:
        return df

    stratify_key = df["language"].astype(str) + "_" + df["stars"].astype(int).astype(str)
    train_index, _ = train_test_split(
        df.index,
        train_size=cap,
        random_state=42,
        stratify=stratify_key,
    )
    return df.loc[train_index].copy()


def main():
    train_raw = pd.read_csv("data/train.csv")
    val_raw = pd.read_csv("data/validation.csv")
    test_raw = pd.read_csv("data/test.csv")

    train_df = pd.DataFrame(
        {
            "text": build_text(train_raw).values,
            "labels": (train_raw["stars"].astype(int) - 1).values,
        }
    )
    val_df = pd.DataFrame(
        {
            "text": build_text(val_raw).values,
            "labels": (val_raw["stars"].astype(int) - 1).values,
        }
    )
    test_df = pd.DataFrame(
        {
            "text": build_text(test_raw).values,
            "labels": (test_raw["stars"].astype(int) - 1).values,
        }
    )

    if QUICK_MODE:
        train_df = stratified_cap(train_raw, min(30000, len(train_raw)))
        train_df = pd.DataFrame(
            {
                "text": build_text(train_df).values,
                "labels": (train_df["stars"].astype(int) - 1).values,
            }
        )
        val_df = val_df.sample(n=min(8000, len(val_df)), random_state=42)
        test_df = test_df.sample(n=min(8000, len(test_df)), random_state=42)
    elif TRAIN_SAMPLE_CAP is not None:
        train_raw = stratified_cap(train_raw, min(TRAIN_SAMPLE_CAP, len(train_raw)))
        train_df = pd.DataFrame(
            {
                "text": build_text(train_raw).values,
                "labels": (train_raw["stars"].astype(int) - 1).values,
            }
        )

    effective_batch = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS if not QUICK_MODE else 32
    estimated_steps = int(np.ceil((len(train_df) * (1 if QUICK_MODE else NUM_EPOCHS)) / effective_batch))
    print(
        "Config:",
        {
            "quick_mode": QUICK_MODE,
            "text_source": TEXT_SOURCE,
            "max_length": 64 if QUICK_MODE else MAX_LENGTH,
            "num_epochs": 1 if QUICK_MODE else NUM_EPOCHS,
            "train_batch_size": 32 if QUICK_MODE else TRAIN_BATCH_SIZE,
            "eval_batch_size": 32 if QUICK_MODE else EVAL_BATCH_SIZE,
            "grad_accumulation_steps": 1 if QUICK_MODE else GRADIENT_ACCUMULATION_STEPS,
            "output_dir": OUTPUT_DIR,
        },
    )
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")
    print(f"Epochs: {1 if QUICK_MODE else NUM_EPOCHS} | Effective batch: {effective_batch} | Estimated optimizer steps: {estimated_steps}")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    max_length = 64 if QUICK_MODE else MAX_LENGTH

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding=False,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    val_dataset = val_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=5,
    )

    # Set no_cuda=True if no GPU is available.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1 if QUICK_MODE else NUM_EPOCHS,
        per_device_train_batch_size=32 if QUICK_MODE else TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=32 if QUICK_MODE else EVAL_BATCH_SIZE,
        gradient_accumulation_steps=1 if QUICK_MODE else GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_mae",
        greater_is_better=False,
        fp16=not QUICK_MODE,
        gradient_checkpointing=not QUICK_MODE,
        save_total_limit=2,
        report_to="none",
        seed=42,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Validation metrics:", trainer.evaluate(eval_dataset=val_dataset))
    print("Test metrics:", trainer.evaluate(eval_dataset=test_dataset))
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
