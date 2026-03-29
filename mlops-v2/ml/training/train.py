import argparse
import logging
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parents[2]))
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def generate_synthetic_data(n_samples: int = 2000):
    positive = [
        "This is absolutely wonderful and amazing.",
        "I love this product so much, highly recommended!",
        "Fantastic experience, exceeded all expectations.",
        "Brilliant work, really impressed with the quality.",
        "Outstanding performance, could not be happier.",
        "The best I have ever seen, truly remarkable.",
        "Superb quality and excellent customer service.",
        "Exceptional value for money, will buy again.",
    ]
    negative = [
        "This is terrible and completely disappointing.",
        "Awful experience, would not recommend to anyone.",
        "Poor quality and very bad customer service.",
        "Horrible product, broke within a week of use.",
        "Worst purchase I have ever made, total waste.",
        "Extremely disappointing and overpriced product.",
        "Very poor performance, nothing worked properly.",
        "Dreadful experience from start to finish.",
    ]
    rng = np.random.default_rng(42)
    texts, labels = [], []
    for _ in range(n_samples // 2):
        texts.append(rng.choice(positive))
        labels.append(1)
    for _ in range(n_samples // 2):
        texts.append(rng.choice(negative))
        labels.append(0)
    idx = rng.permutation(len(texts))
    return [texts[i] for i in idx], [labels[i] for i in idx]


def train(epochs=3, batch_size=16, learning_rate=2e-5, max_samples=2000, auto_promote=False, experiment_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name or settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params({
            "model_base": settings.HF_MODEL_NAME,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_samples": max_samples,
            "device": str(device),
        })
        mlflow.set_tags({"task": "sentiment-classification", "framework": "pytorch+transformers"})

        texts, labels = generate_synthetic_data(max_samples)
        split = int(0.8 * len(texts))

        tokenizer = AutoTokenizer.from_pretrained(settings.HF_MODEL_NAME, cache_dir="/tmp/hf_cache")
        train_ds = SentimentDataset(texts[:split], labels[:split], tokenizer)
        val_ds   = SentimentDataset(texts[split:], labels[split:], tokenizer)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        mlflow.log_params({"train_samples": len(train_ds), "val_samples": len(val_ds)})

        model = AutoModelForSequenceClassification.from_pretrained(
            settings.HF_MODEL_NAME, num_labels=2, cache_dir="/tmp/hf_cache"
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        best_val_acc = 0.0
        best_model_path = "/tmp/best_model"

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_losses.append(outputs.loss.item())

            model.eval()
            val_losses, all_preds, all_labels = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    val_losses.append(outputs.loss.item())
                    all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                    all_labels.extend(batch["labels"].numpy())

            val_acc = accuracy_score(all_labels, all_preds)
            val_f1  = f1_score(all_labels, all_preds, average="weighted")
            mlflow.log_metrics({
                f"train_loss_epoch_{epoch}": round(float(np.mean(train_losses)), 4),
                f"val_loss_epoch_{epoch}":   round(float(np.mean(val_losses)), 4),
                f"val_accuracy_epoch_{epoch}": round(val_acc, 4),
                f"val_f1_epoch_{epoch}":       round(val_f1, 4),
            }, step=epoch)
            logger.info(f"Epoch {epoch}/{epochs} | train_loss={np.mean(train_losses):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)

        mlflow.log_metrics({"best_val_accuracy": round(best_val_acc, 4), "final_val_f1": round(val_f1, 4)})
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=settings.MLFLOW_MODEL_NAME,
        )

        if auto_promote:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{settings.MLFLOW_MODEL_NAME}'")
            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)
            if latest:
                client.transition_model_version_stage(
                    name=settings.MLFLOW_MODEL_NAME,
                    version=latest[0].version,
                    stage="Staging",
                )
                logger.info(f"Model v{latest[0].version} promoted to Staging")

        return {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "model_uri": model_info.model_uri,
            "best_val_accuracy": best_val_acc,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--auto-promote", action="store_true")
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()
    result = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        auto_promote=args.auto_promote,
        experiment_name=args.experiment_name,
    )
    logger.info(f"Done: {result}")
