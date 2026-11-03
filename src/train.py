"""FabMind CPU-friendly training script connecting wafer_data.py and model.py."""
from pathlib import Path
import json
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wafer_data import WaferDataProcessor, create_dataloaders
from model import create_model, save_checkpoint


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(config: Dict) -> torch.device:
    requested = str(config.get("device", "cpu")).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested.startswith("cuda"):
        print("CUDA unavailable; using CPU.")
    return torch.device("cpu")


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("Expected a batch containing images and labels.")
    return batch[0], batch[1]


def get_class_names(processor, train_df, num_classes: int) -> List[str]:
    candidates = [
        getattr(processor, "class_names", None),
        getattr(processor, "classes", None),
    ]
    encoder = getattr(processor, "label_encoder", None)
    if encoder is not None:
        candidates.append(getattr(encoder, "classes_", None))

    for values in candidates:
        if values is not None and len(values) == num_classes:
            return [str(value) for value in values]

    if "failureType" in train_df.columns:
        values = sorted(str(v) for v in train_df["failureType"].dropna().unique())
        if len(values) == num_classes:
            print("Note: inferred sorted class names. Confirm this matches the label encoder order.")
            return values
    return [f"class_{i}" for i in range(num_classes)]


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int):
        self.total += float(value) * count
        self.count += count

    @property
    def average(self):
        return self.total / max(self.count, 1)


class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = get_device(config)
        set_random_seed(int(config.get("seed", 42)))
        self.num_classes = int(config["model"].get("num_classes", 9))

        self.output_dir = PROJECT_ROOT / "outputs"
        self.checkpoint_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training device: {self.device}")
        self._setup_data()
        self.model = create_model(config, str(self.device))
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self._setup_optimizer()

    def _setup_data(self):
        print("\\nLoading prepared dataset splits...")
        self.processor = WaferDataProcessor(self.config)
        loaded = self.processor.load_data()
        if not isinstance(loaded, (tuple, list)) or len(loaded) != 3:
            raise ValueError("load_data() must return train_df, val_df, test_df.")
        self.train_df, self.val_df, self.test_df = loaded

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.train_df,
            self.val_df,
            self.test_df,
            self.processor,
            self.config,
        )
        self.class_names = get_class_names(
            self.processor, self.train_df, self.num_classes
        )
        print(f"Train records: {len(self.train_df):,}")
        print(f"Validation records: {len(self.val_df):,}")
        print(f"Test records: {len(self.test_df):,}")
        print(f"Class names/order: {self.class_names}")

        if self.config.get("class_weights", {}).get("use_weights", True):
            self.class_weights = self.processor.compute_class_weights(
                self.train_df
            ).to(self.device).float()
            if len(self.class_weights) != self.num_classes:
                raise ValueError("Number of class weights does not match num_classes.")
            print("Using class-weighted CrossEntropyLoss.")
        else:
            self.class_weights = None

    def _setup_optimizer(self):
        cfg = self.config["training"]
        name = cfg.get("optimizer", "adamw").lower()
        lr = float(cfg.get("learning_rate", 1e-4))
        decay = float(cfg.get("weight_decay", 1e-4))

        if name == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=decay)
        elif name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        else:
            raise ValueError("Supported optimizers are adamw and adam.")

        scheduler_name = cfg.get("scheduler", "cosine").lower()
        epochs = int(cfg.get("num_epochs", 5))
        if scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(epochs, 1)
            )
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.1
            )
        else:
            self.scheduler = None

        self.gradient_clip = float(cfg.get("gradient_clip", 1.0))
        self.patience = int(cfg.get("early_stopping_patience", 5))
        print(f"Optimizer: {name}; learning rate: {lr}; scheduler: {scheduler_name}")
        print("Mixed precision disabled for this CPU-first run.")

    def _run_epoch(self, loader, training: bool, epoch: int):
        self.model.train(training)
        loss_meter = AverageMeter()
        targets, predictions = [], []
        description = f"Train {epoch + 1}" if training else "Validation"
        progress = tqdm(loader, desc=description)

        for batch in progress:
            images, labels = unpack_batch(batch)
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if training:
                    loss.backward()
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()

            loss_meter.update(loss.item(), images.size(0))
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
            progress.set_postfix(loss=f"{loss_meter.average:.4f}")

        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, labels=list(range(self.num_classes)),
            average="macro", zero_division=0,
        )
        return {
            "loss": loss_meter.average, "accuracy": float(accuracy),
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "targets": targets, "predictions": predictions,
        }

    def train(self):
        cfg = self.config["training"]
        epochs = int(cfg.get("num_epochs", 5))
        best_accuracy = -1.0
        best_epoch = 0
        no_improvement = 0
        history = []

        print("\\nStarting FabMind training...")
        for epoch in range(epochs):
            print(f"\\n{'=' * 50}\\nEpoch {epoch + 1}/{epochs}\\n{'=' * 50}")
            train_metrics = self._run_epoch(self.train_loader, True, epoch)
            val_metrics = self._run_epoch(self.val_loader, False, epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            lr_now = self.optimizer.param_groups[0]["lr"]
            print(
                f"Train loss={train_metrics['loss']:.4f}, "
                f"accuracy={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}"
            )
            print(
                f"Val   loss={val_metrics['loss']:.4f}, "
                f"accuracy={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}"
            )
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "learning_rate": lr_now,
            })

            improved = val_metrics["accuracy"] > best_accuracy
            if improved:
                best_accuracy = val_metrics["accuracy"]
                best_epoch = epoch + 1
                no_improvement = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, best_accuracy,
                    str(self.checkpoint_dir / "wafer_vit_checkpoint.pth"),
                    is_best=True,
                )
            else:
                no_improvement += 1

            if no_improvement >= self.patience:
                print(f"Early stopping after {self.patience} epochs without improvement.")
                break

        history_path = self.results_dir / "training_history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Training complete. Best validation accuracy={best_accuracy:.4f} at epoch {best_epoch}.")
        self.evaluate_best_model()

    @torch.no_grad()
    def evaluate_best_model(self):
        best_path = self.checkpoint_dir / "wafer_vit_checkpoint_best.pth"
        if not best_path.exists():
            print("Best checkpoint not found; skipping test evaluation.")
            return

        checkpoint = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        metrics = self._run_epoch(self.test_loader, False, 0)

        report = classification_report(
            metrics["targets"], metrics["predictions"],
            labels=list(range(self.num_classes)),
            target_names=self.class_names, zero_division=0, output_dict=True,
        )
        matrix = confusion_matrix(
            metrics["targets"], metrics["predictions"],
            labels=list(range(self.num_classes)),
        )
        results = {
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
            "macro_precision": metrics["precision"],
            "macro_recall": metrics["recall"],
            "macro_f1": metrics["f1"],
            "class_names": self.class_names,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
        }
        results_path = self.results_dir / "test_metrics.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        print("\\nTest results:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f} (macro)")
        print(f"Recall:    {metrics['recall']:.4f} (macro)")
        print(f"F1-score:  {metrics['f1']:.4f} (macro)")
        print(f"Detailed results saved to: {results_path}")


def main():
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    Trainer(config).train()


if __name__ == "__main__":
    main()
