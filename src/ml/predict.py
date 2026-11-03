
"""
FabMind - Wafer Defect Prediction Service

Loads the trained WaferViT checkpoint and predicts defect classes
using exactly the same preprocessing as the training pipeline.
"""

from pathlib import Path
import copy
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms

# Make the src directory importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import create_model, load_checkpoint, resolve_device
from wafer_data import WaferDataProcessor, prepare_wafer_map


class WaferPredictor:
    """Predict wafer defect classes using the trained FabMind model."""

    def __init__(self, config_path=None, checkpoint_path=None):

        self.project_root = PROJECT_ROOT

        self.config_path = (
            Path(config_path)
            if config_path
            else self.project_root / "config" / "config.yaml"
        )

        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path
            else self.project_root
            / "outputs"
            / "models"
            / "wafer_vit_checkpoint_best.pth"
        )

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Configuration not found: {self.config_path}"
            )

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        # Load configuration.
        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        # Reuse the existing class mapping.
        self.processor = WaferDataProcessor(self.config)
        self.class_names = self.processor.classes

        if len(self.class_names) != int(
            self.config["model"]["num_classes"]
        ):
            raise ValueError("Configured class count does not match model.")

        # Use CPU unless CUDA is explicitly available and configured.
        self.device = resolve_device(
            self.config.get("device", "cpu")
        )

        # Do not download pretrained weights during inference.
        inference_config = copy.deepcopy(self.config)
        inference_config["model"]["pretrained"] = False

        self.model = create_model(
            inference_config,
            str(self.device)
        )

        self.model, _, _ = load_checkpoint(
            self.model,
            str(self.checkpoint_path),
            device=str(self.device)
        )

        self.model.eval()

        # Use the exact normalization used during training.
        preprocessing = self.config["preprocessing"]

        if preprocessing.get("normalize", True):
            mean = preprocessing.get("mean", [0.5])
            std = preprocessing.get("std", [0.5])

            if len(mean) != 1 or len(std) != 1 or std[0] == 0:
                raise ValueError("Invalid normalization configuration.")

            self.normalization = transforms.Normalize(
                mean=mean,
                std=std
            )
        else:
            self.normalization = None

        print("\nFabMind prediction service initialized.")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")

    def preprocess(self, wafer_map):
        """Convert a raw 2D wafer map into a model-ready tensor."""

        wafer_array = np.asarray(wafer_map)

        # Reuse the training preprocessing function.
        processed_map = prepare_wafer_map(
            wafer_array,
            self.config
        )

        # Add single-channel dimension: [H, W] -> [1, H, W].
        wafer_tensor = torch.from_numpy(
            processed_map
        ).unsqueeze(0).float()

        # Apply the same normalization as training.
        if self.normalization is not None:
            wafer_tensor = self.normalization(wafer_tensor)

        # Add batch dimension: [1, H, W] -> [1, 1, H, W].
        wafer_tensor = wafer_tensor.unsqueeze(0)

        return wafer_tensor.to(self.device)

    @torch.no_grad()
    def predict_wafer(self, wafer_map, top_k=3):
        """Predict the defect class and return class probabilities."""

        wafer_tensor = self.preprocess(wafer_map)

        # Forward pass.
        logits = self.model(wafer_tensor)

        # Convert logits into probabilities.
        probabilities = F.softmax(logits, dim=1)[0]

        predicted_id = int(torch.argmax(probabilities).item())

        predicted_class = self.class_names[predicted_id]
        predicted_score = float(probabilities[predicted_id].item())

        # Probabilities for all nine classes.
        class_probabilities = {
            class_name: round(float(probabilities[index].item()), 6)
            for index, class_name in enumerate(self.class_names)
        }

        # Return the top predictions.
        top_k = min(top_k, len(self.class_names))

        top_values, top_indices = torch.topk(
            probabilities,
            k=top_k
        )

        top_predictions = []

        for score, class_id in zip(
            top_values.tolist(),
            top_indices.tolist()
        ):
            top_predictions.append({
                "class": self.class_names[class_id],
                "score": round(float(score), 6)
            })

        return {
            "predicted_class": predicted_class,
            "class_id": predicted_id,
            "score": round(predicted_score, 6),
            "class_probabilities": class_probabilities,
            "top_predictions": top_predictions,
            "model": "WaferViT",
            "status": "prediction_completed"
        }


def main():
    """Smoke test using one actual wafer from the test split."""

    predictor = WaferPredictor()

    # Load the existing test split.
    test_path = (
        predictor.processor.processed_dir
        / predictor.config["data"]["test_path"]
    )

    import pandas as pd

    test_df = pd.read_pickle(test_path)

    # Select one actual wafer.
    sample = test_df.iloc[0]

    wafer_map = sample[predictor.processor.wafer_column]

    actual_label = predictor.processor.clean_label(
        sample[predictor.processor.label_column]
    )

    result = predictor.predict_wafer(wafer_map)

    print("\n========== FabMind Prediction Test ==========")
    print(f"Actual defect:    {actual_label}")
    print(f"Predicted defect: {result['predicted_class']}")
    print(f"Prediction score: {result['score']:.4f}")

    print("\nTop predictions:")

    for item in result["top_predictions"]:
        print(
            f"{item['class']:<15} "
            f"{item['score']:.4f}"
        )

    print("\nAll class probabilities:")

    for class_name, probability in (
        result["class_probabilities"].items()
    ):
        print(f"{class_name:<15} {probability:.4f}")

    print("\nPrediction service verification completed.")


if __name__ == "__main__":
    main()
