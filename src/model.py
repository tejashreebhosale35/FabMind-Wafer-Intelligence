"""
FabMind - Wafer Vision Transformer model.

FabMind-specific changes:
- Accepts single-channel wafer maps: [batch, 1, 224, 224].
- Keeps the ViT Tiny + LayerNorm + Dropout + Linear classification head.
- Reads the existing FabMind YAML model configuration.
- Defaults to CPU when CUDA is unavailable or not requested.
- Includes checkpoint save/load and a model smoke test.

Install:
    pip install timm torch pyyaml
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import timm


class WaferViT(nn.Module):
    """Vision Transformer for single-channel wafer defect classification."""

    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        num_classes: int = 9,
        pretrained: bool = True,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        in_chans: int = 1,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.in_chans = in_chans

        # in_chans=1 is important: FabMind's WaferMapDataset returns
        # [batch, 1, 224, 224], not the 3-channel RGB input used by the
        # original reference implementation.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )

        # Retain the reference implementation's classification head design.
        if hasattr(self.backbone, "head") and isinstance(
            self.backbone.head, nn.Linear
        ):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        elif hasattr(self.backbone, "head") and isinstance(
            self.backbone.head, nn.Sequential
        ):
            # Most timm ViT models expose a linear head. This branch makes
            # the wrapper fail clearly if a different head layout is used.
            in_features = self.backbone.get_classifier().in_features
            self.backbone.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )

        print(f"Model initialized: {model_name}")
        print(f"Input channels: {in_chans}")
        print(f"Number of classes: {num_classes}")
        print(f"Pretrained: {pretrained}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits; do not apply softmax during training."""
        if x.ndim != 4:
            raise ValueError(
                f"Expected input [batch, channels, height, width], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_chans:
            raise ValueError(
                f"Expected {self.in_chans} input channel(s), got {x.shape[1]}"
            )
        return self.backbone(x)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def get_trainable_params(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def resolve_device(requested_device: Optional[str] = None) -> torch.device:
    """Resolve the requested device safely, defaulting to CPU."""
    if requested_device:
        requested = requested_device.lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA was requested but is unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(
    config: Dict[str, Any],
    device: Optional[str] = None,
) -> WaferViT:
    """Create FabMind's wafer classifier from config.yaml."""
    model_config = config["model"]

    selected_device = resolve_device(
        device or config.get("device", "cpu")
    )

    model = WaferViT(
        model_name=model_config.get("name", "vit_tiny_patch16_224"),
        num_classes=int(model_config.get("num_classes", 9)),
        pretrained=bool(model_config.get("pretrained", True)),
        dropout=float(model_config.get("dropout", 0.1)),
        drop_path_rate=float(model_config.get("drop_path_rate", 0.1)),
        in_chans=int(model_config.get("in_chans", 1)),
    ).to(selected_device)

    total_params = model.get_num_params()
    trainable_params = model.get_trainable_params()

    print("\nModel statistics")
    print(f"Device: {selected_device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Approx. parameter size (FP32): {total_params * 4 / (1024 ** 2):.2f} MB")

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_accuracy: float,
    checkpoint_path: str,
    is_best: bool = False,
) -> None:
    """Save a training checkpoint and optionally a separate best checkpoint."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_accuracy": best_accuracy,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

    if is_best:
        best_path = path.with_name(f"{path.stem}_best{path.suffix}")
        torch.save(checkpoint, best_path)
        print(f"Best checkpoint saved: {best_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[nn.Module, int, float]:
    """Load a FabMind checkpoint. Only load checkpoints you trust."""
    selected_device = resolve_device(device)
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    best_accuracy = float(
        checkpoint.get("best_accuracy", checkpoint.get("best_acc", 0.0))
    )

    print(
        f"Loaded checkpoint from epoch {epoch}; "
        f"best accuracy = {best_accuracy:.4f}"
    )
    return model, epoch, best_accuracy


def main() -> None:
    """Build the model and verify its output shape using a dummy wafer batch."""
    import yaml

    # src/model.py -> FabMind project root
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Expected config/config.yaml under the FabMind project root."
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    device = resolve_device(config.get("device", "cpu"))
    print(f"Using device: {device}")

    model = create_model(config, str(device))
    model.eval()

    image_size = int(config.get("data", {}).get("image_size", 224))
    in_chans = int(config.get("model", {}).get("in_chans", 1))

    dummy_input = torch.randn(2, in_chans, image_size, image_size).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    expected_classes = int(config["model"].get("num_classes", 9))
    print(f"\nInput shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")

    if output.shape != (2, expected_classes):
        raise RuntimeError(
            f"Unexpected output shape {tuple(output.shape)}; "
            f"expected (2, {expected_classes})."
        )

    print("FabMind WaferViT model verification passed.")


if __name__ == "__main__":
    main()
