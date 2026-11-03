"""
FabMind - Wafer Data Pipeline

Loads prepared WM-811K train/validation/test splits, validates them,
preprocesses wafer maps, and creates PyTorch DataLoaders.
It does not load the original LSWMD.pkl file or train a model.
"""

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WaferDataProcessor:
    """Loads, validates, and summarizes the prepared dataset splits."""

    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config["data"]
        # Assumes: FabMind/ml/preprocessing/wafer_data.py
        self.project_root = Path(__file__).resolve().parent.parent
        self.processed_dir = self.project_root / self.data_config["processed_dir"]

        self.wafer_column = self.data_config.get("wafer_column", "waferMap")
        self.label_column = self.data_config.get("label_column", "failureType")
        self.classes = self.data_config["classes"]
        self.num_classes = int(self.data_config["num_classes"])

        if len(self.classes) != self.num_classes:
            raise ValueError(
                f"Configured {len(self.classes)} classes, but "
                f"num_classes is {self.num_classes}."
            )
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Configured class names must be unique.")

        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.classes)
        }

    def load_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the existing train, validation, and test pickle files."""
        if not self.data_config.get("use_existing_split", False):
            raise ValueError(
                "This module expects existing splits. Run prepare_dataset.py first."
            )

        filenames = {
            "train": self.data_config["train_path"],
            "validation": self.data_config["validation_path"],
            "test": self.data_config["test_path"],
        }
        loaded = {}

        for split_name, filename in filenames.items():
            file_path = self.processed_dir / filename
            if not file_path.is_file():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")

            print(f"Loading {split_name}: {file_path}")
            dataframe = pd.read_pickle(file_path)

            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(f"{filename} must contain a pandas DataFrame.")

            required = [self.wafer_column, self.label_column]
            missing = [column for column in required if column not in dataframe]
            if missing:
                raise ValueError(f"{filename} is missing columns: {missing}")

            loaded[split_name] = dataframe
            print(f"{split_name}: {len(dataframe):,} records")

        return loaded["train"], loaded["validation"], loaded["test"]

    @staticmethod
    def clean_label(value) -> Optional[str]:
        """
        Unwrap WM-811K nested labels and normalize their representation.
        A literal None represents the normal class; NaN/empty values are missing.
        """
        while isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return None
            value = value[0]

        if value is None:
            return "none"
        if value is pd.NA:
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        label = str(value).strip()
        if label.lower() in {"none", "none found"}:
            return "none"
        if label.lower() in {"nan", ""}:
            return None
        return label

    def encode_label(self, value) -> int:
        """Convert a defect label to its integer class ID."""
        label = self.clean_label(value)
        if label is None:
            raise ValueError("A missing defect label was found.")
        if label not in self.class_to_idx:
            raise ValueError(
                f"Unknown defect label {label!r}. Expected: {self.classes}"
            )
        return self.class_to_idx[label]

    def get_class_distribution(self, dataframe: pd.DataFrame) -> pd.Series:
        """Count records in each cleaned defect class."""
        labels = dataframe[self.label_column].apply(self.clean_label)
        return labels.value_counts(dropna=False)

    def validate_dataframe(
        self, dataframe: pd.DataFrame, split_name: str
    ) -> None:
        """Check that every row has a valid label and a non-empty 2D wafer map."""
        if dataframe.empty:
            raise ValueError(f"{split_name} dataset is empty.")

        for row_index, row in dataframe.iterrows():
            self.encode_label(row[self.label_column])
            wafer_map = row[self.wafer_column]
            if wafer_map is None:
                raise ValueError(
                    f"Missing wafer map in {split_name}, row {row_index}."
                )
            wafer_array = np.asarray(wafer_map)
            if wafer_array.ndim != 2 or wafer_array.size == 0:
                raise ValueError(
                    f"Invalid wafer map in {split_name}, row {row_index}: "
                    f"shape={wafer_array.shape}"
                )

        print(f"{split_name} validation passed.")

    def compute_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        """
        Calculate inverse-frequency weights from training labels.
        Use the returned tensor in torch.nn.CrossEntropyLoss(weight=...).
        """
        labels = train_df[self.label_column].apply(self.encode_label)
        counts = np.bincount(
            labels.to_numpy(dtype=np.int64), minlength=self.num_classes
        )

        missing = [
            self.classes[i] for i, count in enumerate(counts) if count == 0
        ]
        if missing:
            raise ValueError(
                f"Training split has no samples for: {missing}. "
                "Review the dataset split."
            )

        weights = len(labels) / (self.num_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)


def prepare_wafer_map(wafer_map: np.ndarray, config: Dict) -> np.ndarray:
    """
    Resize a categorical wafer map to image_size x image_size.
    Nearest-neighbour interpolation preserves discrete values 0, 1, and 2.
    """
    image_size = int(config["data"]["image_size"])
    preprocessing = config["preprocessing"]
    wafer_map = np.asarray(wafer_map, dtype=np.float32)

    if wafer_map.ndim != 2 or wafer_map.size == 0:
        raise ValueError(f"Expected a non-empty 2D map; got {wafer_map.shape}.")

    wafer_map = np.nan_to_num(
        wafer_map, nan=0.0, posinf=0.0, neginf=0.0
    )
    unique_values = np.unique(wafer_map)
    if not np.all(np.isin(unique_values, [0, 1, 2])):
        raise ValueError(f"Unexpected wafer-map values: {unique_values}")

    height, width = wafer_map.shape
    scale = min(image_size / height, image_size / width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    resized = cv2.resize(
        wafer_map,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )

    padding_mode = preprocessing.get("padding_mode", "constant").lower()
    padding_value = float(preprocessing.get("padding_value", 0))
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT,
        "reflect_101": cv2.BORDER_REFLECT_101,
    }
    if padding_mode not in border_modes:
        raise ValueError(
            f"Unsupported padding_mode {padding_mode!r}; "
            f"choose from {list(border_modes)}."
        )

    top = (image_size - new_height) // 2
    bottom = image_size - new_height - top
    left = (image_size - new_width) // 2
    right = image_size - new_width - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        border_modes[padding_mode], value=padding_value
    )
    return padded.astype(np.float32) / 2.0


class WaferAugmentation:
    """
    Optional geometric augmentation.
    Brightness/contrast changes are omitted because map values are categories,
    not ordinary image intensities.
    """

    def __init__(self, config: Dict):
        self.config = config.get("augmentation", {})

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        max_rotation = float(self.config.get("random_rotation", 0))
        if max_rotation > 0:
            angle = random.uniform(-max_rotation, max_rotation)
            tensor = TF.rotate(
                tensor,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        if self.config.get("random_flip_horizontal", False):
            if random.random() < 0.5:
                tensor = TF.hflip(tensor)

        if self.config.get("random_flip_vertical", False):
            if random.random() < 0.5:
                tensor = TF.vflip(tensor)

        return tensor


class WaferMapDataset(Dataset):
    """Provide one preprocessed wafer tensor and its class ID per index."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        processor: WaferDataProcessor,
        config: Dict,
        training: bool = False,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.processor = processor
        self.config = config
        self.training = training
        self.augmentation = WaferAugmentation(config)

        preprocessing = config["preprocessing"]
        self.normalize = preprocessing.get("normalize", True)

        if self.normalize:
            mean = preprocessing.get("mean", [0.5])
            std = preprocessing.get("std", [0.5])
            if len(mean) != 1 or len(std) != 1:
                raise ValueError(
                    "Single-channel wafer maps require one mean and one std."
                )
            if std[0] == 0:
                raise ValueError("Normalization std must not be zero.")
            self.normalization = transforms.Normalize(mean=mean, std=std)

    def __len__(self) -> int:
        """Return the number of examples in this split."""
        return len(self.dataframe)

    def __getitem__(self, index: int):
        """Fetch, preprocess, and return one image tensor and label tensor."""
        row = self.dataframe.iloc[index]
        wafer_array = prepare_wafer_map(
            row[self.processor.wafer_column], self.config
        )

        # Add channel dimension: [H, W] becomes [1, H, W].
        wafer_tensor = torch.from_numpy(wafer_array).unsqueeze(0)

        if self.training:
            wafer_tensor = self.augmentation(wafer_tensor)
        if self.normalize:
            wafer_tensor = self.normalization(wafer_tensor)

        label = self.processor.encode_label(
            row[self.processor.label_column]
        )
        return wafer_tensor, torch.tensor(label, dtype=torch.long)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processor: WaferDataProcessor,
    config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for all three dataset splits."""
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config.get("num_workers", 0))
    if batch_size < 1:
        raise ValueError("training.batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    pin_memory = (
        bool(config.get("pin_memory", False))
        and torch.cuda.is_available()
    )

    train_dataset = WaferMapDataset(
        train_df, processor, config, training=True
    )
    val_dataset = WaferMapDataset(
        val_df, processor, config, training=False
    )
    test_dataset = WaferMapDataset(
        test_df, processor, config, training=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader


def main() -> None:
    """Run a basic end-to-end check of the data pipeline."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    set_seed(int(config.get("seed", 42)))
    processor = WaferDataProcessor(config)
    train_df, val_df, test_df = processor.load_data()

    processor.validate_dataframe(train_df, "Training")
    processor.validate_dataframe(val_df, "Validation")
    processor.validate_dataframe(test_df, "Testing")

    print("\nTraining class distribution:")
    print(processor.get_class_distribution(train_df))

    class_weight_config = config.get("class_weights", {})
    if (
        class_weight_config.get("use_weights", False)
        and class_weight_config.get("compute_from_data", False)
    ):
        class_weights = processor.compute_class_weights(train_df)
        print("\nClass weights:")
        for class_name, weight in zip(processor.classes, class_weights):
            print(f"{class_name}: {weight.item():.4f}")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, processor, config
    )

    print("\nDataLoader information:")
    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Testing batches:    {len(test_loader)}")

    wafer_batch, label_batch = next(iter(train_loader))
    print("\nBatch verification:")
    print(f"Wafer batch shape: {tuple(wafer_batch.shape)}")
    print(f"Label batch shape: {tuple(label_batch.shape)}")
    print(f"First label IDs: {label_batch[:10].tolist()}")
    print("\nData pipeline verification completed.")


if __name__ == "__main__":
    main()
