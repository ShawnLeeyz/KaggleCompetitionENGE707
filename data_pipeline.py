"""Task 2 preprocessing, reusable without executing a notebook.

Call ``create_dataloaders(data_dir=".")`` from the project directory.
For a Kaggle notebook, pass the mounted dataset root instead, for example
``create_dataloaders(data_dir=Path("/kaggle/input/your-dataset"))``.
The returned bundle includes loaders, label mappings and source DataFrames.
No dataset access or random seeding occurs at import time. Images are transformed
on demand; source files are never rewritten.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF, InterpolationMode


SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def calculate_hash(path):
    """Hash the original file bytes, exactly as in Task 2."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class ResizeAndPad:
    def __init__(self, size=224, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        image = image.convert("RGB")
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, min(self.size, round(width * scale)))
        new_height = max(1, min(self.size, round(height * scale)))
        image = TF.resize(
            image,
            [new_height, new_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        dw = self.size - new_width
        dh = self.size - new_height
        image = TF.pad(
            image,
            [dw // 2, dh // 2, dw - dw // 2, dh - dh // 2],
            fill=self.fill,
            padding_mode="constant",
        )
        return image


class FacialFeatureDataset(Dataset):
    def __init__(self, dataframe, transform=None, labelled=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labelled = labelled

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        if self.labelled:
            label = int(row["label"])
            return image, label
        return image, str(row["image_id"])


def build_transforms():
    """Return Task 2's training and shared validation/test transforms."""
    train_transform = transforms.Compose([
        ResizeAndPad(size=224, fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        ResizeAndPad(size=224, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


@dataclass
class DataPipeline:
    """Loaders and metadata; datasets are available as ``loader.dataset``."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    train_df: pd.DataFrame
    train_clean_df: pd.DataFrame
    train_split_df: pd.DataFrame
    val_split_df: pd.DataFrame
    test_df: pd.DataFrame


def create_dataloaders(data_dir="."):
    """Build Task 2's loaders with its unchanged configuration.

    ``data_dir`` accepts a string or pathlib.Path, relative to the notebook's
    working directory or absolute on the current operating system. All input
    paths are joined beneath this root using pathlib.Path; there are no
    machine-specific paths or changes to the working directory. The root must
    contain this same layout on local Windows and Kaggle Linux:

        train_labels.csv
        sample_submission.csv
        train/train/<image files>
        test/test/<image files>

    Duplicate removal precedes the stratified 80/20 split.
    The split uses random_state=42; augmentation and loader shuffling retain
    PyTorch's normal RNG behaviour, with no added seed or generator.

    All loaders use batch_size=32, num_workers=0 and the remaining PyTorch
    defaults (including drop_last=False). Only training is shuffled.
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train" / "train"
    train_df = pd.read_csv(data_dir / "train_labels.csv", dtype={"image_id": str})
    train_map = {
        int(p.stem): p
        for p in train_dir.iterdir()
        if p.is_file() and p.stem.isdecimal()
    }
    train_df["path"] = train_df["image_id"].astype(int).map(train_map)
    if train_df["path"].isna().any():
        raise FileNotFoundError("Some training image IDs have no matching file.")

    train_df["sha256"] = train_df["path"].apply(calculate_hash)
    label_counts = train_df.groupby("sha256")["class"].nunique()
    conflicting_hashes = label_counts[label_counts > 1].index
    if len(conflicting_hashes) > 0:
        raise ValueError(
            "Conflicting duplicate labels found. Review before deduplication."
        )
    train_clean_df = (
        train_df.drop_duplicates(subset="sha256", keep="first")
        .copy()
        .reset_index(drop=True)
    )
    train_split_df, val_split_df = train_test_split(
        train_clean_df,
        test_size=0.20,
        stratify=train_clean_df["class"],
        random_state=SEED,
    )
    train_split_df = train_split_df.reset_index(drop=True)
    val_split_df = val_split_df.reset_index(drop=True)
    assert len(train_split_df) + len(val_split_df) == len(train_clean_df)
    assert set(train_split_df["image_id"]).isdisjoint(val_split_df["image_id"])
    assert set(train_split_df["sha256"]).isdisjoint(val_split_df["sha256"])

    classes = sorted(train_clean_df["class"].unique())
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    train_split_df["label"] = train_split_df["class"].map(class_to_idx)
    val_split_df["label"] = val_split_df["class"].map(class_to_idx)
    assert len(classes) == 8
    assert train_split_df["label"].notna().all()
    assert val_split_df["label"].notna().all()

    train_transform, eval_transform = build_transforms()
    train_dataset = FacialFeatureDataset(train_split_df, transform=train_transform)
    val_dataset = FacialFeatureDataset(val_split_df, transform=eval_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_dir = data_dir / "test" / "test"
    test_df = pd.read_csv(
        data_dir / "sample_submission.csv", dtype={"image_id": str}
    )
    test_files = {int(path.stem): path for path in test_dir.glob("*.jpg")}
    test_df["path"] = test_df["image_id"].apply(
        lambda image_id: test_files.get(int(image_id))
    )
    assert test_df["image_id"].is_unique
    assert test_df["path"].notna().all(), "Some test images are missing!"
    assert test_df["path"].is_unique
    assert len(test_df) == len(test_files)
    test_dataset = FacialFeatureDataset(
        test_df, transform=eval_transform, labelled=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )
    return DataPipeline(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        train_df=train_df,
        train_clean_df=train_clean_df,
        train_split_df=train_split_df,
        val_split_df=val_split_df,
        test_df=test_df,
    )
