import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

IMAGE_SIZE = 128
RANDOM_SEED = 42
UNKNOWN_THRESHOLD = 0.65


class PalmDataset(Dataset):
    def __init__(self, image_paths, labels, label_to_index, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.label_to_index = label_to_index
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return image

    def _augment(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
        if np.random.rand() < 0.4:
            angle = np.random.uniform(-10, 10)
            mat = cv2.getRotationMatrix2D((IMAGE_SIZE / 2, IMAGE_SIZE / 2), angle, 1.0)
            image = cv2.warpAffine(
                image,
                mat,
                (IMAGE_SIZE, IMAGE_SIZE),
                borderMode=cv2.BORDER_REFLECT
            )
        return image

    def __getitem__(self, index):
        image = self._load_image(self.image_paths[index])
        if self.augment:
            image = self._augment(image)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        label = self.label_to_index[self.labels[index]]
        return torch.tensor(image), label


class PalmCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def preprocess_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image = extract_palm_roi(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def extract_palm_roi(gray: np.ndarray) -> np.ndarray:
    if len(gray.shape) != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    pad = int(0.05 * max(w, h))
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, gray.shape[1])
    y1 = min(y + h + pad, gray.shape[0])
    cropped = gray[y0:y1, x0:x1]

    side = max(cropped.shape)
    square = np.zeros((side, side), dtype=cropped.dtype)
    y_off = (side - cropped.shape[0]) // 2
    x_off = (side - cropped.shape[1]) // 2
    square[y_off:y_off + cropped.shape[0], x_off:x_off + cropped.shape[1]] = cropped
    return cv2.resize(square, (IMAGE_SIZE, IMAGE_SIZE))


def process_folder(input_dir: Path, output_dir: Path) -> None:
    for person_dir in sorted(input_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        out_person_dir = output_dir / person_dir.name
        out_person_dir.mkdir(parents=True, exist_ok=True)
        for image_path in person_dir.glob("*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                processed = preprocess_image(image_path)
                out_path = out_person_dir / image_path.name
                cv2.imwrite(str(out_path), processed)
            except ValueError:
                continue


def list_images(data_dir: Path):
    classes = {}
    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        images = [
            p for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if images:
            classes[person_dir.name] = sorted(images)
    return classes


def split_dataset(classes, train_ratio=0.7, val_ratio=0.15):
    rng = np.random.default_rng(RANDOM_SEED)
    splits = {}
    for label, images in classes.items():
        indices = np.arange(len(images))
        rng.shuffle(indices)
        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)
        split = {
            "train": [str(images[i]) for i in indices[:train_end]],
            "val": [str(images[i]) for i in indices[train_end:val_end]],
            "test": [str(images[i]) for i in indices[val_end:]],
        }
        splits[label] = split
    return splits


def save_splits(splits, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def load_splits(split_path: Path):
    with split_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_image_list(splits, split_name: str):
    image_paths = []
    labels = []
    for label, split in splits.items():
        for path_str in split[split_name]:
            image_paths.append(Path(path_str))
            labels.append(label)
    return image_paths, labels


def preprocess_cmd(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    process_folder(input_dir, output_dir)
    print(f"Preprocessing complete. Output: {output_dir}")


def capture_cmd(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Unable to open camera.")
    count = 0
    print("Press SPACE to capture, Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Palm", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            count += 1
            out_path = output_dir / f"{args.prefix}_{count:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"Saved {out_path}")
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def generate_synthetic_palm(rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(0.5, 0.08, (IMAGE_SIZE, IMAGE_SIZE))
    image = np.clip(base * 255, 0, 255).astype(np.uint8)

    center = (IMAGE_SIZE // 2 + rng.integers(-6, 6), IMAGE_SIZE // 2 + rng.integers(-6, 6))
    axes = (rng.integers(38, 46), rng.integers(50, 58))
    cv2.ellipse(image, center, axes, 0, 0, 360, 140, thickness=1)

    for _ in range(rng.integers(10, 16)):
        points = []
        x, y = rng.integers(10, IMAGE_SIZE - 10, size=2)
        for _ in range(rng.integers(3, 6)):
            x += int(rng.integers(-18, 18))
            y += int(rng.integers(-18, 18))
            x = int(np.clip(x, 5, IMAGE_SIZE - 5))
            y = int(np.clip(y, 5, IMAGE_SIZE - 5))
            points.append([x, y])
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        color = int(rng.integers(90, 150))
        thickness = int(rng.integers(1, 2))
        cv2.polylines(image, [pts], False, color, thickness=thickness)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image


def download_cmd(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    for class_idx in range(1, args.classes + 1):
        class_dir = output_dir / f"{args.prefix}_{class_idx:02d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        class_seed = rng.integers(0, 1_000_000)
        class_rng = np.random.default_rng(class_seed)

        for image_idx in range(1, args.images_per_class + 1):
            image = generate_synthetic_palm(class_rng)
            out_path = class_dir / f"sample_{image_idx:04d}.jpg"
            cv2.imwrite(str(out_path), image)

    print(f"Sample dataset created at {output_dir}")


def compute_class_mapping(classes):
    return {label: idx for idx, label in enumerate(sorted(classes))}


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train_cmd(args):
    data_dir = Path(args.data)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")
    classes = list_images(data_dir)
    if len(classes) < 4:
        raise SystemExit("Need at least 4 identities for this assignment.")
    splits = split_dataset(classes)
    splits_path = MODELS_DIR / "splits.json"
    save_splits(splits, splits_path)

    label_to_index = compute_class_mapping(classes.keys())
    train_paths, train_labels = build_image_list(splits, "train")
    val_paths, val_labels = build_image_list(splits, "val")

    train_dataset = PalmDataset(train_paths, train_labels, label_to_index, augment=True)
    val_dataset = PalmDataset(val_paths, val_labels, label_to_index, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PalmCNN(num_classes=len(label_to_index)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        history.append([epoch, train_loss, train_acc, val_loss, val_acc])
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = MODELS_DIR / args.save
            torch.save(model.state_dict(), save_path)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = REPORTS_DIR / "train_history.csv"
    with history_path.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for row in history:
            f.write(",".join(str(x) for x in row) + "\n")

    class_path = MODELS_DIR / "classes.json"
    with class_path.open("w", encoding="utf-8") as f:
        json.dump(label_to_index, f, indent=2)

    print(f"Training complete. Best val acc: {best_acc:.3f}")


@torch.no_grad()
def evaluate_known(model, loader, device):
    model.eval()
    preds = []
    labels = []
    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
        labels.extend(targets.numpy().tolist())
    return np.array(labels), np.array(preds)


@torch.no_grad()
def evaluate_unknown(model, unknown_dir: Path, device):
    model.eval()
    image_paths = [
        p for p in unknown_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not image_paths:
        return None
    dummy_labels = ["unknown"] * len(image_paths)
    label_to_index = {"unknown": 0}
    dataset = PalmDataset(image_paths, dummy_labels, label_to_index, augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    rejected = 0
    for images, _ in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        max_conf = probs.max(dim=1).values
        rejected += (max_conf < UNKNOWN_THRESHOLD).sum().item()
    return rejected, len(image_paths)


def plot_confusion_matrix(cm, labels, out_path: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_cmd(args):
    data_dir = Path(args.data)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    model_path = Path(args.model)
    classes_path = MODELS_DIR / "classes.json"
    splits_path = MODELS_DIR / "splits.json"
    if not model_path.exists() or not classes_path.exists() or not splits_path.exists():
        raise SystemExit("Missing model or split metadata. Train the model first.")

    with classes_path.open("r", encoding="utf-8") as f:
        label_to_index = json.load(f)
    index_to_label = {v: k for k, v in label_to_index.items()}

    splits = load_splits(splits_path)
    test_paths, test_labels = build_image_list(splits, "test")
    test_dataset = PalmDataset(test_paths, test_labels, label_to_index, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PalmCNN(num_classes=len(label_to_index)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_true, y_pred = evaluate_known(model, test_loader, device)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    print(classification_report(y_true, y_pred, target_names=[index_to_label[i] for i in sorted(index_to_label)]))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, [index_to_label[i] for i in sorted(index_to_label)], REPORTS_DIR / "confusion_matrix.png")

    if args.unknown:
        unknown_dir = Path(args.unknown)
        result = evaluate_unknown(model, unknown_dir, device)
        if result:
            rejected, total = result
            print(f"Unknown rejection rate: {rejected}/{total} = {rejected / total:.3f}")
        else:
            print("No unknown images found.")


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = extract_palm_roi(frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = gray.astype(np.float32) / 255.0
    return gray


def webcam_cmd(args):
    model_path = Path(args.model)
    classes_path = Path(args.classes)
    with classes_path.open("r", encoding="utf-8") as f:
        label_to_index = json.load(f)
    index_to_label = {v: k for k, v in label_to_index.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PalmCNN(num_classes=len(label_to_index)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Unable to open camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = preprocess_frame(frame)
        tensor = torch.tensor(processed).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        conf_val = conf.item()
        label = index_to_label[pred_idx.item()] if conf_val >= UNKNOWN_THRESHOLD else "Unknown"

        display = frame.copy()
        cv2.putText(
            display,
            f"{label} ({conf_val:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
        cv2.imshow("Palm Recognition", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(description="Palm recognition toolkit.")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Create a synthetic sample dataset.")
    download.add_argument("--output", default="data/raw")
    download.add_argument("--classes", type=int, default=4)
    download.add_argument("--images-per-class", type=int, default=30)
    download.add_argument("--prefix", default="person")
    download.set_defaults(func=download_cmd)

    capture = sub.add_parser("capture", help="Capture palm images from webcam.")
    capture.add_argument("--output", required=True)
    capture.add_argument("--prefix", default="palm")
    capture.add_argument("--camera", type=int, default=0)
    capture.set_defaults(func=capture_cmd)

    preprocess = sub.add_parser("preprocess", help="Preprocess raw images.")
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--output", required=True)
    preprocess.set_defaults(func=preprocess_cmd)

    train = sub.add_parser("train", help="Train palm recognition model.")
    train.add_argument("--data", required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--save", default="palm_cnn.pt")
    train.set_defaults(func=train_cmd)

    evaluate = sub.add_parser("evaluate", help="Evaluate model performance.")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--unknown")
    evaluate.set_defaults(func=evaluate_cmd)

    webcam = sub.add_parser("webcam", help="Run live webcam recognition.")
    webcam.add_argument("--model", required=True)
    webcam.add_argument("--classes", required=True)
    webcam.add_argument("--camera", type=int, default=0)
    webcam.set_defaults(func=webcam_cmd)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
