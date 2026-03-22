import pandas as pd
import numpy as np
import mlflow
import os
import requests
import pickle
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import onnxruntime as rt
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR         = "data/processed"
MODEL_DIR             = "models/classifier"
IMAGE_CACHE           = "data/processed/images"
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(IMAGE_CACHE, exist_ok=True)

MIN_SAMPLES_PER_CLASS = 150
MAX_SAMPLES_PER_CLASS = 500
IMG_SIZE              = 224
BATCH_SIZE            = 32
CONFIGS = [
    {"epochs": 20, "lr": 0.001, "freeze_layers": False, "dropout": 0.4},
]

class ProductImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def download_images(df):
    print("Downloading images...")
    paths = []
    for i, row in df.iterrows():
        filename = f"{IMAGE_CACHE}/{abs(hash(row['image_url'])) % 1000000}.jpg"

        if os.path.exists(filename):
            paths.append(filename)
            continue

        try:
            response = requests.get(row["image_url"], timeout=5)
            img      = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(filename)
            paths.append(filename)
        except:
            paths.append(None)

        if i % 100 == 0:
            print(f"  Downloaded {len([p for p in paths if p])} / {len(paths)} images...")

    df = df.copy()
    df["image_path"] = paths
    df = df[df["image_path"].notna()]
    print(f"  Successfully downloaded: {len(df)} images")
    return df

def prepare_data():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "products_clean.csv"))

    cat_counts = df["main_category"].value_counts()
    valid_cats = cat_counts[cat_counts >= MIN_SAMPLES_PER_CLASS].index
    df         = df[df["main_category"].isin(valid_cats)]

    df = df.groupby("main_category").apply(
        lambda x: x.sample(min(len(x), MAX_SAMPLES_PER_CLASS), random_state=42)
    ).reset_index(drop=True)

    print("Dataset after balancing:")
    print(df["main_category"].value_counts())

    df = download_images(df)

    le          = LabelEncoder()
    df["label"] = le.fit_transform(df["main_category"])
    num_classes = len(le.classes_)

    print(f"\nClasses ({num_classes}): {list(le.classes_)}")

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    return train_df, test_df, le, num_classes

def build_model(num_classes, freeze_layers=True, dropout=0.3):
    model = models.resnet50(weights="IMAGENET1K_V1")

    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, classes, run_name):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"Confusion Matrix - {run_name}")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, f"confusion_matrix_{run_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, run_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses,   label="Val")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(train_accs, label="Train")
    ax2.plot(val_accs,   label="Val")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.suptitle(f"Training Curves - {run_name}")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, f"training_curves_{run_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

def export_to_onnx(model, run_name, device):
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    path        = os.path.join(MODEL_DIR, f"classifier_{run_name}.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=11,
    )
    return path

def verify_onnx(onnx_path, test_loader, device):
    sess       = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    correct, total = 0, 0
    for images, labels in list(test_loader)[:5]:
        preds = sess.run(None, {input_name: images.numpy()})[0]
        preds = np.argmax(preds, axis=1)
        correct += (preds == labels.numpy()).sum()
        total   += len(labels)
    return correct / total

def train_and_log(config, train_df, test_df, le, num_classes, run_index):
    run_name = f"resnet50_run_{run_index:02d}"
    device   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Using device: {device}")

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = ProductImageDataset(
        train_df["image_path"].tolist(),
        train_df["label"].tolist(),
        transform_train
    )
    test_dataset = ProductImageDataset(
        test_df["image_path"].tolist(),
        test_df["label"].tolist(),
        transform_test
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)
        mlflow.log_param("model",       "ResNet50")
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("train_size",  len(train_df))
        mlflow.log_param("test_size",   len(test_df))
        mlflow.log_param("img_size",    IMG_SIZE)
        mlflow.log_param("batch_size",  BATCH_SIZE)
        mlflow.log_param("device",      str(device))

        model     = build_model(
            num_classes,
            freeze_layers=config["freeze_layers"],
            dropout=config["dropout"]
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["lr"]
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=3
        )

        train_losses, val_losses = [], []
        train_accs,   val_accs   = [], []

        for epoch in range(config["epochs"]):
            if epoch == 3 and not config["freeze_layers"]:
                for name, param in model.named_parameters():
                    if "layer2" in name or "layer3" in name or "layer4" in name or "fc" in name:
                        param.requires_grad = True
                print("    Unfreezing layer2, layer3, layer4 after warmup")
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=config["lr"] * 0.1
                )

            train_loss, train_acc         = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss,   val_acc, _, _     = evaluate(model, test_loader, criterion, device)
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss",   val_loss,   step=epoch)
            mlflow.log_metric("train_acc",  train_acc,  step=epoch)
            mlflow.log_metric("val_acc",    val_acc,    step=epoch)

            print(f"    Epoch {epoch+1}/{config['epochs']} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        _, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
        report = classification_report(
            all_labels, all_preds, target_names=le.classes_, output_dict=True
        )

        mlflow.log_metric("final_accuracy", final_acc)
        mlflow.log_metric("macro_f1",       report["macro avg"]["f1-score"])
        print(f"  {run_name} | final_acc={final_acc:.4f} | macro_f1={report['macro avg']['f1-score']:.4f}")

        cm_path     = plot_confusion_matrix(all_labels, all_preds, le.classes_, run_name)
        curves_path = plot_training_curves(train_losses, val_losses, train_accs, val_accs, run_name)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(curves_path)

        model_path = os.path.join(MODEL_DIR, f"resnet50_{run_name}.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        mlflow.set_tag("model_path", model_path)

        onnx_path = export_to_onnx(model.cpu(), run_name, torch.device("cpu"))
        onnx_acc  = verify_onnx(onnx_path, test_loader, device)
        mlflow.log_metric("onnx_accuracy", onnx_acc)
        mlflow.log_artifact(onnx_path)
        mlflow.set_tag("onnx_path", onnx_path)
        print(f"    ONNX verified | acc={onnx_acc:.4f}")

        return {
            "run_name":    run_name,
            "config":      config,
            "accuracy":    final_acc,
            "macro_f1":    report["macro avg"]["f1-score"],
            "onnx_path":   onnx_path,
            "model_path":  model_path,
            "num_classes": num_classes,
            "classes":     list(le.classes_),
        }

def main():
    mlflow.set_experiment("image_classifier")
    train_df, test_df, le, num_classes = prepare_data()

    results = []
    for i, config in enumerate(CONFIGS):
        print(f"\nTraining config {i+1}/{len(CONFIGS)}: {config}")
        result = train_and_log(config, train_df, test_df, le, num_classes, i + 1)
        results.append(result)

    best = max(results, key=lambda x: x["accuracy"])

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        marker = " <- best" if r["run_name"] == best["run_name"] else ""
        print(f"  {r['run_name']} | acc={r['accuracy']:.4f} | macro_f1={r['macro_f1']:.4f}{marker}")

    print(f"\nBest model  : {best['run_name']}")
    print(f"Accuracy    : {best['accuracy']:.4f}")
    print(f"Classes     : {best['classes']}")
    print(f"ONNX model  : {best['onnx_path']}")
    print("\nCheck MLflow UI at http://localhost:5001")

if __name__ == "__main__":
    main()