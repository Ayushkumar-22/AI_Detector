import argparse
import io
import os
import random
import time
from PIL import Image, ImageFilter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JpegCompression:
    def __init__(self, quality_lower=50, quality_upper=90, p=0.5):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.quality_lower, self.quality_upper)
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer)
        return img

class GaussianNoise:
    def __init__(self, mean=0.0, std_limit=0.05, p=0.5):
        self.mean = mean
        self.std_limit = std_limit
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            std = random.uniform(0.01, self.std_limit)
            noise = torch.randn(tensor.size()) * std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

class MotionBlur:
    def __init__(self, kernel_size_range=(3, 7), p=0.5):
        self.kernel_size_range = kernel_size_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            kernel_size = random.choice(range(self.kernel_size_range[0], self.kernel_size_range[1] + 1, 2))
            kernel = [1.0 / kernel_size] * kernel_size
            direction = random.choice(["horizontal", "vertical"])
            try:
                if direction == "horizontal":
                    img = img.filter(ImageFilter.Kernel((kernel_size, 1), kernel, scale=1))
                else:
                    img = img.filter(ImageFilter.Kernel((1, kernel_size), kernel, scale=1))
            except ValueError:
                pass
        return img

def get_data_transforms(image_size: int = 224):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.25, hue=0.05)
        ], p=0.85),
        JpegCompression(quality_lower=50, quality_upper=90, p=0.5),
        MotionBlur(kernel_size_range=(3, 7), p=0.4),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))
        ], p=0.6),
        transforms.RandomApply([
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5)
        ], p=0.4),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        GaussianNoise(std_limit=0.05, p=0.4),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms


def build_model(num_classes: int = 2, backbone: str = "resnet50", dropout_prob: float = 0.4):
    if backbone == "efficientnet_b0":
        try:
            model = models.efficientnet_b0(pretrained=True)
            in_features = model.classifier[1].in_features
            # Robust classification head
            model.classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(dropout_prob / 2),
                nn.Linear(512, num_classes),
            )
            return model
        except AttributeError:
            pass

    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout_prob),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(dropout_prob / 2),
        nn.Linear(512, num_classes),
    )
    return model


def mixup_data(x, y, alpha: float = 0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def prepare_dataloaders(data_dir: Path, batch_size: int, image_size: int = 224, num_workers: int = 4):
    train_transforms, val_transforms = get_data_transforms(image_size)
    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset.classes


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    conf_mat = confusion_matrix(all_labels, all_preds)

    # Confidence Threshold Tuning
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        thresh_preds = (np.array(all_probs) >= thresh).astype(int)
        temp_f1 = f1_score(all_labels, thresh_preds, zero_division=0)
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_thresh = thresh

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": conf_mat.tolist(),
        "optimal_threshold": float(best_thresh),
        "optimal_f1": float(best_f1),
        "predictions": all_preds,
        "probabilities": all_probs,
        "labels": all_labels,
    }


def train_one_epoch(model, criterion, optimizer, dataloader, device, mixup_alpha: float = 0.3, scheduler=None):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if mixup_alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def train(model, train_loader, val_loader, device, epochs: int, lr: float, weight_decay: float, patience: int, output_dir: Path):
    # Label smoothing acts as probability calibration and reduces overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Differential learning rates: preserve backbone features, adapt head quickly
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': head_params, 'lr': lr}
    ], weight_decay=weight_decay)
    
    # OneCycleLR handles warmup and annealing per batch, leading to much better generalization
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[lr * 0.1, lr], 
        epochs=epochs, 
        steps_per_epoch=len(train_loader)
    )

    best_auc = 0.0
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, scheduler=scheduler)
        val_metrics = evaluate_model(model, val_loader, device)

        elapsed = time.time() - start_time
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1_score"],
            "val_auc": val_metrics["roc_auc"],
            "elapsed": elapsed,
        })

        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_auc={val_metrics['roc_auc']:.4f} | val_f1={val_metrics['f1_score']:.4f} (Opt Thresh: {val_metrics['optimal_threshold']:.2f}) | time={elapsed:.1f}s")

        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_state = model.state_dict()
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
            }, output_dir / "best_model.pth")

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_auc


def main():
    parser = argparse.ArgumentParser(description="Train an AI vs Real image classifier")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory containing train/val folders")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"], help="Backbone architecture")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mixup-alpha", type=float, default=0.3, help="MixUp alpha for augmentation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        print("Please create the directory structure:")
        print(f"  {data_dir}/train/real/  (put real images here)")
        print(f"  {data_dir}/train/ai/    (put AI-generated images here)")
        print(f"  {data_dir}/val/real/    (put real validation images here)")
        print(f"  {data_dir}/val/ai/      (put AI validation images here)")
        return

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        print(f"Error: Missing train/val subdirectories in '{data_dir}'.")
        print("Expected structure:")
        print(f"  {train_dir}/real/")
        print(f"  {train_dir}/ai/")
        print(f"  {val_dir}/real/")
        print(f"  {val_dir}/ai/")
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = prepare_dataloaders(data_dir, args.batch_size, args.image_size)
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

    model = build_model(num_classes=len(classes), backbone=args.backbone)
    model.to(device)

    model, history, best_auc = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        output_dir=output_dir,
    )

    print(f"Training complete. Best validation AUC: {best_auc:.4f}")
    final_metrics = evaluate_model(model, val_loader, device)
    print("Final validation metrics:")
    print(final_metrics)

    torch.save(model.state_dict(), output_dir / "final_model.pth")

if __name__ == "__main__":
    main()
