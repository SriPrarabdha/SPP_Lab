"""
FixMatch Semi-Supervised Learning Implementation
Signal Processing in Practice - Assignment 3

Dataset layout expected:
    cifar10_3class_fixmatch/
        labeled/    cat/  deer/  dog/
        val/        cat/  deer/  dog/
        test/       cat/  deer/  dog/
        unlabeled/  *.png  (flat, no subfolders)

Directory Structure (outputs):
    outputs/
        checkpoints/       - Model checkpoints
        logs/              - Training logs (CSV + JSON)
        plots/
            supervised/    - Q1 supervised training plots
            fixmatch/      - Q2 FixMatch training plots
            analysis/      - Q3 comparison plots
        visualizations/
            augmentations/ - Augmentation examples
            predictions/   - Prediction confidence plots
            embeddings/    - Feature space visualizations
"""

# ══════════════════════════════════════════════════════════════
#  CFG  — Central config block. Edit everything here.
# ══════════════════════════════════════════════════════════════
class CFG:
    # ── Paths ──────────────────────────────────────────────────
    DATA_DIR   = "./cifar10_3class_fixmatch"   # root dataset folder
    OUTPUT_DIR = "./outputs"

    # ── Data ───────────────────────────────────────────────────
    IMG_SIZE    = 32          # resize all images to this
    NUM_CLASSES = 3
    CLASS_NAMES = ["cat", "deer", "dog"]

    # ── Model ──────────────────────────────────────────────────
    IN_CHANNELS = 3

    # ── Training (General) ─────────────────────────────────────
    SEED        = 42
    EPOCHS      = 100         # epochs per run  (use 150–200 for best results)
    BATCH_SIZE  = 64          # labeled batch size B
    MU          = 7           # unlabeled/labeled ratio  → unlabeled batch = B*MU
    NUM_WORKERS = 4

    # ── Optimizer ──────────────────────────────────────────────
    LR           = 0.03       # SGD lr (works well for FixMatch)
    MOMENTUM     = 0.9
    WEIGHT_DECAY = 5e-4
    NESTEROV     = True

    # ── Scheduler ──────────────────────────────────────────────
    # "cosine" | "step" | "onecycle"
    SCHEDULER    = "cosine"
    LR_STEP_SIZE = 30         # only for "step"
    LR_GAMMA     = 0.1        # only for "step"

    # ── FixMatch defaults (used for the default run) ────────────
    TAU    = 0.95             # confidence threshold
    LAMBDA = 1.0              # weight of unlabeled loss

    # ── Hyperparameter Sweep ────────────────────────────────────
    TAU_VALUES    = [0.70, 0.85, 0.95]   # thresholds to sweep
    LAMBDA_VALUES = [0.5,  1.0,  2.0]   # lambda values to sweep

    # ── Augmentation ───────────────────────────────────────────
    RANDAUG_N    = 2          # RandAugment num_ops
    RANDAUG_M    = 9          # RandAugment magnitude
    ERASE_PROB   = 0.5        # RandomErasing probability
    NORM_MEAN    = (0.4914, 0.4822, 0.4465)
    NORM_STD     = (0.2470, 0.2435, 0.2616)

    # ── Misc ───────────────────────────────────────────────────
    SAVE_EVERY   = 10         # save checkpoint every N epochs
    TSNE_SAMPLES = 2000       # max samples for t-SNE

# ══════════════════════════════════════════════════════════════

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for VM/server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────────
# Directory Setup
# ─────────────────────────────────────────────
def setup_dirs(base=CFG.OUTPUT_DIR):
    dirs = {
        "base": base,
        "checkpoints": f"{base}/checkpoints",
        "logs": f"{base}/logs",
        "plots_supervised": f"{base}/plots/supervised",
        "plots_fixmatch": f"{base}/plots/fixmatch",
        "plots_analysis": f"{base}/plots/analysis",
        "viz_augmentations": f"{base}/visualizations/augmentations",
        "viz_predictions": f"{base}/visualizations/predictions",
        "viz_embeddings": f"{base}/visualizations/embeddings",
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

DIRS = setup_dirs()

# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────
def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

sup_logger = setup_logger("supervised", f"{DIRS['logs']}/supervised.log")
fm_logger  = setup_logger("fixmatch",   f"{DIRS['logs']}/fixmatch.log")

# ─────────────────────────────────────────────
# ResNet-9 Architecture
# ─────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Residual 1
            ResidualBlock(128, 128),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Residual 2
            ResidualBlock(512, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        out = self.classifier(pooled)
        if return_features:
            return out, pooled
        return out


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
def get_transforms(mode="train"):
    mean = CFG.NORM_MEAN
    std  = CFG.NORM_STD

    if mode == "test":
        return transforms.Compose([
            transforms.Resize(CFG.IMG_SIZE),
            transforms.CenterCrop(CFG.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    if mode == "weak":
        return transforms.Compose([
            transforms.Resize(CFG.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    if mode == "strong":
        return transforms.Compose([
            transforms.Resize(CFG.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandAugment(num_ops=CFG.RANDAUG_N, magnitude=CFG.RANDAUG_M),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandomErasing(p=CFG.ERASE_PROB, scale=(0.02, 0.33)),
        ])
    # labeled training
    return transforms.Compose([
        transforms.Resize(CFG.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(CFG.IMG_SIZE, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ─────────────────────────────────────────────
# Custom Dataset Classes
# ─────────────────────────────────────────────
class FolderDataset(Dataset):
    """
    Loads labeled/val/test splits from ImageFolder-style directories:
        root/cat/*.png
        root/deer/*.png
        root/dog/*.png
    """
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples   = []   # (path, class_idx)
        self.class_to_idx = {c: i for i, c in enumerate(sorted(CFG.CLASS_NAMES))}
        for cls_name in sorted(CFG.CLASS_NAMES):
            cls_dir = Path(root) / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {cls_dir}")
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
                    self.samples.append((str(p), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class UnlabeledFlatDataset(Dataset):
    """
    Loads unlabeled images from a FLAT directory (no subfolders):
        root/unlabeled/*.png
    Returns (weak_aug, strong_aug) pairs.
    """
    def __init__(self, root, weak_transform, strong_transform):
        self.weak_tf   = weak_transform
        self.strong_tf = strong_transform
        self.paths = sorted([
            str(p) for p in Path(root).iterdir()
            if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        ])
        if not self.paths:
            raise FileNotFoundError(f"No images found in unlabeled dir: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.weak_tf(img), self.strong_tf(img)


def load_datasets():
    """
    Loads all splits from CFG.DATA_DIR.
    Returns: labeled_ds, val_ds, test_ds, unlabeled_ds
    """
    root = Path(CFG.DATA_DIR)
    labeled_ds   = FolderDataset(root / "labeled",    transform=get_transforms("weak"))
    val_ds       = FolderDataset(root / "val",         transform=get_transforms("test"))
    test_ds      = FolderDataset(root / "test",        transform=get_transforms("test"))
    unlabeled_ds = UnlabeledFlatDataset(
        root / "unlabeled",
        weak_transform=get_transforms("weak"),
        strong_transform=get_transforms("strong"),
    )

    print(f"  Labeled   : {len(labeled_ds):,} samples")
    print(f"  Val       : {len(val_ds):,} samples")
    print(f"  Test      : {len(test_ds):,} samples")
    print(f"  Unlabeled : {len(unlabeled_ds):,} samples")
    return labeled_ds, val_ds, test_ds, unlabeled_ds


# ─────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────
CLASS_NAMES = CFG.CLASS_NAMES  # shorthand alias

def denormalize(tensor, mean=CFG.NORM_MEAN, std=CFG.NORM_STD):
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)


def save_augmentation_grid(pil_imgs, labels, save_path, n=8):
    """Shows original | weak aug | strong aug side-by-side."""
    weak_tf   = get_transforms("weak")
    strong_tf = get_transforms("strong")
    n = min(n, len(pil_imgs))
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.5), squeeze=False)
    fig.suptitle("Augmentation Examples\n(Original | Weak | Strong)", fontsize=14, fontweight='bold')

    for i in range(n):
        img = pil_imgs[i]
        lbl = CLASS_NAMES[labels[i]] if labels[i] < len(CLASS_NAMES) else str(labels[i])
        w   = denormalize(weak_tf(img)).permute(1, 2, 0).numpy()
        s   = denormalize(strong_tf(img)).permute(1, 2, 0).numpy()
        orig = np.array(img.resize((CFG.IMG_SIZE, CFG.IMG_SIZE))) / 255.0

        axes[i, 0].imshow(orig); axes[i, 0].set_title(f"Original\n({lbl})", fontsize=8)
        axes[i, 1].imshow(w);    axes[i, 1].set_title("Weak Aug",            fontsize=8)
        axes[i, 2].imshow(s);    axes[i, 2].set_title("Strong Aug",          fontsize=8)
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(history, save_dir, title_prefix=""):
    """Comprehensive training curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix} Training Curves", fontsize=16, fontweight='bold')

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], color='steelblue', marker='o', ms=3, label='Train Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(epochs, history['val_loss'], color='tomato', marker='o', ms=3, label='Val Loss')
    axes[0, 0].set_title("Loss"); axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], color='steelblue', marker='o', ms=3, label='Train Acc')
    if 'val_acc' in history and len(history['val_acc']) == len(epochs):
        axes[0, 1].plot(epochs, history['val_acc'], color='darkorange', marker='o', ms=3, label='Val Acc')
    axes[0, 1].plot(epochs, history['test_acc'], color='mediumseagreen', marker='o', ms=3, label='Test Acc')
    axes[0, 1].set_title("Accuracy"); axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # Learning Rate
    if 'lr' in history:
        axes[1, 0].plot(epochs, history['lr'], color='purple', marker=None)
        axes[1, 0].set_title("Learning Rate"); axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("LR"); axes[1, 0].set_yscale('log'); axes[1, 0].grid(alpha=0.3)
    else:
        axes[1, 0].axis('off')

    # Smoothed test accuracy
    smoothed = pd.Series(history['test_acc']).rolling(5, min_periods=1).mean().tolist()
    axes[1, 1].plot(epochs, history['test_acc'], color='lightsteelblue', lw=1, label='Raw Test Acc')
    axes[1, 1].plot(epochs, smoothed, color='mediumseagreen', lw=2, label='Smoothed (k=5)')
    axes[1, 1].set_title("Smoothed Test Accuracy"); axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/training_curves.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_fixmatch_curves(history, save_dir):
    """FixMatch-specific plots including pseudo-label stats."""
    # FixMatch history uses 'total_loss', not 'train_loss'
    loss_key = 'total_loss' if 'total_loss' in history else 'train_loss'
    epochs = range(1, len(history[loss_key]) + 1)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("FixMatch Training Analysis", fontsize=16, fontweight='bold')

    # Loss breakdown
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history[loss_key], 'b-', lw=1.5, label='Total')
    if 'sup_loss' in history:
        ax1.plot(epochs, history['sup_loss'],   'g--', lw=1.5, label='Supervised')
    if 'unsup_loss' in history:
        ax1.plot(epochs, history['unsup_loss'], 'r--', lw=1.5, label='Unsupervised')
    ax1.set_title("Loss Breakdown"); ax1.set_xlabel("Epoch")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # Accuracy (test + val if available)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['test_acc'], 'g-o', ms=3, lw=1.5, label='Test Acc')
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'b-o', ms=3, lw=1.5, label='Val Acc')
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Acc (%)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Mask ratio
    ax3 = fig.add_subplot(gs[0, 2])
    if 'mask_ratio' in history:
        ax3.plot(epochs, history['mask_ratio'], color='darkorange', lw=1.5)
        ax3.set_title("Pseudo-label Mask Ratio"); ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Ratio"); ax3.set_ylim(0, 1); ax3.grid(alpha=0.3)

    # Per-class pseudo-label confidence (last epoch)
    ax4 = fig.add_subplot(gs[1, :2])
    if 'class_confidence' in history and history['class_confidence']:
        data = np.array(history['class_confidence'][-1])
        n_cls = len(data)
        bar_colors = [plt.cm.Set2.colors[i % 8] for i in range(n_cls)]
        ax4.bar(CLASS_NAMES[:n_cls], data, color=bar_colors)
        ax4.set_title("Per-Class Avg Pseudo-label Confidence (Last Epoch)")
        ax4.set_ylabel("Avg Confidence"); ax4.set_ylim(0, 1); ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    # Lambda sweep comparison
    ax5 = fig.add_subplot(gs[1, 2])
    if 'lambda_sweep' in history:
        ls = history['lambda_sweep']
        ax5.bar([str(l) for l in ls['lambdas']], ls['best_acc'], color='steelblue')
        ax5.set_title("Lambda Sweep: Best Test Acc"); ax5.set_xlabel("Lambda")
        ax5.set_ylabel("Acc (%)"); ax5.grid(axis='y', alpha=0.3)

    # Threshold sweep comparison
    ax6 = fig.add_subplot(gs[2, :2])
    if 'tau_sweep' in history:
        ts = history['tau_sweep']
        ax6.bar([str(t) for t in ts['taus']], ts['best_acc'], color='coral')
        ax6.set_title("Threshold (τ) Sweep: Best Test Acc"); ax6.set_xlabel("τ")
        ax6.set_ylabel("Acc (%)"); ax6.grid(axis='y', alpha=0.3)

    # LR
    ax7 = fig.add_subplot(gs[2, 2])
    if 'lr' in history:
        ax7.plot(epochs, history['lr'], color='purple', lw=1.5)
        ax7.set_title("Learning Rate"); ax7.set_xlabel("Epoch")
        ax7.set_yscale('log'); ax7.grid(alpha=0.3)

    path = f"{save_dir}/fixmatch_analysis.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for ax, data, fmt, t in zip(axes, [cm, cm_norm], ['d', '.2f'],
                                 ['Count', 'Normalized']):
        sns.heatmap(data, annot=True, fmt=fmt, ax=ax,
                    xticklabels=classes, yticklabels=classes,
                    cmap='Blues', linewidths=0.5)
        ax.set_title(t); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_accuracy(y_true, y_pred, classes, save_path, title="Per-Class Accuracy"):
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1) * 100

    # Build a color per bar using RdYlGn: low acc=red, high acc=green
    cmap = plt.cm.RdYlGn
    bar_colors = [cmap(v / 100.0) for v in per_class]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, per_class, color=bar_colors)
    ax.set_title(title, fontsize=13); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    for bar, val in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confidence_histogram(probs, y_true, y_pred, save_path):
    """Histogram of model confidence for correct vs incorrect predictions."""
    probs = np.array(probs)
    max_conf = probs.max(axis=1)
    correct = np.array(y_true) == np.array(y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Prediction Confidence Distribution", fontsize=13, fontweight='bold')
    
    axes[0].hist(max_conf[correct],   bins=40, alpha=0.7, color='green',  label='Correct', density=True)
    axes[0].hist(max_conf[~correct],  bins=40, alpha=0.7, color='red',    label='Wrong',   density=True)
    axes[0].set_title("Confidence: Correct vs Wrong")
    axes[0].set_xlabel("Max Probability"); axes[0].set_ylabel("Density")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    
    axes[1].hist(max_conf, bins=40, color='steelblue', edgecolor='white')
    axes[1].set_title("Overall Confidence Distribution")
    axes[1].set_xlabel("Max Probability"); axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne(features, labels, save_path, title="t-SNE Feature Space", classes=None):
    """t-SNE visualization of feature embeddings."""
    print("  Running t-SNE (this may take a moment)...")
    labels = np.array(labels)
    unique_classes = sorted(np.unique(labels).tolist())
    n_cls = len(unique_classes)

    # Clamp perplexity: must be < n_samples
    n_samples = len(features)
    perplexity = min(40, n_samples - 1)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    emb  = tsne.fit_transform(features)

    # Use tab10 for up to 10 classes; sample evenly
    cmap   = plt.cm.tab10
    colors = [cmap(i / max(n_cls - 1, 1)) for i in range(n_cls)]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, c in enumerate(unique_classes):
        mask = labels == c
        lbl  = classes[c] if (classes is not None and c < len(classes)) else str(c)
        ax.scatter(emb[mask, 0], emb[mask, 1], color=colors[i],
                   label=lbl, alpha=0.6, s=12)

    ax.legend(markerscale=2, fontsize=9, loc='best')
    ax.set_title(title, fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison(sup_history, fm_history, best_fm_history, save_dir):
    """Q3 Analysis: side-by-side comparison of supervised vs FixMatch."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Analysis: Supervised vs FixMatch", fontsize=16, fontweight='bold')
    
    e_sup = range(1, len(sup_history['test_acc'])+1)
    e_fm  = range(1, len(fm_history['test_acc'])+1)
    e_bfm = range(1, len(best_fm_history['test_acc'])+1)
    
    # Test accuracy over time
    ax1 = fig.add_subplot(gs[0,:2])
    ax1.plot(e_sup, sup_history['test_acc'],      'b-o', ms=3, lw=1.5, label='Supervised (Q1)')
    ax1.plot(e_fm,  fm_history['test_acc'],       'r-o', ms=3, lw=1.5, label='FixMatch (default τ,λ)')
    ax1.plot(e_bfm, best_fm_history['test_acc'],  'g-o', ms=3, lw=1.5, label='FixMatch (best τ,λ)')
    ax1.set_title("Test Accuracy over Epochs"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
    ax1.legend(); ax1.grid(alpha=0.3)
    
    # Final accuracy bar
    ax2 = fig.add_subplot(gs[0,2])
    names = ['Supervised', 'FixMatch\n(default)', 'FixMatch\n(best)']
    accs  = [max(sup_history['test_acc']),
             max(fm_history['test_acc']),
             max(best_fm_history['test_acc'])]
    bars  = ax2.bar(names, accs, color=['steelblue','tomato','mediumseagreen'], edgecolor='white')
    ax2.set_title("Best Test Accuracy Comparison"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(min(accs)-5, min(100, max(accs)+5)); ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, accs):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.2f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Loss comparison
    ax3 = fig.add_subplot(gs[1,0])
    ax3.plot(e_sup, sup_history['train_loss'], 'b-', lw=1.5, label='Supervised')
    ax3.plot(e_fm,  fm_history['total_loss'],  'r-', lw=1.5, label='FixMatch')
    ax3.set_title("Training Loss"); ax3.set_xlabel("Epoch"); ax3.legend(); ax3.grid(alpha=0.3)
    
    # Improvement over epochs
    ax4 = fig.add_subplot(gs[1,1])
    sup_smooth = pd.Series(sup_history['test_acc']).rolling(5, min_periods=1).mean()
    fm_smooth  = pd.Series(fm_history['test_acc']).rolling(5, min_periods=1).mean()
    ax4.plot(e_sup, sup_smooth, 'b-', lw=2, label='Supervised (smoothed)')
    ax4.plot(e_fm,  fm_smooth,  'r-', lw=2, label='FixMatch (smoothed)')
    ax4.set_title("Smoothed Test Accuracy"); ax4.set_xlabel("Epoch"); ax4.legend(); ax4.grid(alpha=0.3)
    
    # Mask ratio shows pseudo-label quality
    ax5 = fig.add_subplot(gs[1,2])
    if 'mask_ratio' in fm_history:
        ax5.plot(e_fm, fm_history['mask_ratio'], 'orange', lw=2)
        ax5.set_title("FixMatch: Pseudo-label Mask Ratio")
        ax5.set_xlabel("Epoch"); ax5.set_ylabel("Mask Ratio"); ax5.grid(alpha=0.3)
    
    # Summary text box
    ax6 = fig.add_subplot(gs[2,:])
    ax6.axis('off')
    sup_best = max(sup_history['test_acc'])
    fm_best  = max(best_fm_history['test_acc'])
    gain = fm_best - sup_best
    summary = (
        f"Summary\n\n"
        f"Supervised Baseline Best Accuracy : {sup_best:.2f}%\n"
        f"FixMatch Best Accuracy (best τ,λ)  : {fm_best:.2f}%\n"
        f"Absolute Gain from Unlabeled Data  : {gain:+.2f}%\n\n"
        f"FixMatch uses pseudo-labeling on unlabeled data (mask ratio shows how many samples exceed threshold τ).\n"
        f"Higher mask ratio = more unlabeled samples used = more signal from unlabeled data.\n"
        f"As training progresses, the model becomes more confident → mask ratio increases → virtuous cycle."
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    path = f"{save_dir}/supervised_vs_fixmatch.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_hyperparameter_heatmap(results_grid, taus, lambdas, save_path):
    """2D heatmap of τ × λ sweep results."""
    grid = np.array([[results_grid[(t,l)] for l in lambdas] for t in taus])
    
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(grid, cmap='YlGn', aspect='auto',
                   vmin=grid.min()-1, vmax=grid.max()+1)
    
    ax.set_xticks(range(len(lambdas))); ax.set_xticklabels([str(l) for l in lambdas])
    ax.set_yticks(range(len(taus)));   ax.set_yticklabels([str(t) for t in taus])
    ax.set_xlabel("Lambda (λ)"); ax.set_ylabel("Threshold (τ)")
    ax.set_title("Hyperparameter Sweep: Test Accuracy (%) — τ × λ", fontsize=13)
    
    for i in range(len(taus)):
        for j in range(len(lambdas)):
            ax.text(j, i, f'{grid[i,j]:.1f}', ha='center', va='center',
                    fontsize=10, color='black', fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Test Accuracy (%)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# Training Utilities
# ─────────────────────────────────────────────
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val*n; self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def evaluate(model, loader, device, return_preds=False):
    model.eval()
    correct = total = 0
    all_preds, all_labels, all_probs, all_feats = [], [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if return_preds:
            out, feats = model(imgs, return_features=True)
        else:
            out = model(imgs)

        probs = F.softmax(out, dim=-1)
        preds = probs.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if return_preds:
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())
            all_feats.append(feats.cpu().numpy())

    acc = 100.0 * correct / total
    if return_preds:
        return acc, all_labels, all_preds, np.array(all_probs), np.vstack(all_feats)
    return acc


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt.get('epoch', 0), ckpt.get('best_acc', 0)


def save_history(history, path):
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


# ─────────────────────────────────────────────
# Scheduler helper
# ─────────────────────────────────────────────
def build_scheduler(optimizer, epochs, steps_per_epoch=1):
    if CFG.SCHEDULER == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif CFG.SCHEDULER == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=CFG.LR_STEP_SIZE,
                                          gamma=CFG.LR_GAMMA)
    elif CFG.SCHEDULER == "onecycle":
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.LR,
                                              steps_per_epoch=steps_per_epoch,
                                              epochs=epochs)
    raise ValueError(f"Unknown scheduler: {CFG.SCHEDULER}. Choose 'cosine', 'step', or 'onecycle'.")


# ─────────────────────────────────────────────
# Q1: Supervised Training
# ─────────────────────────────────────────────
def train_supervised():
    print("\n" + "="*60)
    print("Q1: SUPERVISED TRAINING")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labeled_ds, val_ds, test_ds, unlabeled_ds = load_datasets()

    labeled_loader = DataLoader(labeled_ds, batch_size=CFG.BATCH_SIZE,
                                shuffle=True,  num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader     = DataLoader(val_ds,     batch_size=256,
                                shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    test_loader    = DataLoader(test_ds,    batch_size=256,
                                shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # ── Augmentation examples from labeled set ──
    raw_imgs = [labeled_ds.samples[i][0] for i in range(min(8, len(labeled_ds)))]
    raw_pil  = [Image.open(p).convert("RGB") for p in raw_imgs]
    raw_lbls = [labeled_ds.samples[i][1] for i in range(min(8, len(labeled_ds)))]
    save_augmentation_grid(raw_pil, raw_lbls,
                           f"{DIRS['viz_augmentations']}/augmentation_examples.png")

    model     = ResNet9(in_channels=CFG.IN_CHANNELS,
                        num_classes=CFG.NUM_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=CFG.LR, momentum=CFG.MOMENTUM,
                          weight_decay=CFG.WEIGHT_DECAY, nesterov=CFG.NESTEROV)
    scheduler = build_scheduler(optimizer, CFG.EPOCHS,
                                 steps_per_epoch=len(labeled_loader))
    criterion = nn.CrossEntropyLoss()

    history  = defaultdict(list)
    best_acc = 0.0

    for epoch in range(1, CFG.EPOCHS + 1):
        model.train()
        loss_m = AverageMeter()
        correct = total = 0

        for imgs, labels in labeled_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            loss_m.update(loss.item(), imgs.size(0))
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        train_acc  = 100.0 * correct / total
        val_acc    = evaluate(model, val_loader,  device)
        test_acc   = evaluate(model, test_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(loss_m.avg)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)

        sup_logger.info(f"Epoch {epoch:3d}/{CFG.EPOCHS} | "
                        f"Loss: {loss_m.avg:.4f} | Train: {train_acc:.2f}% | "
                        f"Val: {val_acc:.2f}% | Test: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(),
                             'optimizer_state': optimizer.state_dict(),
                             'best_acc': best_acc},
                            f"{DIRS['checkpoints']}/supervised_best.pt")

        if epoch % CFG.SAVE_EVERY == 0:
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(),
                             'best_acc': best_acc},
                            f"{DIRS['checkpoints']}/supervised_epoch{epoch}.pt")

    sup_logger.info(f"Best Test Accuracy: {best_acc:.2f}%")

    # ── Persist history ──
    save_history(dict(history), f"{DIRS['logs']}/supervised_history.json")
    pd.DataFrame({k: v for k, v in history.items()}).to_csv(
        f"{DIRS['logs']}/supervised_history.csv", index=False)

    # ── Training curves ──
    plot_training_curves(dict(history), DIRS['plots_supervised'],
                         title_prefix="Supervised (Q1)")

    # ── Evaluation plots using best checkpoint ──
    load_checkpoint(f"{DIRS['checkpoints']}/supervised_best.pt", model)
    _, y_true, y_pred, probs, feats = evaluate(model, test_loader, device, return_preds=True)

    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
                          f"{DIRS['plots_supervised']}/confusion_matrix.png",
                          "Supervised: Confusion Matrix")
    plot_per_class_accuracy(y_true, y_pred, CLASS_NAMES,
                            f"{DIRS['plots_supervised']}/per_class_accuracy.png",
                            "Supervised: Per-Class Accuracy")
    plot_confidence_histogram(probs, y_true, y_pred,
                              f"{DIRS['viz_predictions']}/supervised_confidence.png")

    n_tsne = min(CFG.TSNE_SAMPLES, len(feats))
    idx    = np.random.choice(len(feats), n_tsne, replace=False)
    plot_tsne(feats[idx], np.array(y_true)[idx],
              f"{DIRS['viz_embeddings']}/supervised_tsne.png",
              "Supervised: t-SNE Feature Space", CLASS_NAMES)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(f"{DIRS['logs']}/supervised_classification_report.txt", 'w') as f:
        f.write(report)
    print(report)

    return dict(history), best_acc, test_loader


# ─────────────────────────────────────────────
# Q2: FixMatch Training
# ─────────────────────────────────────────────
def train_fixmatch(tau=CFG.TAU, lam=CFG.LAMBDA, tag="default"):
    """Single FixMatch run. tau = confidence threshold, lam = unlabeled loss weight."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled_ds, val_ds, test_ds, unlabeled_ds = load_datasets()

    labeled_loader   = DataLoader(labeled_ds,   batch_size=CFG.BATCH_SIZE,
                                   shuffle=True,  num_workers=CFG.NUM_WORKERS,
                                   pin_memory=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_ds,  batch_size=CFG.BATCH_SIZE * CFG.MU,
                                   shuffle=True,  num_workers=CFG.NUM_WORKERS,
                                   pin_memory=True, drop_last=True)
    val_loader       = DataLoader(val_ds,        batch_size=256,
                                   shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    test_loader      = DataLoader(test_ds,       batch_size=256,
                                   shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model     = ResNet9(in_channels=CFG.IN_CHANNELS,
                        num_classes=CFG.NUM_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=CFG.LR, momentum=CFG.MOMENTUM,
                          weight_decay=CFG.WEIGHT_DECAY, nesterov=CFG.NESTEROV)
    steps_per_epoch = max(len(labeled_loader), len(unlabeled_loader))
    scheduler = build_scheduler(optimizer, CFG.EPOCHS,
                                 steps_per_epoch=steps_per_epoch)

    history  = defaultdict(list)
    best_acc = 0.0

    for epoch in range(1, CFG.EPOCHS + 1):
        model.train()
        total_loss_m = AverageMeter()
        sup_loss_m   = AverageMeter()
        unsup_loss_m = AverageMeter()
        mask_total   = mask_count = 0
        class_conf   = np.zeros(CFG.NUM_CLASSES)
        class_cnt    = np.zeros(CFG.NUM_CLASSES) + 1e-8

        labeled_iter   = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        for _ in range(steps_per_epoch):
            # labeled
            try:   imgs_l, labels_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                imgs_l, labels_l = next(labeled_iter)

            # unlabeled (weak + strong)
            try:   imgs_w, imgs_s = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                imgs_w, imgs_s = next(unlabeled_iter)

            imgs_l, labels_l = imgs_l.to(device), labels_l.to(device)
            imgs_w, imgs_s   = imgs_w.to(device),  imgs_s.to(device)

            optimizer.zero_grad()

            # Supervised loss (Ls)
            out_l  = model(imgs_l)
            loss_s = F.cross_entropy(out_l, labels_l)

            # Pseudo-labels from weak view
            with torch.no_grad():
                q           = F.softmax(model(imgs_w), dim=-1)
                conf, pseudo = q.max(dim=-1)
                mask        = (conf >= tau).float()

            # Unsupervised loss on strong view (Lu)
            out_s  = model(imgs_s)
            loss_u = (F.cross_entropy(out_s, pseudo, reduction='none') * mask).mean()

            loss = loss_s + lam * loss_u
            loss.backward()
            optimizer.step()

            bs = imgs_l.size(0)
            total_loss_m.update(loss.item(), bs)
            sup_loss_m.update(loss_s.item(), bs)
            unsup_loss_m.update(loss_u.item(), bs)
            mask_total += mask.sum().item()
            mask_count += mask.numel()

            for c in range(CFG.NUM_CLASSES):
                cidx = (pseudo == c) & mask.bool()
                if cidx.sum() > 0:
                    class_conf[c] += conf[cidx].sum().item()
                    class_cnt[c]  += cidx.sum().item()

        scheduler.step()
        val_acc    = evaluate(model, val_loader,  device)
        test_acc   = evaluate(model, test_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        mr = mask_total / mask_count if mask_count > 0 else 0.0

        history['total_loss'].append(total_loss_m.avg)
        history['sup_loss'].append(sup_loss_m.avg)
        history['unsup_loss'].append(unsup_loss_m.avg)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        history['mask_ratio'].append(mr)
        history['lr'].append(current_lr)
        history['class_confidence'].append((class_conf / class_cnt).tolist())

        fm_logger.info(
            f"[{tag}] τ={tau} λ={lam} | Epoch {epoch:3d}/{CFG.EPOCHS} | "
            f"Loss: {total_loss_m.avg:.4f} (sup:{sup_loss_m.avg:.4f} "
            f"unsup:{unsup_loss_m.avg:.4f}) | "
            f"Val: {val_acc:.2f}% | Test: {test_acc:.2f}% | Mask: {mr:.3f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(),
                             'best_acc': best_acc},
                            f"{DIRS['checkpoints']}/fixmatch_{tag}_best.pt")

        if epoch % CFG.SAVE_EVERY == 0:
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(),
                             'best_acc': best_acc},
                            f"{DIRS['checkpoints']}/fixmatch_{tag}_epoch{epoch}.pt")

    fm_logger.info(f"[{tag}] Best Test Accuracy: {best_acc:.2f}%")
    return dict(history), best_acc, model, test_loader, device


def run_hyperparameter_sweep():
    """Grid search over CFG.TAU_VALUES × CFG.LAMBDA_VALUES."""
    results_grid  = {}
    all_histories = {}

    print("\n" + "="*60)
    print("Q2: HYPERPARAMETER SWEEP (τ × λ)")
    print(f"     τ values : {CFG.TAU_VALUES}")
    print(f"     λ values : {CFG.LAMBDA_VALUES}")
    print("="*60)

    for tau in CFG.TAU_VALUES:
        for lam in CFG.LAMBDA_VALUES:
            tag = f"tau{tau}_lam{lam}"
            print(f"\n--- Sweep: τ={tau}, λ={lam} ---")
            hist, best_acc, _, _, _ = train_fixmatch(tau=tau, lam=lam, tag=tag)
            results_grid[(tau, lam)] = best_acc
            all_histories[tag] = hist
            save_history(hist, f"{DIRS['logs']}/fixmatch_{tag}_history.json")

    return results_grid, all_histories


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="FixMatch — SPP Assignment 3")
    parser.add_argument("--skip_q1",    action='store_true',
                        help="Skip supervised training (load saved history)")
    parser.add_argument("--skip_sweep", action='store_true',
                        help="Skip hyperparameter sweep; only run default τ,λ")
    args = parser.parse_args()

    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)

    print("\n" + "━"*60)
    print("  FixMatch — Signal Processing in Practice  Assignment 3")
    print("━"*60)
    print(f"  Data dir    : {CFG.DATA_DIR}")
    print(f"  Classes     : {CFG.CLASS_NAMES}")
    print(f"  Epochs      : {CFG.EPOCHS}")
    print(f"  Batch size  : {CFG.BATCH_SIZE}  (×{CFG.MU} unlabeled)")
    print(f"  τ (default) : {CFG.TAU}    λ (default): {CFG.LAMBDA}")
    print(f"  τ sweep     : {CFG.TAU_VALUES}")
    print(f"  λ sweep     : {CFG.LAMBDA_VALUES}")
    print("━"*60)

    # ──────────────────
    # Q1: Supervised
    # ──────────────────
    if not args.skip_q1:
        sup_history, sup_best, _ = train_supervised()
    else:
        json_path = f"{DIRS['logs']}/supervised_history.json"
        with open(json_path) as f:
            sup_history = json.load(f)
        sup_best = max(sup_history['test_acc'])
        print(f"[skip_q1] Loaded supervised history. Best acc: {sup_best:.2f}%")

    # ──────────────────
    # Q2: FixMatch
    # ──────────────────
    # Default run (CFG.TAU, CFG.LAMBDA) — used as the baseline FixMatch result
    print(f"\n--- FixMatch Default Run: τ={CFG.TAU}, λ={CFG.LAMBDA} ---")
    fm_default_hist, fm_default_best, fm_model, test_loader, device = \
        train_fixmatch(tau=CFG.TAU, lam=CFG.LAMBDA, tag="default")
    save_history(fm_default_hist, f"{DIRS['logs']}/fixmatch_default_history.json")
    flat_hist = {k: v for k, v in fm_default_hist.items()
                 if v and not isinstance(v[0], list)}
    pd.DataFrame(flat_hist).to_csv(
        f"{DIRS['logs']}/fixmatch_default_history.csv", index=False)

    # Hyperparameter sweep
    if not args.skip_sweep:
        results_grid, all_histories = run_hyperparameter_sweep()

        best_config = max(results_grid.keys(), key=lambda k: results_grid[k])
        best_tau, best_lam = best_config
        print(f"\nBest config: τ={best_tau}, λ={best_lam} → {results_grid[best_config]:.2f}%")

        plot_hyperparameter_heatmap(
            results_grid, CFG.TAU_VALUES, CFG.LAMBDA_VALUES,
            f"{DIRS['plots_fixmatch']}/hyperparameter_heatmap.png"
        )

        tau_sweep = {
            'taus':     CFG.TAU_VALUES,
            'best_acc': [results_grid.get((t, CFG.LAMBDA), 0) for t in CFG.TAU_VALUES],
        }
        lam_sweep = {
            'lambdas':  CFG.LAMBDA_VALUES,
            'best_acc': [results_grid.get((CFG.TAU, l), 0) for l in CFG.LAMBDA_VALUES],
        }

        # Re-run best config for clean evaluation plots
        print(f"\n--- FixMatch Best Config Re-run: τ={best_tau}, λ={best_lam} ---")
        best_fm_hist_full, best_fm_acc, best_model, test_loader, device = \
            train_fixmatch(tau=best_tau, lam=best_lam, tag="best")
        save_history(best_fm_hist_full, f"{DIRS['logs']}/fixmatch_best_history.json")
    else:
        best_fm_hist_full = fm_default_hist
        best_model        = fm_model
        best_fm_acc       = fm_default_best
        tau_sweep = {'taus': CFG.TAU_VALUES,
                     'best_acc': [fm_default_best] * len(CFG.TAU_VALUES)}
        lam_sweep = {'lambdas': CFG.LAMBDA_VALUES,
                     'best_acc': [fm_default_best] * len(CFG.LAMBDA_VALUES)}

    # ── FixMatch training plots ──
    fm_plot_hist = dict(fm_default_hist)
    fm_plot_hist['tau_sweep']    = tau_sweep
    fm_plot_hist['lambda_sweep'] = lam_sweep
    plot_fixmatch_curves(fm_plot_hist, DIRS['plots_fixmatch'])
    # ── FixMatch training curves (uses total_loss as train_loss for this generic plot) ──
    fm_curve_hist = {
        'train_loss': fm_default_hist['total_loss'],
        'train_acc':  fm_default_hist['test_acc'],   # no separate train acc in FM; use test
        'test_acc':   fm_default_hist['test_acc'],
        'val_acc':    fm_default_hist.get('val_acc', []),
        'lr':         fm_default_hist['lr'],
    }
    plot_training_curves(fm_curve_hist, DIRS['plots_fixmatch'], title_prefix="FixMatch (Q2)")

    # ── Best FixMatch model — evaluation plots ──
    _, y_true, y_pred, probs, feats = evaluate(best_model, test_loader, device, return_preds=True)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
                          f"{DIRS['plots_fixmatch']}/confusion_matrix.png",
                          "FixMatch: Confusion Matrix")
    plot_per_class_accuracy(y_true, y_pred, CLASS_NAMES,
                            f"{DIRS['plots_fixmatch']}/per_class_accuracy.png",
                            "FixMatch: Per-Class Accuracy")
    plot_confidence_histogram(probs, y_true, y_pred,
                              f"{DIRS['viz_predictions']}/fixmatch_confidence.png")
    n_tsne = min(CFG.TSNE_SAMPLES, len(feats))
    idx    = np.random.choice(len(feats), n_tsne, replace=False)
    plot_tsne(feats[idx], np.array(y_true)[idx],
              f"{DIRS['viz_embeddings']}/fixmatch_tsne.png",
              "FixMatch: t-SNE Feature Space", CLASS_NAMES)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(f"{DIRS['logs']}/fixmatch_classification_report.txt", 'w') as f:
        f.write(report)
    print(report)

    # ──────────────────
    # Q3: Analysis
    # ──────────────────
    print("\n" + "="*60)
    print("Q3: GENERATING ANALYSIS PLOTS")
    print("="*60)
    plot_comparison(sup_history, fm_default_hist, best_fm_hist_full, DIRS['plots_analysis'])

    # Final summary
    summary = {
        "config": {
            "data_dir": CFG.DATA_DIR,
            "classes":  CFG.CLASS_NAMES,
            "epochs":   CFG.EPOCHS,
            "batch_size": CFG.BATCH_SIZE,
            "mu": CFG.MU,
            "lr": CFG.LR,
            "default_tau": CFG.TAU,
            "default_lambda": CFG.LAMBDA,
        },
        "results": {
            "supervised_best_acc":      max(sup_history['test_acc']),
            "fixmatch_default_best_acc": fm_default_best,
            "fixmatch_best_acc":         best_fm_acc,
            "gain_over_supervised":      best_fm_acc - max(sup_history['test_acc']),
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(f"{DIRS['logs']}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    r = summary["results"]
    print("\n" + "━"*60)
    print("  ALL DONE!  Results saved in outputs/")
    print(f"  Supervised Best Acc   : {r['supervised_best_acc']:.2f}%")
    print(f"  FixMatch Default Acc  : {r['fixmatch_default_best_acc']:.2f}%")
    print(f"  FixMatch Best Acc     : {r['fixmatch_best_acc']:.2f}%")
    print(f"  Gain (vs supervised)  : {r['gain_over_supervised']:+.2f}%")
    print("━"*60)


if __name__ == "__main__":
    main()