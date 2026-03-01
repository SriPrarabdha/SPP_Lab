import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report , roc_curve, auc

import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * x.size(0)
        preds = out.argmax(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cuda"):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.3f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.3f}")

    return model, history

def plot_history(history, title="Training curves", save_fig=None):
    """
    Plots training history and optionally saves to disk.
    save_fig: Base path/filename (e.g., 'plots/model_1'). 
              Suffixes '_loss.png' and '_acc.png' will be added.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    
    if save_fig:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        plt.savefig(f"{save_fig}/{title}_loss.png")
        print(f"Saved loss plot to: {save_fig}/{title}_loss.png")
    
    plt.show()
    plt.close() 

    # 2. Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()

    if save_fig:
        plt.savefig(f"{save_fig}/{title}_acc.png")
        print(f"Saved accuracy plot to: {save_fig}/{title}_acc.png")

    plt.show()
    plt.close()

def plot_confusion(y_true, y_pred, class_names=("Control", "Dysarthric"), title="", save_fig=None):
    """
    Plots a confusion matrix and optionally saves the figure.
    save_fig: Path/filename where the plot will be saved.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create the display
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues)

    full_title = f"Confusion Matrix: {title}" if title else "Confusion Matrix"
    plt.title(full_title)

    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        path = save_fig if save_fig.endswith('.png') else f"{save_fig}/{title}_cm.png"
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved confusion matrix to: {path}")

    plt.show()
    plt.close(fig) 

    print(f"\nClassification Report for {title}:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def run_experiment(model_class, train_dataset, test_dataset, batch_size=256, epochs=20, lr=1e-3, device="cuda"):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = model_class()

    model, history = train_model(model, train_loader, test_loader, epochs, lr, device)

    plot_history(history, title=model_class.__name__)

    # Final evaluation for confusion matrix
    _, _, preds, labels = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    plot_confusion(labels, preds)

    return model, history


@torch.no_grad()
def plot_roc_auc(model, loader, device="cuda", title="ROC Curve", save_fig=None):
    model.eval()

    probs_list = []
    labels_list = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        # Assuming index 1 is the positive class (Dysarthric)
        probs = F.softmax(logits, dim=1)[:, 1] 

        probs_list.extend(probs.cpu().numpy())
        labels_list.extend(y.numpy())

    fpr, tpr, _ = roc_curve(labels_list, probs_list)
    roc_auc = auc(fpr, tpr)

    # Create Plot
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC area = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC: {roc_auc:.4f})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save Logic
    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        path = save_fig if save_fig.endswith('.png') else f"{save_fig}/{title}_roc.png"
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved ROC curve to: {path}")

    plt.show()
    plt.close(fig) 

    return roc_auc
