import time
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    for cat_x, num_x, targets in dataloader:
        cat_x = cat_x.to(device)
        num_x = num_x.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(cat_x, num_x).view(-1)
        loss = criterion(logits, targets)

        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_targets, all_preds)
    return avg_loss, auc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    for cat_x, num_x, targets in dataloader:
        cat_x = cat_x.to(device)
        num_x = num_x.to(device)
        targets = targets.to(device)

        logits = model(cat_x, num_x).view(-1)
        loss = criterion(logits, targets)

        total_loss += loss.item() * len(targets)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    y_true = np.array(all_targets)
    y_prob = np.array(all_preds)
    auc = roc_auc_score(y_true, y_prob)
    return avg_loss, auc, y_true, y_prob


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    max_epochs: int = 15,
    patience: int = 4,
    device: Optional[torch.device] = None,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Starting training on device: {device}")
    print(f"Epochs: {max_epochs}, Early Stopping Patience: {patience}")

    best_val_auc = -1.0
    best_weights = None
    epochs_no_improve = 0
    history = {
        'train_loss': [],
        'train_auc': [],
        'val_loss': [],
        'val_auc': [],
        'epoch_times': []
    }

    for epoch in range(1, max_epochs + 1):
        start_t = time.time()
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - start_t

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['epoch_times'].append(elapsed)

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_weights, save_path)
            mark = "(*) BEST"
        else:
            epochs_no_improve += 1
            mark = ""

        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} [{elapsed:.1f}s] {mark}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs without improvement.")
            break

    # Reload best model weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {
        'best_val_auc': best_val_auc,
        'history': history,
        'best_model': model
    }
