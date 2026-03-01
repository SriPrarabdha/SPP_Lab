import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .dataset_util import TorgoDataset

def pad_collate(batch):
    """
    Pads variable-length spectrograms in time dimension.
    Input:
        batch = [(feat, label), ...]
        feat shape = (F, T)
    Output:
        feats -> (B, F, T_max)
        labels -> (B,)
    """
    feats, labels = zip(*batch)

    # Find max time length in batch
    max_len = max(f.shape[1] for f in feats)

    padded_feats = []
    for f in feats:
        pad_size = max_len - f.shape[1]
        f_padded = F.pad(f, (0, pad_size))  # pad time dimension
        padded_feats.append(f_padded)

    feats_tensor = torch.stack(padded_feats)
    labels_tensor = torch.tensor(labels)

    return feats_tensor, labels_tensor

def create_clean_dataloaders(train_files, test_files, feature_type, batch_size=256):
    train_ds = TorgoDataset(train_files, feature_type)
    test_ds = TorgoDataset(test_files, feature_type)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate   # ⭐ key fix
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=pad_collate   # ⭐ key fix
    )

    return train_loader, test_loader
