import os
from time import time
from numba import njit
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .dct_1d import DCT1D


class DCT2D:
    def __init__(self, signal: np.ndarray):
        self._signal = signal
        self._N = signal.shape[0]

        dummy_signal = np.zeros(self._N)
        self.dct1d = DCT1D(dummy_signal)
        self.D = self.dct1d.get_dct_matrix()

    def get_basis(self, u: int, v: int) -> np.ndarray:
        phi_u = self.D[u]
        phi_v = self.D[v]
        return np.outer(phi_u, phi_v)
    
    def dct2(self, X: np.ndarray) -> np.ndarray:
        return self.D @ X @ self.D.T

    def idct2(self, C: np.ndarray) -> np.ndarray:
        return self.D.T @ C @ self.D
    
    def reconstruction_error(self, X: np.ndarray) -> float:

        C = self.dct2(X)
        X_rec = self.idct2(C)
        return np.linalg.norm(X - X_rec, ord="fro")
    
    def energy_map(self, C: np.ndarray) -> np.ndarray:
        return C ** 2

    def energy_compaction_ratio(self, C: np.ndarray, k: int) -> float:
        total_energy = np.sum(C ** 2)
        low_freq_energy = np.sum(C[:k, :k] ** 2)
        return low_freq_energy / total_energy

    def visualize_basis(self, show: bool = True, save_path: Optional[str] = None):

        fig, axes = plt.subplots(8, 8, figsize=(10, 10))

        for u in range(8):
            for v in range(8):
                ax = axes[u, v]
                Phi_uv = self.get_basis(u, v)

                ax.imshow(Phi_uv, cmap="gray")
                ax.axis("off")
                ax.set_title(f"({u},{v})", fontsize=8)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()

