import os
from time import time
from numba import njit
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .dct_1d import DCT1D


class DCT2D:
    def __init__(self, signal:np.ndarray):
        self._signal = signal
        self._N = signal.shape[0]

        dummy_signal = np.zeros(self._N)
        self.dct1d = DCT1D(dummy_signal)

        self.D = self.dct1d.get_dct_matrix()

    def get_basis(self, u: int, v: int) -> np.ndarray:

        phi_u = self.D[u]          # (N,)
        phi_v = self.D[v]          # (N,)

        Phi_uv = np.outer(phi_u, phi_v)
        return Phi_uv

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

