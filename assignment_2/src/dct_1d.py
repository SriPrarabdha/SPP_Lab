import os
from time import time
from numba import njit
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


class DCT1D:
    def __init__(self, signal: np.ndarray):
        self._signal = signal
        self._N = signal.shape[0]

    def get_dct_matrix(self) -> np.ndarray:
        N = self._N
        D = np.zeros((N, N), dtype=np.float64)

        for k in range(N):
            if k == 0:
                alpha = np.sqrt(1 / N)
            else:
                alpha = np.sqrt(2 / N)

            for n in range(N):
                D[k, n] = alpha * np.cos(
                    np.pi / N * (n + 0.5) * k
                )
        self._D = D
        return D
    
    def check_orthogonality(self) -> int:
        D = self._D
        I = np.eye(self._N)
        error = np.linalg.norm(D @ D.T - I, ord='fro')
        return error
    
    def viz_dct_matrix(self, show:Optional[str]= None, save_dir:Optional[str] = None):
        D = self._D
        t = np.arange(0, self._N)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for k in range(8):
            ax = axes[k // 4, k % 4]
            ax.plot(t, D[k], linewidth=2)
            ax.set_title(f"$k = {k}$")
            ax.set_xlabel("n")
            ax.set_ylabel(r"$\phi_k[n]$")
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/1d_dct_matrix_viz.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # if save_dir is not None:

        plt.close()

