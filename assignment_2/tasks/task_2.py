from src.dct_2d import DCT2D

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def run_task_2(input_path:str , save_dir: Optional[str] ="op_files/task_2"):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    signal = np.ones(8)
    # print(signal.shape)
    dct = DCT2D(signal=signal)

    dct.visualize_basis(save_path=save_dir)
    
    assert input_path is not None, "Please enter image path"

    img = np.array(Image.open(input_path)).astype(np.float64)
    print(img.shape)

    X = img[:8, :8]
    X = X / 255.0

    dct2d = DCT2D(X)

    C = dct2d.dct2(X)
    X_rec = dct2d.idct2(C)

    error = np.linalg.norm(X - X_rec, ord="fro")
    print("Reconstruction error (Frobenius norm):", error)

    energy_ratio = dct2d.energy_compaction_ratio(C, k=2)
    print("Energy in top-left 2x2 coefficients:", energy_ratio)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(X, cmap="gray")
    axes[0].set_title("Original 8×8 Block")
    axes[0].axis("off")

    axes[1].imshow(np.log(np.abs(C) + 1e-6), cmap="gray")
    axes[1].set_title("Log |DCT Coefficients|")
    axes[1].axis("off")

    axes[2].imshow(X_rec, cmap="gray")
    axes[2].set_title("Reconstructed Block")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruct.png")
    plt.show()
