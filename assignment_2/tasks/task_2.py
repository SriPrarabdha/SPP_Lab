from src.dct_2d import DCT2D

from typing import Optional
import numpy as np

def run_task_2(save_dir: Optional[str] = None):
    signal = np.ones(8)
    # print(signal.shape)
    dct = DCT2D(signal=signal)

    dct.visualize_basis(show=True)