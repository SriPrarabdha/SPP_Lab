from src.dct_1d import DCT1D

from typing import Optional
import numpy as np
import os

def run_task_1(save_dir: Optional[str] = "op_files/task_1"):
    os.makedirs(save_dir, exist_ok=True)
    signal = np.ones(32)
    # print(signal.shape)
    dct = DCT1D(signal=signal)

    matrix = dct.get_dct_matrix()
    print(matrix.shape)
    # print(matrix[0])

    print(dct.check_orthogonality())
    dct.viz_dct_matrix(save_dir=save_dir)