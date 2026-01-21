import numpy as np
from scipy.fftpack import dctn, idctn
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

class JPEGCompressionPipeline:
    
    def __init__(self, image: np.ndarray, block_size: int = 8):
        
        self.original_image = image.astype(np.float64)
        self.block_size = block_size
        self.shape = image.shape
        
        self.padded_image, self.pad_h, self.pad_w = self._pad_image()
        self.padded_shape = self.padded_image.shape
        
        if len(self.padded_shape) >= 2:
            if self.padded_shape[0] % block_size != 0 or self.padded_shape[1] % block_size != 0:
                raise ValueError(
                    f"Padded image shape {self.padded_shape} is not divisible by block_size {block_size}. "
                    f"Original shape: {self.shape}, Padding: ({self.pad_h}, {self.pad_w})"
                )
        
        self.n_blocks_h = self.padded_shape[0] // block_size
        self.n_blocks_w = self.padded_shape[1] // block_size
        
        self.dct_coeffs = None
        self.quantized_coeffs = None
        self.reconstructed_image = None
        
    def _pad_image(self) -> Tuple[np.ndarray, int, int]:
        if len(self.shape) == 2:
            h, w = self.shape
        else:
            h, w = self.shape[:2]
            
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        
        if len(self.shape) == 2:  # Grayscale
            padded = np.pad(self.original_image, 
                          ((0, pad_h), (0, pad_w)), 
                          mode='edge')
        else:  # RGB
            padded = np.pad(self.original_image,
                          ((0, pad_h), (0, pad_w), (0, 0)),
                          mode='edge')
        
        return padded, pad_h, pad_w
    
    def _unpad_image(self, image: np.ndarray) -> np.ndarray:
        if len(self.shape) == 2:
            h, w = self.shape
            return image[:h, :w]
        else:
            h, w = self.shape[:2]
            return image[:h, :w, :]
    
    def compute_block_dct(self) -> np.ndarray:
        bs = self.block_size
        
        if len(self.padded_shape) == 2:  # Grayscale
            dct_coeffs = np.zeros_like(self.padded_image)
            
            for i in range(self.n_blocks_h):
                for j in range(self.n_blocks_w):
                    block = self.padded_image[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
                    
                    # Verify block shape
                    if block.shape != (bs, bs):
                        raise ValueError(f"Invalid block shape: {block.shape} at position ({i}, {j})")
                    
                    dct_block = dctn(block, type=2, norm='ortho')
                    dct_coeffs[i*bs:(i+1)*bs, j*bs:(j+1)*bs] = dct_block
        
        else: 
            dct_coeffs = np.zeros_like(self.padded_image)
            
            for c in range(self.padded_shape[2]):
                for i in range(self.n_blocks_h):
                    for j in range(self.n_blocks_w):
                        block = self.padded_image[i*bs:(i+1)*bs, j*bs:(j+1)*bs, c]
                        
                        # Verify block shape
                        if block.shape != (bs, bs):
                            raise ValueError(f"Invalid block shape: {block.shape} at position ({i}, {j}, {c})")
                        
                        dct_block = dctn(block, type=2, norm='ortho')
                        dct_coeffs[i*bs:(i+1)*bs, j*bs:(j+1)*bs, c] = dct_block
        
        self.dct_coeffs = dct_coeffs
        return dct_coeffs
    
    def reconstruct_from_dct(self, coeffs: np.ndarray = None) -> np.ndarray:
        if coeffs is None:
            coeffs = self.dct_coeffs
        
        bs = self.block_size
        reconstructed = np.zeros_like(coeffs)
        
        if len(self.padded_shape) == 2:  # Grayscale
            for i in range(self.n_blocks_h):
                for j in range(self.n_blocks_w):
                    dct_block = coeffs[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
                    block = idctn(dct_block, type=2, norm='ortho')
                    reconstructed[i*bs:(i+1)*bs, j*bs:(j+1)*bs] = block
        
        else:  # RGB
            for c in range(self.padded_shape[2]):
                for i in range(self.n_blocks_h):
                    for j in range(self.n_blocks_w):
                        dct_block = coeffs[i*bs:(i+1)*bs, j*bs:(j+1)*bs, c]
                        block = idctn(dct_block, type=2, norm='ortho')
                        reconstructed[i*bs:(i+1)*bs, j*bs:(j+1)*bs, c] = block
        
        self.reconstructed_image = self._unpad_image(reconstructed)
        return self.reconstructed_image
    
    def create_quantization_matrix(self, s: float) -> np.ndarray:
        bs = self.block_size
        u, v = np.meshgrid(range(bs), range(bs), indexing='ij')
        Q = 1 + s * (u + v)
        return Q
    
    def quantize(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.dct_coeffs is None:
            self.compute_block_dct()
        
        Q = self.create_quantization_matrix(s)
        bs = self.block_size
        
        quantized = np.zeros_like(self.dct_coeffs)
        dequantized = np.zeros_like(self.dct_coeffs)
        
        if len(self.padded_shape) == 2:  # Grayscale
            for i in range(self.n_blocks_h):
                for j in range(self.n_blocks_w):
                    block = self.dct_coeffs[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
                    q_block = np.round(block / Q)
                    dq_block = q_block * Q
                    
                    quantized[i*bs:(i+1)*bs, j*bs:(j+1)*bs] = q_block
                    dequantized[i*bs:(i+1)*bs, j*bs:(j+1)*bs] = dq_block
        
        else:  # RGB
            for c in range(self.padded_shape[2]):
                for i in range(self.n_blocks_h):
                    for j in range(self.n_blocks_w):
                        block = self.dct_coeffs[i*bs:(i+1)*bs, j*bs:(j+1)*bs, c]
                        q_block = np.round(block / Q)
                        dq_block = q_block * Q
                        
                        quantized[i*bs:(i+1)*bs, j*bs:(j+1)*bs, c] = q_block
                        dequantized[i*bs:(i+1)*bs, j*bs:(j+1)*bs, c] = dq_block
        
        self.quantized_coeffs = quantized
        return quantized, dequantized
    
    def compress_and_reconstruct(self, s: float) -> np.ndarray:
        if self.dct_coeffs is None:
            self.compute_block_dct()
        
        _, dequantized = self.quantize(s)
        reconstructed = self.reconstruct_from_dct(dequantized)
        
        return reconstructed
    
    def compute_psnr(self, reconstructed: np.ndarray = None) -> float:
        if reconstructed is None:
            reconstructed = self.reconstructed_image
        
        mse = np.mean((self.original_image - reconstructed) ** 2)
        
        if mse < 1e-10:  # Avoid log(0)
            return 100.0
        
        max_pixel = 255.0 if self.original_image.max() > 1 else 1.0
        psnr = 10 * np.log10(max_pixel ** 2 / mse)
        
        return psnr
    
    def compute_sparsity(self, quantized: np.ndarray = None) -> float:
        if quantized is None:
            quantized = self.quantized_coeffs
        
        total = quantized.size
        zeros = np.sum(quantized == 0)
        
        return zeros / total
    
    def compute_max_reconstruction_error(self) -> float:
        if self.reconstructed_image is None:
            self.reconstruct_from_dct()
        
        error = np.abs(self.original_image - self.reconstructed_image)
        return np.max(error)
    
    def analyze_energy_compaction(self, region: str = 'topleft') -> Dict[int, float]:
        if self.dct_coeffs is None:
            self.compute_block_dct()
        
        bs = self.block_size
        energy_fractions = {}
        
        # Collect all blocks
        all_blocks = []
        
        if len(self.padded_shape) == 2:  # Grayscale
            for i in range(self.n_blocks_h):
                for j in range(self.n_blocks_w):
                    block = self.dct_coeffs[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
                    all_blocks.append(block)
        else:  # RGB - average over channels
            for i in range(self.n_blocks_h):
                for j in range(self.n_blocks_w):
                    block = self.dct_coeffs[i*bs:(i+1)*bs, j*bs:(j+1)*bs, :]
                    # Average energy across channels
                    block_avg = np.mean(block, axis=2)
                    all_blocks.append(block_avg)
        
        # Compute energy fractions for each K
        for K in range(1, bs + 1):
            fractions = []
            
            for block in all_blocks:
                E_total = np.sum(block ** 2)
                
                if E_total < 1e-10:  # Skip empty blocks
                    continue
                
                if region == 'topleft':
                    # Top-left K×K square 
                    E_K = np.sum(block[:K, :K] ** 2)
                elif region == 'bottomright':
                    # Bottom-right K×K square 
                    E_K = np.sum(block[-K:, -K:] ** 2)
                else:
                    raise ValueError("region must be 'topleft' or 'bottomright'")
                
                fractions.append(E_K / E_total)
            
            energy_fractions[K] = np.mean(fractions)
        
        return energy_fractions
    
    def plot_energy_compaction(self, save_path: str = None):
        low_freq = self.analyze_energy_compaction('topleft')
        high_freq = self.analyze_energy_compaction('bottomright')
        
        K_values = list(low_freq.keys())
        low_values = [low_freq[k] for k in K_values]
        high_values = [high_freq[k] for k in K_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(K_values, low_values, 'o-', linewidth=2, markersize=8,
                label='Low frequency (top-left K×K)')
        plt.plot(K_values, high_values, 's-', linewidth=2, markersize=8,
                label='High frequency (bottom-right K×K)')
        
        plt.xlabel('K (subband size)', fontsize=12)
        plt.ylabel('Average Energy Fraction E_K/E_total', fontsize=12)
        plt.title('Energy Compaction in DCT Coefficients', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.ylim([0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        return plt.gcf()