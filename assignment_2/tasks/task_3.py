import numpy as np
import matplotlib.pyplot as plt
from src.jpeg_compression import JPEGCompressionPipeline
from PIL import Image
import os


def run_task_3(image_path: str, output_dir: str = 'op_files/task_3'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    img = Image.open(image_path)
    img_array = np.array(img)
    
    if len(img_array.shape) == 3:
        img_gray = np.array(img.convert('L'))
    else:
        img_gray = img_array
    
    print(f"Image shape: {img_gray.shape}")
    print(f"range: [{img_gray.min()}, {img_gray.max()}]")
    
    pipeline = JPEGCompressionPipeline(img_gray, block_size=8)

    dct_coeffs = pipeline.compute_block_dct()
    print(f"DCT coefficients shape: {dct_coeffs.shape}")

    reconstructed = pipeline.reconstruct_from_dct()
    
    max_error = pipeline.compute_max_reconstruction_error()
    print(f"\nMaximum absolute reconstruction error: {max_error:.2e}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title(f'Reconstructed (no quantization)\nMax error: {max_error:.2e}')
    axes[1].axis('off')
    
    error_img = np.abs(img_gray - reconstructed)
    im = axes[2].imshow(error_img, cmap='hot')
    axes[2].set_title('Absolute Error Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{output_dir}/part_a_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    s_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    print(f"\nTesting {len(s_values)} compression strengths: {s_values}")
    
    results = []
    
    for s in s_values:
        print(f"\n--- Processing s = {s} ---")
        
        reconstructed = pipeline.compress_and_reconstruct(s)
        
        psnr = pipeline.compute_psnr()
        sparsity = pipeline.compute_sparsity()
        
        results.append({
            's': s,
            'psnr': psnr,
            'sparsity': sparsity,
            'reconstructed': reconstructed.copy()
        })
        
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Sparsity (zero fraction): {sparsity:.4f} ({sparsity*100:.2f}%)")
    
    n_images = len(s_values)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        axes[idx].imshow(result['reconstructed'], cmap='gray')
        axes[idx].set_title(
            f"s = {result['s']}\n"
            f"PSNR: {result['psnr']:.2f} dB\n"
            f"Zeros: {result['sparsity']*100:.1f}%",
            fontsize=10
        )
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{output_dir}/part_d_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR vs s
    s_vals = [r['s'] for r in results]
    psnr_vals = [r['psnr'] for r in results]
    sparsity_vals = [r['sparsity'] for r in results]
    
    ax1.plot(s_vals, psnr_vals, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Compression Strength (s)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('Image Quality vs Compression Strength', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Sparsity vs s
    ax2.plot(s_vals, sparsity_vals, 's-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Compression Strength (s)', fontsize=12)
    ax2.set_ylabel('Sparsity (fraction of zeros)', fontsize=12)
    ax2.set_title('Coefficient Sparsity vs Compression Strength', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{output_dir}/part_d_quality_vs_compression.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    pipeline.compute_block_dct()
    
    low_freq_energy = pipeline.analyze_energy_compaction('topleft')
    high_freq_energy = pipeline.analyze_energy_compaction('bottomright')
    
    fig = pipeline.plot_energy_compaction(f'{output_dir}/part_e_energy_compaction.png')
    plt.close()
    
    print("\n1. Low-frequency region (top-left K×K):")
    print(f"K=1: {low_freq_energy[1]*100:.1f}% of energy (DC component only)")
    print(f"K=2: {low_freq_energy[2]*100:.1f}% of energy (2×2 low-freq)")
    print(f"K=4: {low_freq_energy[4]*100:.1f}% of energy (4×4 low-freq)")
    print(f"K=8: {low_freq_energy[8]*100:.1f}% of energy (all coefficients)")
    
    
    print("\n2. High-frequency region (bottom-right K×K):")
    print(f"K=1: {high_freq_energy[1]*100:.1f}% of energy")
    print(f"K=2: {high_freq_energy[2]*100:.1f}% of energy")
    print(f"K=4: {high_freq_energy[4]*100:.1f}% of energy")
    print(f"K=8: {high_freq_energy[8]*100:.1f}% of energy")
    
    
    print("\n3. Comparison:")
    ratio_k2 = low_freq_energy[2] / max(high_freq_energy[2], 1e-10)
    ratio_k4 = low_freq_energy[4] / max(high_freq_energy[4], 1e-10)
    print(f"At K=2: Low-freq has {ratio_k2:.1f}× more energy than high-freq")
    print(f"At K=4: Low-freq has {ratio_k4:.1f}× more energy than high-freq")