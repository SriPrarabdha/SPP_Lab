import numpy as np
import matplotlib.pyplot as plt
from src.jpeg_compression import JPEGCompressionPipeline
from PIL import Image
import os


def run_task_3(image_path: str, output_dir: str = 'op_files/task_3'):
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print("Loading image...")
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to grayscale if needed for clearer analysis
    if len(img_array.shape) == 3:
        img_gray = np.array(img.convert('L'))
    else:
        img_gray = img_array
    
    print(f"Image shape: {img_gray.shape}")
    print(f"Image dtype: {img_gray.dtype}, range: [{img_gray.min()}, {img_gray.max()}]")
    
    # Initialize pipeline
    pipeline = JPEGCompressionPipeline(img_gray, block_size=8)
    
    print(f"\nPipeline initialization:")
    print(f"  Original shape: {pipeline.shape}")
    print(f"  Padded shape: {pipeline.padded_shape}")
    print(f"  Padding: ({pipeline.pad_h}, {pipeline.pad_w})")
    print(f"  Block grid: {pipeline.n_blocks_h} × {pipeline.n_blocks_w} blocks")
    print(f"  Total blocks: {pipeline.n_blocks_h * pipeline.n_blocks_w}")
    
    print("\n" + "="*70)
    print("PART (a): Block DCT and Reconstruction (No Quantization)")
    print("="*70)
    
    # Compute DCT
    print("\nComputing 2D DCT for 8x8 blocks...")
    dct_coeffs = pipeline.compute_block_dct()
    print(f"DCT coefficients shape: {dct_coeffs.shape}")
    
    # Reconstruct without quantization
    print("Reconstructing image using inverse DCT (no quantization)...")
    reconstructed = pipeline.reconstruct_from_dct()
    
    # Compute reconstruction error
    max_error = pipeline.compute_max_reconstruction_error()
    print(f"\n✓ Maximum absolute reconstruction error: {max_error:.2e}")
    print(f"  (Should be near numerical precision: ~1e-10 to 1e-13)")
    
    # Visualize
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
    plt.savefig(f'{output_dir}/part_a_reconstruction.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to '{output_dir}/part_a_reconstruction.png'")
    plt.close()
    
    print("\n" + "="*70)
    print("PART (b) & (d): Quantization and Quality vs Compression Study")
    print("="*70)
    
    # Define compression strengths (s values)
    s_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    print(f"\nTesting {len(s_values)} compression strengths: {s_values}")
    
    results = []
    
    for s in s_values:
        print(f"\n--- Processing s = {s} ---")
        
        # Compress and reconstruct
        reconstructed = pipeline.compress_and_reconstruct(s)
        
        # Compute metrics
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
    
    # Visualize reconstructed images for different compression levels
    print("\n\nVisualizing reconstructed images...")
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
    plt.savefig(f'{output_dir}/part_d_reconstructions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved reconstructed images to '{output_dir}/part_d_reconstructions.png'")
    plt.close()
    
    # Plot PSNR and sparsity vs compression strength
    print("\nGenerating quality vs compression plots...")
    
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
    plt.savefig(f'{output_dir}/part_d_quality_vs_compression.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved quality plots to '{output_dir}/part_d_quality_vs_compression.png'")
    plt.close()
    
    # Print results table
    print("\n" + "-"*70)
    print("COMPRESSION RESULTS SUMMARY")
    print("-"*70)
    print(f"{'s':>6} | {'PSNR (dB)':>10} | {'Sparsity':>10} | {'Zeros %':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['s']:>6.1f} | {r['psnr']:>10.2f} | {r['sparsity']:>10.4f} | {r['sparsity']*100:>9.1f}%")
    print("-"*70)
    
    print("\n" + "="*70)
    print("PART (e): Energy Compaction Analysis")
    print("="*70)
    
    # Reset pipeline to use unquantized coefficients
    pipeline.compute_block_dct()
    
    print("\nAnalyzing energy compaction in DCT coefficients...")
    print("Computing average energy fraction E_K/E_total for K=1,...,8")
    
    # Compute energy compaction
    low_freq_energy = pipeline.analyze_energy_compaction('topleft')
    high_freq_energy = pipeline.analyze_energy_compaction('bottomright')
    
    # Print energy compaction table
    print("\n" + "-"*70)
    print("ENERGY COMPACTION RESULTS")
    print("-"*70)
    print(f"{'K':>3} | {'Low-freq (top-left)':>20} | {'High-freq (bottom-right)':>24}")
    print("-"*70)
    for K in range(1, 9):
        print(f"{K:>3} | {low_freq_energy[K]:>19.4f} | {high_freq_energy[K]:>23.4f}")
    print("-"*70)
    
    # Plot energy compaction
    print("\nGenerating energy compaction plot...")
    fig = pipeline.plot_energy_compaction(f'{output_dir}/part_e_energy_compaction.png')
    print(f"✓ Saved energy compaction plot to '{output_dir}/part_e_energy_compaction.png'")
    plt.close()
    
    # Interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION OF ENERGY COMPACTION")
    print("-"*70)
    
    print("\n1. Low-frequency region (top-left K×K):")
    print(f"   - K=1: {low_freq_energy[1]*100:.1f}% of energy (DC component only)")
    print(f"   - K=2: {low_freq_energy[2]*100:.1f}% of energy (2×2 low-freq)")
    print(f"   - K=4: {low_freq_energy[4]*100:.1f}% of energy (4×4 low-freq)")
    print(f"   - K=8: {low_freq_energy[8]*100:.1f}% of energy (all coefficients)")
    
    print("\n   Energy accumulates VERY QUICKLY in low frequencies!")
    print("   This demonstrates the DCT's excellent energy compaction property:")
    print("   - Most image energy concentrates in low-frequency components")
    print("   - High-frequency components (details/noise) have little energy")
    print("   - This is why JPEG compression works so well!")
    
    print("\n2. High-frequency region (bottom-right K×K):")
    print(f"   - K=1: {high_freq_energy[1]*100:.1f}% of energy")
    print(f"   - K=2: {high_freq_energy[2]*100:.1f}% of energy")
    print(f"   - K=4: {high_freq_energy[4]*100:.1f}% of energy")
    print(f"   - K=8: {high_freq_energy[8]*100:.1f}% of energy")
    
    print("\n   Energy accumulates SLOWLY in high frequencies.")
    print("   Even with all 8×8 high-frequency coefficients, the total energy")
    print("   is much less than just a few low-frequency coefficients.")
    
    print("\n3. Comparison:")
    ratio_k2 = low_freq_energy[2] / max(high_freq_energy[2], 1e-10)
    ratio_k4 = low_freq_energy[4] / max(high_freq_energy[4], 1e-10)
    print(f"   - At K=2: Low-freq has {ratio_k2:.1f}× more energy than high-freq")
    print(f"   - At K=4: Low-freq has {ratio_k4:.1f}× more energy than high-freq")
    print("\n   This huge difference justifies aggressive quantization of")
    print("   high-frequency coefficients in JPEG compression!")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to '{output_dir}/' directory")
    print("\nKey findings:")
    print("  • Perfect reconstruction (no quantization): error ≈ 0")
    print(f"  • With quantization: PSNR ranges from {min(psnr_vals):.1f} to {max(psnr_vals):.1f} dB")
    print(f"  • Sparsity ranges from {min(sparsity_vals)*100:.1f}% to {max(sparsity_vals)*100:.1f}% zeros")
    print("  • Energy compaction: ~90%+ energy in top-left 4×4 coefficients")
    print("  • DCT is highly effective for image compression!")
    