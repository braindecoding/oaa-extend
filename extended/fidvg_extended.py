#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended FID and Comprehensive Evaluation
Based on original fidvg.py with additional metrics

@author: Rolly Maulana Awangga
"""

import os
import sys
import numpy as np
from pathlib import Path

# GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['THEANO_FLAGS'] = "device=gpu"

# Import existing FID calculation (REUSE)
sys.path.append('../legacy')
from legacy.fidvg_original import calculate_fid

# Import new comprehensive evaluation
from comprehensive_eval import ComprehensiveEvaluator

def calculate_extended_metrics(stimulus_folder, reconstructed_folder, architecture='dgmm'):
    """
    Calculate comprehensive metrics including original FID
    
    Args:
        stimulus_folder: Path to original images
        reconstructed_folder: Path to reconstructed images
        architecture: Model architecture used
    
    Returns:
        dict: All evaluation metrics
    """
    
    # Initialize comprehensive evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Calculate all metrics
    results = evaluator.calculate_all_metrics(stimulus_folder, reconstructed_folder)
    
    # Add architecture information
    results['architecture'] = architecture
    
    return results

def log_extended_results(results, output_file="extended_results.csv"):
    """Log results to CSV with all metrics"""
    
    # Create header if file doesn't exist
    if not Path(output_file).exists():
        header = "Architecture,Latent_Dim,Intermediate_Dim,Batch_Size,Max_Iter," + \
                "FID,PSNR,SSIM,LPIPS,CLIP_Score,MSE,MAE,Uncertainty_Score\n"
        
        with open(output_file, "w") as f:
            f.write(header)
    
    # Format results line
    line = f"{results.get('architecture', 'unknown')}," + \
           f"{results.get('latent_dim', 0)}," + \
           f"{results.get('intermediate_dim', 0)}," + \
           f"{results.get('batch_size', 0)}," + \
           f"{results.get('max_iter', 0)}," + \
           f"{results.get('FID', 0):.4f}," + \
           f"{results.get('PSNR', 0):.4f}," + \
           f"{results.get('SSIM', 0):.4f}," + \
           f"{results.get('LPIPS', 0):.4f}," + \
           f"{results.get('CLIP_Score', 0):.4f}," + \
           f"{results.get('MSE', 0):.4f}," + \
           f"{results.get('MAE', 0):.4f}," + \
           f"{results.get('Uncertainty_Score', 0):.4f}\n"
    
    with open(output_file, "a") as f:
        f.write(line)

def main():
    """Main execution for extended evaluation"""
    if len(sys.argv) < 5:
        print("Usage: python fidvg_extended.py K intermediate_dim batch_size maxiter [architecture]")
        sys.exit(1)
    
    # Parse arguments
    K = int(sys.argv[1])
    intermediate_dim = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    maxiter = int(sys.argv[4])
    architecture = sys.argv[5] if len(sys.argv) > 5 else 'dgmm'
    
    # Setup experiment name
    experiment_name = f"{K}_{intermediate_dim}_{batch_size}_{maxiter}_{architecture}"
    
    # Setup paths
    root_folder = f"results/{experiment_name}/"
    stimulus_folder = root_folder + "stim"
    reconstructed_folder = root_folder + "rec"
    
    print(f"Evaluating experiment: {experiment_name}")
    print(f"Stimulus folder: {stimulus_folder}")
    print(f"Reconstructed folder: {reconstructed_folder}")
    
    # Calculate comprehensive metrics
    try:
        results = calculate_extended_metrics(stimulus_folder, reconstructed_folder, architecture)
        
        # Add experiment parameters
        results.update({
            'latent_dim': K,
            'intermediate_dim': intermediate_dim,
            'batch_size': batch_size,
            'max_iter': maxiter
        })
        
        # Print results
        print("\n=== COMPREHENSIVE EVALUATION RESULTS ===")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        # Log to CSV
        output_file = f"results/comprehensive_results_{architecture}.csv"
        log_extended_results(results, output_file)
        
        print(f"\nResults logged to: {output_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()