#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Script: Multi-Subject Alignment with Real Data
Production-ready script for running multi-subject functional alignment
with digit69_28x28.mat data.

@author: Rolly Maulana Awangga
"""

import numpy as np
import sys
import os
import time
import pickle
import json
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extended'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from extended.real_data_adapter import RealDataAdapter
from extended.alignment_methods import (
    RidgeAlignment,
    Hyperalignment,
    MultiSubjectAlignmentPipeline,
    compare_alignment_methods
)
from extended.multi_subject_dgmm import MultiSubjectDGMM
from extended.evaluation_multi_subject import CrossSubjectEvaluator

def save_alignment_results(aligned_data, alignment_metrics, loso_results,
                          alignment_method, output_dir='./outputs'):
    """
    Save alignment results to files for modular processing

    Args:
        aligned_data: dict {subject_id: aligned_fMRI_data}
        alignment_metrics: dict with alignment metrics
        loso_results: dict with LOSO evaluation results
        alignment_method: str, alignment method used
        output_dir: str, output directory

    Returns:
        output_files: dict with paths to saved files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"alignment_{alignment_method}_{timestamp}"

    output_files = {}

    # 1. Save aligned fMRI data (numpy format for easy loading)
    aligned_data_file = os.path.join(output_dir, f"{base_name}_aligned_data.npz")
    np.savez_compressed(aligned_data_file, **aligned_data)
    output_files['aligned_data'] = aligned_data_file
    print(f"✓ Saved aligned data: {aligned_data_file}")

    # 2. Save alignment metrics (JSON format for readability)
    metrics_file = os.path.join(output_dir, f"{base_name}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            'alignment_metrics': alignment_metrics,
            'loso_results': loso_results,
            'alignment_method': alignment_method,
            'timestamp': timestamp,
            'data_shapes': {sid: data.shape for sid, data in aligned_data.items()}
        }, f, indent=2, default=str)
    output_files['metrics'] = metrics_file
    print(f"✓ Saved metrics: {metrics_file}")

    # 3. Save pipeline object (pickle format for reuse)
    pipeline_file = os.path.join(output_dir, f"{base_name}_pipeline.pkl")
    # Note: pipeline object will be saved separately
    output_files['pipeline'] = pipeline_file

    # 4. Create summary file
    summary_file = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Multi-Subject Alignment Results\n")
        f.write(f"{'='*40}\n")
        f.write(f"Method: {alignment_method}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of subjects: {len(aligned_data)}\n")
        f.write(f"\nData shapes:\n")
        for subject_id, data in aligned_data.items():
            f.write(f"  {subject_id}: {data.shape}\n")
        f.write(f"\nAlignment metrics:\n")
        for metric, value in alignment_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write(f"\nLOSO evaluation:\n")
        f.write(f"  Mean correlation: {loso_results['mean_scores']['correlation']:.4f} ± {loso_results['mean_scores']['correlation_std']:.4f}\n")
        f.write(f"  Mean R²: {loso_results['mean_scores']['r2']:.4f} ± {loso_results['mean_scores']['r2_std']:.4f}\n")
        f.write(f"\nFiles generated:\n")
        for file_type, file_path in output_files.items():
            f.write(f"  {file_type}: {os.path.basename(file_path)}\n")
    output_files['summary'] = summary_file
    print(f"✓ Saved summary: {summary_file}")

    return output_files

def load_alignment_results(aligned_data_file):
    """
    Load alignment results from saved files

    Args:
        aligned_data_file: path to .npz file with aligned data

    Returns:
        aligned_data: dict {subject_id: aligned_fMRI_data}
    """
    print(f"Loading alignment results from: {aligned_data_file}")

    # Load aligned data
    loaded = np.load(aligned_data_file)
    aligned_data = {key: loaded[key] for key in loaded.files}

    print(f"✓ Loaded {len(aligned_data)} subjects")
    for subject_id, data in aligned_data.items():
        print(f"  {subject_id}: {data.shape}")

    return aligned_data

def main_alignment_pipeline(n_subjects=3, alignment_method='hyperalignment'):
    """
    Main pipeline for multi-subject alignment with real data

    Args:
        n_subjects: Number of subjects to create from real data
        alignment_method: 'ridge', 'hyperalignment', or 'procrustes'
    """
    print("MULTI-SUBJECT ALIGNMENT WITH REAL DATA")
    print("=" * 60)
    print(f"Method: {alignment_method.upper()}")
    print(f"Subjects: {n_subjects}")
    print("=" * 60)

    start_time = time.time()

    try:
        # Step 1: Load and prepare real data
        print("\n1. Loading real data...")
        adapter = RealDataAdapter(data_path='./data')

        # Load original data
        original_data = adapter.load_digit_data()
        print(f"✓ Loaded digit69_28x28.mat")
        print(f"  Training: {original_data['X_train'].shape[0]} samples")
        print(f"  Testing: {original_data['X_test'].shape[0]} samples")
        print(f"  fMRI dimensions: {original_data['Y_train'].shape[1]} voxels")

        # Step 2: Create multi-subject data
        print(f"\n2. Creating multi-subject data...")
        multi_subject_data = adapter.create_multi_subject_data(
            n_subjects=n_subjects,
            split_method='random'
        )

        print(f"✓ Created {len(multi_subject_data)} subjects")
        for subject_id, subject_data in multi_subject_data.items():
            print(f"  {subject_id}: {subject_data['X'].shape[0]} samples")

        # Step 3: Apply alignment
        print(f"\n3. Applying {alignment_method} alignment...")

        # Extract fMRI data for alignment
        fmri_data = {sid: data['Y'] for sid, data in multi_subject_data.items()}

        # Set parameters based on method
        if alignment_method == 'ridge':
            params = {'alpha': 'auto', 'normalize': True}
        elif alignment_method == 'hyperalignment':
            # Use normalize=False to avoid dimension mismatch issues
            params = {'n_components': 50, 'max_iterations': 5, 'normalize': False}
        elif alignment_method == 'procrustes':
            params = {'normalize': True}
        else:
            params = {}

        # Create and run pipeline
        pipeline = MultiSubjectAlignmentPipeline(
            alignment_method=alignment_method,
            alignment_params=params
        )

        aligned_fmri, alignment_metrics = pipeline.fit_transform(fmri_data)

        print(f"✓ Alignment completed")
        print(f"  Alignment metrics:")
        for metric, value in alignment_metrics.items():
            print(f"    {metric}: {value:.4f}")

        # Step 4: Cross-subject evaluation
        print(f"\n4. Cross-subject evaluation...")

        evaluator = CrossSubjectEvaluator()
        loso_results = evaluator.evaluate_leave_one_subject_out(
            fmri_data,
            alignment_method=alignment_method,
            alignment_params=params
        )

        print(f"✓ LOSO evaluation completed")
        print(f"  Mean correlation: {loso_results['mean_scores']['correlation']:.4f} ± {loso_results['mean_scores']['correlation_std']:.4f}")
        print(f"  Mean R²: {loso_results['mean_scores']['r2']:.4f} ± {loso_results['mean_scores']['r2_std']:.4f}")

        # Step 5: Train subject-agnostic model
        print(f"\n5. Training subject-agnostic DGMM...")

        # Use smaller subset for DGMM training
        training_subset = {}
        for subject_id, subject_data in multi_subject_data.items():
            n_samples = min(30, subject_data['X'].shape[0])  # Max 30 samples per subject
            training_subset[subject_id] = {
                'X': subject_data['X'][:n_samples],
                'Y': subject_data['Y'][:n_samples]
            }

        # Adjust parameters for DGMM training
        dgmm_params = params.copy()
        if alignment_method == 'hyperalignment':
            dgmm_params['n_components'] = 30  # Smaller for DGMM training
            dgmm_params['max_iterations'] = 3

        ms_dgmm = MultiSubjectDGMM(
            alignment_method=alignment_method,
            alignment_params=dgmm_params
        )

        ms_dgmm.fit(training_subset)

        print(f"✓ Subject-agnostic DGMM trained")

        # Test prediction
        test_subject = list(training_subset.keys())[0]
        test_fmri = training_subset[test_subject]['Y'][:5]
        predictions = ms_dgmm.predict(test_fmri)

        print(f"  Test prediction shape: {predictions.shape}")

        # Step 6: Save results to files for modular processing
        print(f"\n6. Saving results to files...")

        output_files = save_alignment_results(
            aligned_fmri,
            alignment_metrics,
            loso_results,
            alignment_method
        )

        # Save pipeline object
        pipeline_file = output_files['pipeline']
        with open(pipeline_file, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"✓ Saved pipeline: {pipeline_file}")

        # Save trained DGMM model
        dgmm_file = pipeline_file.replace('_pipeline.pkl', '_dgmm_model.pkl')
        with open(dgmm_file, 'wb') as f:
            pickle.dump(ms_dgmm, f)
        output_files['dgmm_model'] = dgmm_file
        print(f"✓ Saved DGMM model: {dgmm_file}")

        # Step 7: Summary
        elapsed_time = time.time() - start_time

        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Execution time: {elapsed_time:.2f} seconds")
        print(f"Alignment method: {alignment_method}")
        print(f"Number of subjects: {n_subjects}")
        print(f"Cross-subject correlation: {loso_results['mean_scores']['correlation']:.4f}")
        print(f"Subject-agnostic model: Ready for new subjects")

        print(f"\n📁 OUTPUT FILES:")
        for file_type, file_path in output_files.items():
            print(f"  {file_type}: {os.path.basename(file_path)}")

        return {
            'multi_subject_data': multi_subject_data,
            'aligned_fmri': aligned_fmri,
            'alignment_metrics': alignment_metrics,
            'loso_results': loso_results,
            'ms_dgmm': ms_dgmm,
            'pipeline': pipeline,
            'output_files': output_files
        }

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_all_methods(n_subjects=3):
    """Compare all alignment methods"""
    print("\nCOMPARING ALL ALIGNMENT METHODS")
    print("=" * 60)

    # Load data
    adapter = RealDataAdapter()
    multi_subject_data = adapter.create_multi_subject_data(n_subjects=n_subjects)
    fmri_data = {sid: data['Y'] for sid, data in multi_subject_data.items()}

    # Compare methods
    methods = ['ridge', 'hyperalignment', 'procrustes']
    results = compare_alignment_methods(fmri_data, methods)

    print("\nCOMPARISON RESULTS:")
    print("-" * 40)

    for method, result in results.items():
        print(f"\n{method.upper()}:")
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            metrics = result['metrics']
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

    return results

def quick_test():
    """Quick test to verify everything works"""
    print("QUICK COMPATIBILITY TEST")
    print("=" * 40)

    try:
        # Test data loading
        adapter = RealDataAdapter()
        data = adapter.load_digit_data()
        print("✓ Data loading works")

        # Test multi-subject creation
        multi_data = adapter.create_multi_subject_data(n_subjects=2)
        print("✓ Multi-subject creation works")

        # Test alignment
        fmri_data = {sid: data['Y'] for sid, data in multi_data.items()}
        ridge = RidgeAlignment(alpha=1.0)

        subject_ids = list(fmri_data.keys())
        ridge.fit(fmri_data[subject_ids[1]], fmri_data[subject_ids[0]])

        print(f"✓ Ridge alignment works (R² = {ridge.alignment_score:.4f})")

        print("\n✅ ALL SYSTEMS OPERATIONAL")
        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == '__main__':
    print("MULTI-SUBJECT FUNCTIONAL ALIGNMENT")
    print("Production System with Real Data")
    print("=" * 70)

    # Quick test first
    if not quick_test():
        print("System check failed. Please verify installation.")
        sys.exit(1)

    print("\n" + "=" * 70)

    # Run main pipeline with different methods
    methods_to_test = ['ridge', 'hyperalignment']

    for method in methods_to_test:
        print(f"\n{'='*70}")
        print(f"RUNNING PIPELINE WITH {method.upper()}")
        print(f"{'='*70}")

        results = main_alignment_pipeline(
            n_subjects=3,
            alignment_method=method
        )

        if results:
            print(f"✅ {method.upper()} pipeline completed successfully")
        else:
            print(f"❌ {method.upper()} pipeline failed")

    # Compare all methods
    print(f"\n{'='*70}")
    compare_all_methods(n_subjects=3)

    print(f"\n{'='*70}")
    print("PRODUCTION SYSTEM READY")
    print("Use the trained models for subject-agnostic fMRI reconstruction")
    print(f"{'='*70}")
