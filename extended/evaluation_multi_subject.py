#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Framework for Multi-Subject Alignment
Provides metrics and benchmarks for assessing alignment quality
and cross-subject generalization performance.

@author: Rolly Maulana Awangga
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alignment_methods import (
    RidgeAlignment, 
    Hyperalignment, 
    ProcrustesAlignment,
    MultiSubjectAlignmentPipeline,
    AlignmentEvaluator
)

class CrossSubjectEvaluator:
    """
    Comprehensive evaluation of cross-subject generalization
    """
    
    def __init__(self):
        self.results = {}
        
    def evaluate_leave_one_subject_out(self, multi_subject_data, alignment_method='ridge', 
                                     alignment_params=None):
        """
        Leave-one-subject-out cross-validation for alignment
        
        Args:
            multi_subject_data: dict {subject_id: fMRI_data}
            alignment_method: Alignment method to use
            alignment_params: Parameters for alignment
            
        Returns:
            results: dict with evaluation metrics
        """
        print(f"Running leave-one-subject-out evaluation with {alignment_method}")
        
        subject_ids = list(multi_subject_data.keys())
        n_subjects = len(subject_ids)
        
        if n_subjects < 3:
            raise ValueError("Need at least 3 subjects for LOSO evaluation")
            
        results = {
            'subject_scores': {},
            'mean_scores': {},
            'alignment_method': alignment_method
        }
        
        all_correlations = []
        all_mse_scores = []
        all_r2_scores = []
        
        for test_subject in subject_ids:
            print(f"Testing on subject: {test_subject}")
            
            # Split data
            train_subjects = {sid: data for sid, data in multi_subject_data.items() 
                            if sid != test_subject}
            test_data = multi_subject_data[test_subject]
            
            # Train alignment on training subjects
            pipeline = MultiSubjectAlignmentPipeline(
                alignment_method=alignment_method,
                alignment_params=alignment_params or {}
            )
            
            aligned_train_data, _ = pipeline.fit_transform(train_subjects)
            
            # Align test subject
            aligned_test_data = pipeline.transform_new_subject(test_data, test_subject)
            
            # Evaluate alignment quality
            scores = self._evaluate_single_subject(
                test_data, aligned_test_data, aligned_train_data
            )
            
            results['subject_scores'][test_subject] = scores
            all_correlations.append(scores['correlation'])
            all_mse_scores.append(scores['mse'])
            all_r2_scores.append(scores['r2'])
            
            print(f"  Correlation: {scores['correlation']:.4f}")
            print(f"  MSE: {scores['mse']:.4f}")
            print(f"  R²: {scores['r2']:.4f}")
            
        # Calculate mean scores
        results['mean_scores'] = {
            'correlation': np.mean(all_correlations),
            'correlation_std': np.std(all_correlations),
            'mse': np.mean(all_mse_scores),
            'mse_std': np.std(all_mse_scores),
            'r2': np.mean(all_r2_scores),
            'r2_std': np.std(all_r2_scores)
        }
        
        print(f"\nMean correlation: {results['mean_scores']['correlation']:.4f} ± {results['mean_scores']['correlation_std']:.4f}")
        print(f"Mean R²: {results['mean_scores']['r2']:.4f} ± {results['mean_scores']['r2_std']:.4f}")
        
        return results
        
    def _evaluate_single_subject(self, original_data, aligned_data, reference_data):
        """Evaluate alignment for a single subject"""
        
        # Calculate correlation with reference (mean of other subjects)
        reference_mean = np.mean(list(reference_data.values()), axis=0)
        
        # Ensure same number of timepoints
        min_timepoints = min(aligned_data.shape[0], reference_mean.shape[0])
        aligned_sub = aligned_data[:min_timepoints]
        reference_sub = reference_mean[:min_timepoints]
        
        # Calculate metrics
        correlation = np.corrcoef(aligned_sub.flatten(), reference_sub.flatten())[0, 1]
        mse = mean_squared_error(reference_sub, aligned_sub)
        r2 = r2_score(reference_sub, aligned_sub)
        
        return {
            'correlation': correlation if not np.isnan(correlation) else 0.0,
            'mse': mse,
            'r2': r2 if not np.isnan(r2) else 0.0
        }
        
    def compare_alignment_methods(self, multi_subject_data, methods=None):
        """
        Compare multiple alignment methods using LOSO
        
        Args:
            multi_subject_data: dict {subject_id: fMRI_data}
            methods: List of methods to compare
            
        Returns:
            comparison_results: DataFrame with results
        """
        if methods is None:
            methods = ['ridge', 'hyperalignment', 'procrustes']
            
        print(f"Comparing alignment methods: {methods}")
        
        results = []
        
        for method in methods:
            print(f"\n--- Evaluating {method} ---")
            
            try:
                # Set default parameters
                if method == 'ridge':
                    params = {'alpha': 'auto', 'normalize': True}
                elif method == 'hyperalignment':
                    params = {'max_iterations': 5, 'normalize': True}
                elif method == 'procrustes':
                    params = {'normalize': True}
                else:
                    params = {}
                    
                # Run evaluation
                method_results = self.evaluate_leave_one_subject_out(
                    multi_subject_data, method, params
                )
                
                # Store results
                result_row = {
                    'method': method,
                    'mean_correlation': method_results['mean_scores']['correlation'],
                    'std_correlation': method_results['mean_scores']['correlation_std'],
                    'mean_r2': method_results['mean_scores']['r2'],
                    'std_r2': method_results['mean_scores']['r2_std'],
                    'mean_mse': method_results['mean_scores']['mse'],
                    'std_mse': method_results['mean_scores']['mse_std']
                }
                
                results.append(result_row)
                
            except Exception as e:
                print(f"Error with {method}: {str(e)}")
                result_row = {
                    'method': method,
                    'mean_correlation': np.nan,
                    'std_correlation': np.nan,
                    'mean_r2': np.nan,
                    'std_r2': np.nan,
                    'mean_mse': np.nan,
                    'std_mse': np.nan
                }
                results.append(result_row)
                
        return pd.DataFrame(results)

class ReconstructionEvaluator:
    """
    Evaluate reconstruction quality with alignment
    """
    
    def __init__(self):
        self.baseline_scores = {}
        self.aligned_scores = {}
        
    def evaluate_reconstruction_improvement(self, multi_subject_data, 
                                          reconstruction_targets=None,
                                          alignment_method='ridge'):
        """
        Evaluate how alignment improves reconstruction quality
        
        Args:
            multi_subject_data: dict {subject_id: fMRI_data}
            reconstruction_targets: dict {subject_id: target_data} (optional)
            alignment_method: Alignment method to use
            
        Returns:
            improvement_metrics: dict with improvement scores
        """
        print(f"Evaluating reconstruction improvement with {alignment_method}")
        
        if reconstruction_targets is None:
            # Use synthetic targets based on shared signal
            reconstruction_targets = self._generate_synthetic_targets(multi_subject_data)
            
        # Evaluate baseline (no alignment)
        baseline_scores = self._evaluate_reconstruction_baseline(
            multi_subject_data, reconstruction_targets
        )
        
        # Evaluate with alignment
        aligned_scores = self._evaluate_reconstruction_aligned(
            multi_subject_data, reconstruction_targets, alignment_method
        )
        
        # Calculate improvement
        improvement = {
            'baseline_r2': baseline_scores['mean_r2'],
            'aligned_r2': aligned_scores['mean_r2'],
            'r2_improvement': aligned_scores['mean_r2'] - baseline_scores['mean_r2'],
            'baseline_correlation': baseline_scores['mean_correlation'],
            'aligned_correlation': aligned_scores['mean_correlation'],
            'correlation_improvement': aligned_scores['mean_correlation'] - baseline_scores['mean_correlation'],
            'relative_improvement': (aligned_scores['mean_r2'] - baseline_scores['mean_r2']) / (baseline_scores['mean_r2'] + 1e-10)
        }
        
        print(f"R² improvement: {improvement['r2_improvement']:.4f}")
        print(f"Correlation improvement: {improvement['correlation_improvement']:.4f}")
        print(f"Relative improvement: {improvement['relative_improvement']:.2%}")
        
        return improvement
        
    def _generate_synthetic_targets(self, multi_subject_data):
        """Generate synthetic reconstruction targets"""
        targets = {}
        
        # Use PCA to find shared components
        from sklearn.decomposition import PCA
        
        all_data = np.vstack(list(multi_subject_data.values()))
        pca = PCA(n_components=10)
        shared_components = pca.fit_transform(all_data)
        
        # Split back to subjects
        start_idx = 0
        for subject_id, data in multi_subject_data.items():
            end_idx = start_idx + data.shape[0]
            targets[subject_id] = shared_components[start_idx:end_idx]
            start_idx = end_idx
            
        return targets
        
    def _evaluate_reconstruction_baseline(self, multi_subject_data, targets):
        """Evaluate reconstruction without alignment"""
        
        all_r2_scores = []
        all_correlations = []
        
        for subject_id, fmri_data in multi_subject_data.items():
            target_data = targets[subject_id]
            
            # Simple linear regression as baseline
            min_samples = min(fmri_data.shape[0], target_data.shape[0])
            X = fmri_data[:min_samples]
            y = target_data[:min_samples]
            
            # Cross-validation
            reg = LinearRegression()
            cv_scores = cross_val_score(reg, X, y, cv=5, scoring='r2')
            
            r2_score = np.mean(cv_scores)
            
            # Calculate correlation
            reg.fit(X, y)
            y_pred = reg.predict(X)
            correlation = np.corrcoef(y.flatten(), y_pred.flatten())[0, 1]
            
            all_r2_scores.append(r2_score)
            all_correlations.append(correlation if not np.isnan(correlation) else 0.0)
            
        return {
            'mean_r2': np.mean(all_r2_scores),
            'mean_correlation': np.mean(all_correlations)
        }
        
    def _evaluate_reconstruction_aligned(self, multi_subject_data, targets, alignment_method):
        """Evaluate reconstruction with alignment"""
        
        # Apply alignment
        pipeline = MultiSubjectAlignmentPipeline(alignment_method=alignment_method)
        aligned_data, _ = pipeline.fit_transform(multi_subject_data)
        
        all_r2_scores = []
        all_correlations = []
        
        for subject_id, fmri_data in aligned_data.items():
            target_data = targets[subject_id]
            
            # Linear regression on aligned data
            min_samples = min(fmri_data.shape[0], target_data.shape[0])
            X = fmri_data[:min_samples]
            y = target_data[:min_samples]
            
            # Cross-validation
            reg = LinearRegression()
            cv_scores = cross_val_score(reg, X, y, cv=5, scoring='r2')
            
            r2_score = np.mean(cv_scores)
            
            # Calculate correlation
            reg.fit(X, y)
            y_pred = reg.predict(X)
            correlation = np.corrcoef(y.flatten(), y_pred.flatten())[0, 1]
            
            all_r2_scores.append(r2_score)
            all_correlations.append(correlation if not np.isnan(correlation) else 0.0)
            
        return {
            'mean_r2': np.mean(all_r2_scores),
            'mean_correlation': np.mean(all_correlations)
        }

def run_comprehensive_evaluation():
    """Run comprehensive evaluation of alignment methods"""
    print("COMPREHENSIVE MULTI-SUBJECT ALIGNMENT EVALUATION")
    print("=" * 60)
    
    # Generate synthetic multi-subject data
    def generate_test_data():
        np.random.seed(42)
        n_subjects = 5
        n_timepoints = 80
        n_voxels = 100
        
        # Shared signal
        shared_signal = np.random.randn(n_timepoints, 20)
        shared_weights = np.random.randn(20, n_voxels)
        ground_truth = shared_signal @ shared_weights
        
        multi_subject_data = {}
        for i in range(n_subjects):
            # Subject-specific transformation
            transform = np.random.randn(n_voxels, n_voxels) * 0.1 + np.eye(n_voxels)
            noise = np.random.randn(n_timepoints, n_voxels) * 0.1
            
            subject_data = ground_truth @ transform + noise
            multi_subject_data[f'subject_{i}'] = subject_data
            
        return multi_subject_data
    
    # Generate test data
    test_data = generate_test_data()
    print(f"Generated test data with {len(test_data)} subjects")
    
    # Cross-subject evaluation
    print("\n1. CROSS-SUBJECT GENERALIZATION EVALUATION")
    print("-" * 40)
    
    evaluator = CrossSubjectEvaluator()
    comparison_df = evaluator.compare_alignment_methods(test_data)
    
    print("\nComparison Results:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Reconstruction evaluation
    print("\n2. RECONSTRUCTION IMPROVEMENT EVALUATION")
    print("-" * 40)
    
    recon_evaluator = ReconstructionEvaluator()
    
    for method in ['ridge', 'hyperalignment']:
        print(f"\n--- {method.upper()} ---")
        try:
            improvement = recon_evaluator.evaluate_reconstruction_improvement(
                test_data, alignment_method=method
            )
            
            print(f"R² improvement: {improvement['r2_improvement']:.4f}")
            print(f"Relative improvement: {improvement['relative_improvement']:.2%}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

if __name__ == '__main__':
    run_comprehensive_evaluation()
