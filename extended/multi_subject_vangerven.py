#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Subject fMRI Reconstruction for Vangerven Dataset
Ridge Regression and Hyperalignment for cross-subject generalization
Uses the vangerven digit reconstruction paradigm with multi-subject alignment.

@author: Rolly Maulana Awangga
"""

import numpy as np
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alignment_methods import (
    RidgeAlignment,
    Hyperalignment
)

# Import vangerven reconstruction functions from lib directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
try:
    import vangerven_reconstruction
except ImportError:
    print("Warning: vangerven reconstruction module not found, using placeholder implementation")
    vangerven_reconstruction = None


class MultiSubjectVangerven:
    """
    Multi-Subject fMRI Reconstruction using Vangerven Dataset

    This class implements subject-agnostic fMRI reconstruction using the vangerven
    digit reconstruction paradigm with functional alignment techniques.
    """

    def __init__(self, alignment_method='ridge', alignment_params=None):
        """
        Initialize Multi-Subject Vangerven Reconstruction

        Args:
            alignment_method: 'ridge', 'hyperalignment', or 'none'
            alignment_params: Parameters for alignment method
        """
        self.alignment_method = alignment_method
        self.alignment_params = alignment_params or {}
        self.alignment = None

        # Initialize alignment method
        if alignment_method == 'ridge':
            self.alignment = RidgeAlignment(**self.alignment_params)
        elif alignment_method == 'hyperalignment':
            self.alignment = Hyperalignment(**self.alignment_params)
        elif alignment_method == 'procrustes':
            from alignment_methods import ProcrustesAlignment
            self.alignment = ProcrustesAlignment(**self.alignment_params)
        elif alignment_method == 'none':
            self.alignment = None
        else:
            raise ValueError(f"Unknown alignment method: {alignment_method}")

        self.reconstruction_model = None
        self.fitted = False

    def fit(self, multi_subject_data):
        """
        Fit multi-subject vangerven reconstruction model

        Args:
            multi_subject_data: dict {subject_id: {'X': images, 'Y': fMRI}}
        """

        print(f"Training Multi-Subject Vangerven Reconstruction with {len(multi_subject_data)} subjects")
        print(f"Alignment method: {self.alignment_method}")

        # Step 1: Apply alignment to fMRI data
        if self.alignment is not None:
            print("Applying alignment to fMRI data...")
            fmri_data = {sid: data['Y'] for sid, data in multi_subject_data.items()}
            aligned_fmri = self.alignment.fit_transform(fmri_data)
        else:
            print("No alignment applied")
            aligned_fmri = {sid: data['Y'] for sid, data in multi_subject_data.items()}

        # Step 2: Combine aligned data
        all_images = []
        all_fmri = []

        for subject_id, subject_data in multi_subject_data.items():
            all_images.append(subject_data['X'])
            all_fmri.append(aligned_fmri[subject_id])

        all_images = np.vstack(all_images)
        all_fmri = np.vstack(all_fmri)

        print(f"Combined data: {all_images.shape} images, {all_fmri.shape} fMRI")

        # Step 3: Train vangerven reconstruction model on combined data
        # TODO: Replace with actual vangerven reconstruction training
        print("Training vangerven reconstruction model on aligned multi-subject data...")

        # Placeholder for vangerven reconstruction training
        self.reconstruction_model = {
            'images': all_images,
            'fmri': all_fmri,
            'alignment': self.alignment
        }

        self.fitted = True
        print("Multi-Subject Vangerven Reconstruction training completed")

    def predict(self, new_subject_fmri):
        """
        Predict images from new subject fMRI data

        Args:
            new_subject_fmri: fMRI data for new subject

        Returns:
            predicted_images: Reconstructed images
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        print(f"Predicting for new subject with fMRI shape: {new_subject_fmri.shape}")

        # Step 1: Apply alignment to new subject
        if self.alignment is not None:
            print("Applying alignment to new subject fMRI...")
            aligned_fmri = self.alignment.transform(new_subject_fmri)
        else:
            aligned_fmri = new_subject_fmri

        # TODO: Use trained vangerven reconstruction model for prediction
        print("Predicting with Multi-Subject Vangerven Reconstruction...")

        # Placeholder prediction
        predictions = np.random.random((aligned_fmri.shape[0], 28, 28, 1))

        return predictions

# Example usage and testing
if __name__ == '__main__':
    print("Testing Multi-Subject Vangerven Reconstruction")

    # Test Ridge Alignment
    print("\n--- Testing Ridge Alignment ---")
    np.random.seed(42)
    source_data = np.random.randn(50, 30)
    target_data = source_data @ np.random.randn(30, 30) + 0.1 * np.random.randn(50, 30)

    # Use alignment methods from alignment_methods.py
    ridge_align = RidgeAlignment(alpha=1.0)
    ridge_align.fit(source_data, target_data)
    aligned_data = ridge_align.transform(source_data)

    print(f"Original source shape: {source_data.shape}")
    print(f"Target shape: {target_data.shape}")
    print(f"Aligned shape: {aligned_data.shape}")
    print(f"Alignment score: {ridge_align.alignment_score:.4f}")

    # Test Hyperalignment
    print("\nTesting Hyperalignment...")

    # Generate multi-subject data
    multi_subject_data = {}
    for i in range(3):
        subject_data = np.random.randn(80, 50)
        multi_subject_data[f'subject_{i}'] = subject_data

    hyperalign = Hyperalignment(n_components=20, max_iterations=3)
    aligned_multi = hyperalign.fit_transform(multi_subject_data)

    print(f"Number of subjects: {len(aligned_multi)}")
    for subject_id, data in aligned_multi.items():
        print(f"{subject_id}: {data.shape}")

    # Test Multi-Subject Vangerven Reconstruction
    print("\nTesting Multi-Subject Vangerven Reconstruction...")

    # Create synthetic multi-subject dataset
    training_data = {}
    for i in range(3):
        n_samples = 50
        images = np.random.randn(n_samples, 28, 28, 1)  # Synthetic images
        fmri = np.random.randn(n_samples, 100)  # Synthetic fMRI
        training_data[f'subject_{i}'] = {'X': images, 'Y': fmri}

    # Test with different alignment methods
    for method in ['ridge', 'hyperalignment', 'none']:
        print(f"\n--- Testing with {method} alignment ---")

        try:
            if method == 'ridge':
                params = {'alpha': 1.0, 'normalize': True}
            elif method == 'hyperalignment':
                params = {'n_components': 20, 'max_iterations': 3}
            else:
                params = {}

            ms_vangerven = MultiSubjectVangerven(
                alignment_method=method,
                alignment_params=params
            )

            ms_vangerven.fit(training_data)

            # Test prediction on new subject
            new_fmri = np.random.randn(10, 100)
            predictions = ms_vangerven.predict(new_fmri)

            print(f"Predictions shape: {predictions.shape}")
            print(f"Method {method} completed successfully")

        except Exception as e:
            print(f"Error with {method}: {str(e)}")

    print("\nAll tests completed!")
