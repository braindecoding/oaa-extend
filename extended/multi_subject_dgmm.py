#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Subject DGMM Implementation
Ridge Regression and Hyperalignment for cross-subject generalization

@author: Rolly Maulana Awangga
"""

import numpy as np
import sys
import os

# Add the extended directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alignment_methods import (
    RidgeAlignment,
    Hyperalignment,
    ProcrustesAlignment,
    MultiSubjectAlignmentPipeline,
    AlignmentEvaluator,
    compare_alignment_methods
)

# Import DGMM from lib directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
try:
    from dgmm import DGMM
except ImportError:
    print("Warning: DGMM not found, using placeholder implementation")
    DGMM = None



class MultiSubjectDGMM:
    """Extended DGMM for multi-subject data"""

    def __init__(self, alignment_method='ridge', alignment_params=None):
        """
        Initialize Multi-Subject DGMM

        Args:
            alignment_method: 'ridge', 'hyperalignment', or 'none'
            alignment_params: Parameters for alignment method
        """
        self.alignment_method = alignment_method
        self.alignment_params = alignment_params or {}

        # Initialize alignment
        if alignment_method == 'ridge':
            self.alignment = RidgeAlignment(**self.alignment_params)
        elif alignment_method == 'hyperalignment':
            self.alignment = Hyperalignment(**self.alignment_params)
        elif alignment_method == 'none':
            self.alignment = None
        else:
            raise ValueError(f"Unknown alignment method: {alignment_method}")

        self.dgmm_model = None
        self.fitted = False

    def fit(self, multi_subject_data):
        """
        Fit multi-subject DGMM

        Args:
            multi_subject_data: dict {subject_id: {'X': images, 'Y': fMRI}}
        """

        print(f"Training Multi-Subject DGMM with {len(multi_subject_data)} subjects")
        print(f"Alignment method: {self.alignment_method}")

        # Step 1: Align fMRI data if needed
        if self.alignment is not None:
            aligned_fmri = self._align_subjects(multi_subject_data)
        else:
            aligned_fmri = {sid: data['Y'] for sid, data in multi_subject_data.items()}

        # Step 2: Combine data for training
        all_images = np.vstack([data['X'] for data in multi_subject_data.values()])
        all_fmri = np.vstack(list(aligned_fmri.values()))

        print(f"Combined training data - Images: {all_images.shape}, fMRI: {all_fmri.shape}")

        # Step 3: Train DGMM on combined data
        # TODO: Replace with actual DGMM training from oaavangerven_extended.py
        print("Training DGMM on aligned multi-subject data...")

        # Placeholder for DGMM training
        self.dgmm_model = {
            'images': all_images,
            'fmri': all_fmri,
            'alignment': self.alignment
        }

        self.fitted = True

        return self

    def _align_subjects(self, multi_subject_data):
        """Align fMRI data across subjects"""

        subject_ids = list(multi_subject_data.keys())
        fmri_data = {sid: data['Y'] for sid, data in multi_subject_data.items()}

        if self.alignment_method == 'ridge':
            # Use first subject as reference
            reference_id = subject_ids[0]
            reference_data = fmri_data[reference_id]

            aligned_fmri = {reference_id: reference_data}

            # Align all other subjects to reference
            for subject_id in subject_ids[1:]:
                source_data = fmri_data[subject_id]
                aligned_data = self.alignment.fit_transform(source_data, reference_data)
                aligned_fmri[subject_id] = aligned_data

        elif self.alignment_method == 'hyperalignment':
            aligned_fmri = self.alignment.fit_transform(fmri_data)

        return aligned_fmri

    def predict(self, new_subject_fmri, calibration_data=None):
        """
        Predict for new subject

        Args:
            new_subject_fmri: fMRI data from new subject
            calibration_data: Optional data for alignment

        Returns:
            predictions: Reconstructed images
        """

        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # Align new subject data if needed
        if self.alignment is not None and calibration_data is not None:
            aligned_fmri = self.alignment.transform(new_subject_fmri)
        else:
            aligned_fmri = new_subject_fmri

        # TODO: Use trained DGMM for prediction
        print("Predicting with Multi-Subject DGMM...")

        # Placeholder prediction
        predictions = np.random.random((aligned_fmri.shape[0], 28, 28, 1))

        return predictions

# Example usage and testing
if __name__ == '__main__':
    # Test Ridge Alignment
    print("Testing Ridge Alignment...")

    # Generate synthetic data
    np.random.seed(42)
    source_data = np.random.randn(100, 50)  # 100 samples, 50 voxels
    target_data = source_data @ np.random.randn(50, 50) + 0.1 * np.random.randn(100, 50)

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

    # Test Multi-Subject DGMM
    print("\nTesting Multi-Subject DGMM...")

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

            ms_dgmm = MultiSubjectDGMM(
                alignment_method=method,
                alignment_params=params
            )

            ms_dgmm.fit(training_data)

            # Test prediction on new subject
            new_fmri = np.random.randn(10, 100)
            predictions = ms_dgmm.predict(new_fmri)

            print(f"Predictions shape: {predictions.shape}")
            print(f"Method {method} completed successfully")

        except Exception as e:
            print(f"Error with {method}: {str(e)}")

    print("\nAll tests completed!")