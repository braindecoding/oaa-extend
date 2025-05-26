#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Subject DGMM Implementation
Ridge Regression and Hyperalignment for cross-subject generalization

@author: Rolly Maulana Awangga
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

class RidgeAlignment:
    """Ridge regression for functional alignment between subjects"""
    
    def __init__(self, alpha=1.0):
        """
        Initialize Ridge alignment
        
        Args:
            alpha: Regularization strength
        """
        self.alpha = alpha
        self.alignment_matrix = None
        self.fitted = False
    
    def fit(self, source_data, target_data):
        """
        Fit alignment transformation from source to target subject
        
        Args:
            source_data: fMRI data from source subject (N × voxels)
            target_data: fMRI data from target subject (N × voxels)
        
        Returns:
            self: Fitted alignment object
        """
        
        print(f"Fitting Ridge alignment (alpha={self.alpha})")
        print(f"Source data shape: {source_data.shape}")
        print(f"Target data shape: {target_data.shape}")
        
        # Ridge regression: W = argmin ||target - source*W||² + α||W||²
        ridge = Ridge(alpha=self.alpha, fit_intercept=False)
        ridge.fit(source_data, target_data)
        
        self.alignment_matrix = ridge.coef_.T
        self.fitted = True
        
        print(f"Alignment matrix shape: {self.alignment_matrix.shape}")
        
        return self
    
    def transform(self, source_data):
        """
        Transform source data to target space
        
        Args:
            source_data: Data to transform (N × voxels)
        
        Returns:
            transformed_data: Aligned data (N × voxels)
        """
        
        if not self.fitted:
            raise ValueError("Alignment must be fitted before transform")
        
        transformed_data = source_data @ self.alignment_matrix
        
        print(f"Transformed data shape: {transformed_data.shape}")
        
        return transformed_data
    
    def fit_transform(self, source_data, target_data):
        """Fit alignment and transform source data"""
        return self.fit(source_data, target_data).transform(source_data)

class Hyperalignment:
    """Hyperalignment for multi-subject fMRI data"""
    
    def __init__(self, n_components=None, max_iterations=10):
        """
        Initialize Hyperalignment
        
        Args:
            n_components: Number of components for common space
            max_iterations: Maximum iterations for convergence
        """
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.transformation_matrices = {}
        self.common_space = None
        self.fitted = False
    
    def fit_transform(self, multi_subject_data):
        """
        Fit hyperalignment and return aligned data
        
        Args:
            multi_subject_data: dict {subject_id: fMRI_data}
        
        Returns:
            aligned_data: dict {subject_id: aligned_fMRI_data}
        """
        
        print(f"Fitting Hyperalignment with {len(multi_subject_data)} subjects")
        
        # Step 1: Initialize common space with PCA
        all_data = np.vstack([data for data in multi_subject_data.values()])
        print(f"Combined data shape: {all_data.shape}")
        
        if self.n_components is None:
            self.n_components = min(all_data.shape[1], 100)  # Default to 100 or less
        
        pca = PCA(n_components=self.n_components)
        self.common_space = pca.fit_transform(all_data)
        
        print(f"Common space shape: {self.common_space.shape}")
        
        # Step 2: Iterative alignment
        aligned_data = {}
        
        for iteration in range(self.max_iterations):
            print(f"Hyperalignment iteration {iteration + 1}/{self.max_iterations}")
            
            temp_aligned = {}
            
            # Update transformation matrices for each subject
            for subject_id, data in multi_subject_data.items():
                # Procrustes alignment to common space
                W = self._procrustes_alignment(data, self.common_space)
                self.transformation_matrices[subject_id] = W
                temp_aligned[subject_id] = data @ W
            
            # Update common space as mean of aligned data
            self.common_space = np.mean(list(temp_aligned.values()), axis=0)
            aligned_data = temp_aligned
        
        self.fitted = True
        
        print("Hyperalignment completed")
        
        return aligned_data
    
    def _procrustes_alignment(self, source, target):
        """
        Procrustes alignment between source and target
        
        Args:
            source: Source data matrix
            target: Target data matrix
        
        Returns:
            W: Transformation matrix
        """
        
        # Ensure same number of samples
        min_samples = min(source.shape[0], target.shape[0])
        source_sub = source[:min_samples]
        target_sub = target[:min_samples]
        
        # Procrustes solution: W = U @ V.T where U, S, V = svd(target.T @ source)
        U, _, Vt = np.linalg.svd(target_sub.T @ source_sub, full_matrices=False)
        W = Vt.T @ U.T
        
        return W
    
    def transform(self, new_subject_data, subject_id):
        """
        Transform new subject data using fitted alignment
        
        Args:
            new_subject_data: Data from new subject
            subject_id: ID for the subject
        
        Returns:
            aligned_data: Transformed data
        """
        
        if not self.fitted:
            raise ValueError("Hyperalignment must be fitted before transform")
        
        if subject_id in self.transformation_matrices:
            W = self.transformation_matrices[subject_id]
        else:
            # Create new transformation for unseen subject
            W = self._procrustes_alignment(new_subject_data, self.common_space)
        
        return new_subject_data @ W

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
    
    ridge_align = RidgeAlignment(alpha=1.0)
    ridge_align.fit(source_data, target