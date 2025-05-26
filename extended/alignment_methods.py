#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Functional Alignment Methods for Multi-Subject fMRI Data
Implements Ridge Regression, Hyperalignment, and other alignment techniques
for subject-agnostic fMRI reconstruction models.

@author: Rolly Maulana Awangga
"""

import numpy as np
import scipy.stats as stats
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
import warnings

class BaseAlignment:
    """Base class for all alignment methods"""

    def __init__(self):
        self.fitted = False
        self.transformation_matrices = {}

    def fit(self, source_data, target_data):
        """Fit alignment transformation"""
        raise NotImplementedError

    def transform(self, data, subject_id=None):
        """Transform data using fitted alignment"""
        raise NotImplementedError

    def fit_transform(self, source_data, target_data):
        """Fit and transform in one step"""
        return self.fit(source_data, target_data).transform(source_data)

class RidgeAlignment(BaseAlignment):
    """
    Ridge regression alignment for functional connectivity
    Learns linear transformation W: source -> target
    Minimizes: ||target - source*W||² + α||W||²
    """

    def __init__(self, alpha=1.0, cv_folds=5, normalize=True):
        """
        Initialize Ridge alignment

        Args:
            alpha: Regularization strength (float or 'auto' for CV)
            cv_folds: Number of CV folds if alpha='auto'
            normalize: Whether to normalize data before alignment
        """
        super().__init__()
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.normalize = normalize
        self.alignment_matrix = None
        self.scaler_source = None
        self.scaler_target = None
        self.alignment_score = None

    def fit(self, source_data, target_data):
        """
        Fit Ridge alignment transformation

        Args:
            source_data: Source subject fMRI data (n_samples × n_voxels)
            target_data: Target subject fMRI data (n_samples × n_voxels)

        Returns:
            self: Fitted alignment object
        """
        print(f"Fitting Ridge alignment...")
        print(f"Source shape: {source_data.shape}, Target shape: {target_data.shape}")

        # Validate input
        if source_data.shape[0] != target_data.shape[0]:
            raise ValueError("Source and target must have same number of samples")

        # Normalize data if requested
        if self.normalize:
            self.scaler_source = StandardScaler()
            self.scaler_target = StandardScaler()
            source_norm = self.scaler_source.fit_transform(source_data)
            target_norm = self.scaler_target.fit_transform(target_data)
        else:
            source_norm = source_data
            target_norm = target_data

        # Determine alpha via cross-validation if needed
        if self.alpha == 'auto':
            alphas = np.logspace(-3, 3, 20)
            ridge_cv = RidgeCV(alphas=alphas, cv=self.cv_folds, fit_intercept=False)
            ridge_cv.fit(source_norm, target_norm)
            self.alpha = ridge_cv.alpha_
            print(f"Optimal alpha via CV: {self.alpha:.6f}")

        # Fit Ridge regression
        ridge = Ridge(alpha=self.alpha, fit_intercept=False)
        ridge.fit(source_norm, target_norm)

        self.alignment_matrix = ridge.coef_.T

        # Calculate alignment quality
        aligned_source = source_norm @ self.alignment_matrix
        self.alignment_score = r2_score(target_norm, aligned_source)

        print(f"Alignment matrix shape: {self.alignment_matrix.shape}")
        print(f"Alignment R² score: {self.alignment_score:.4f}")

        self.fitted = True
        return self

    def transform(self, source_data, subject_id=None):
        """
        Transform source data to target space

        Args:
            source_data: Data to transform (n_samples × n_voxels)
            subject_id: Optional subject identifier

        Returns:
            transformed_data: Aligned data
        """
        if not self.fitted:
            raise ValueError("Alignment must be fitted before transform")

        # Apply normalization if used during fitting
        if self.normalize and self.scaler_source is not None:
            source_norm = self.scaler_source.transform(source_data)
        else:
            source_norm = source_data

        # Apply alignment transformation
        aligned_data = source_norm @ self.alignment_matrix

        # Denormalize if needed
        if self.normalize and self.scaler_target is not None:
            aligned_data = self.scaler_target.inverse_transform(aligned_data)

        return aligned_data

class Hyperalignment(BaseAlignment):
    """
    Hyperalignment for multi-subject fMRI data
    Creates a common representational space across subjects
    """

    def __init__(self, n_components=None, max_iterations=10,
                 convergence_threshold=1e-6, normalize=True):
        """
        Initialize Hyperalignment

        Args:
            n_components: Dimensionality of common space (None for auto)
            max_iterations: Maximum iterations for convergence
            convergence_threshold: Convergence criterion
            normalize: Whether to normalize data
        """
        super().__init__()
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.normalize = normalize
        self.common_space = None
        self.scalers = {}
        self.convergence_history = []

    def fit_transform(self, multi_subject_data):
        """
        Fit hyperalignment and return aligned data

        Args:
            multi_subject_data: dict {subject_id: fMRI_data}

        Returns:
            aligned_data: dict {subject_id: aligned_fMRI_data}
        """
        print(f"Fitting Hyperalignment with {len(multi_subject_data)} subjects")

        # Validate and normalize data
        normalized_data = self._prepare_data(multi_subject_data)

        # Initialize common space with PCA
        self._initialize_common_space(normalized_data)

        # Iterative alignment
        aligned_data = self._iterative_alignment(normalized_data)

        # Denormalize if needed
        if self.normalize:
            aligned_data = self._denormalize_data(aligned_data)

        self.fitted = True
        print(f"Hyperalignment converged after {len(self.convergence_history)} iterations")

        return aligned_data

    def _prepare_data(self, multi_subject_data):
        """Prepare and normalize data"""
        normalized_data = {}

        for subject_id, data in multi_subject_data.items():
            if self.normalize:
                scaler = StandardScaler()
                normalized_data[subject_id] = scaler.fit_transform(data)
                self.scalers[subject_id] = scaler
            else:
                normalized_data[subject_id] = data

        return normalized_data

    def _initialize_common_space(self, normalized_data):
        """Initialize common space using PCA"""
        # Concatenate all data
        all_data = np.vstack(list(normalized_data.values()))
        print(f"Combined data shape: {all_data.shape}")

        # Determine number of components
        if self.n_components is None:
            # Use 90% variance or max 200 components
            pca_temp = PCA()
            pca_temp.fit(all_data)
            cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
            self.n_components = min(np.where(cumvar >= 0.9)[0][0] + 1, 200)

        print(f"Using {self.n_components} components for common space")

        # Create initial common space
        pca = PCA(n_components=self.n_components)
        self.common_space = pca.fit_transform(all_data)

        # Reshape to match original structure
        start_idx = 0
        common_by_subject = {}
        for subject_id, data in normalized_data.items():
            end_idx = start_idx + data.shape[0]
            common_by_subject[subject_id] = self.common_space[start_idx:end_idx]
            start_idx = end_idx

        self.common_space = common_by_subject

    def _iterative_alignment(self, normalized_data):
        """Perform iterative alignment"""
        aligned_data = {}
        prev_common = None

        for iteration in range(self.max_iterations):
            print(f"Hyperalignment iteration {iteration + 1}/{self.max_iterations}")

            temp_aligned = {}

            # Update transformation matrices
            for subject_id, data in normalized_data.items():
                common_data = self.common_space[subject_id]
                W = self._procrustes_alignment(data, common_data)
                self.transformation_matrices[subject_id] = W
                temp_aligned[subject_id] = data @ W

            # Update common space
            prev_common = self.common_space.copy()
            self._update_common_space(temp_aligned)

            # Check convergence
            convergence_metric = self._compute_convergence(prev_common, self.common_space)
            self.convergence_history.append(convergence_metric)

            print(f"Convergence metric: {convergence_metric:.8f}")

            if convergence_metric < self.convergence_threshold:
                print(f"Converged at iteration {iteration + 1}")
                break

            aligned_data = temp_aligned

        return aligned_data

    def _procrustes_alignment(self, source, target):
        """Procrustes alignment between source and target"""
        # Ensure same number of samples
        min_samples = min(source.shape[0], target.shape[0])
        source_sub = source[:min_samples]
        target_sub = target[:min_samples]

        # Procrustes solution
        try:
            U, _, Vt = np.linalg.svd(target_sub.T @ source_sub, full_matrices=False)
            W = Vt.T @ U.T
        except np.linalg.LinAlgError:
            # Fallback to identity if SVD fails
            warnings.warn("SVD failed, using identity transformation")
            W = np.eye(source.shape[1])

        return W

    def _update_common_space(self, aligned_data):
        """Update common space as mean of aligned data"""
        # Calculate mean across subjects for each timepoint
        all_subjects = list(aligned_data.keys())
        n_timepoints = aligned_data[all_subjects[0]].shape[0]
        n_features = aligned_data[all_subjects[0]].shape[1]

        new_common = {}
        for subject_id in all_subjects:
            new_common[subject_id] = np.zeros((n_timepoints, n_features))

        # Average across subjects
        for t in range(n_timepoints):
            timepoint_data = []
            for subject_id in all_subjects:
                if t < aligned_data[subject_id].shape[0]:
                    timepoint_data.append(aligned_data[subject_id][t])

            if timepoint_data:
                mean_timepoint = np.mean(timepoint_data, axis=0)
                for subject_id in all_subjects:
                    if t < new_common[subject_id].shape[0]:
                        new_common[subject_id][t] = mean_timepoint

        self.common_space = new_common

    def _compute_convergence(self, prev_common, curr_common):
        """Compute convergence metric"""
        if prev_common is None:
            return float('inf')

        total_diff = 0
        total_norm = 0

        for subject_id in curr_common.keys():
            diff = np.linalg.norm(curr_common[subject_id] - prev_common[subject_id])
            norm = np.linalg.norm(prev_common[subject_id])
            total_diff += diff
            total_norm += norm

        return total_diff / (total_norm + 1e-10)

    def _denormalize_data(self, aligned_data):
        """Denormalize aligned data"""
        denormalized = {}
        for subject_id, data in aligned_data.items():
            if subject_id in self.scalers:
                denormalized[subject_id] = self.scalers[subject_id].inverse_transform(data)
            else:
                denormalized[subject_id] = data
        return denormalized

    def transform(self, new_subject_data, subject_id):
        """Transform new subject data using fitted alignment"""
        if not self.fitted:
            raise ValueError("Hyperalignment must be fitted before transform")

        # Normalize if needed
        if self.normalize:
            if subject_id in self.scalers:
                normalized_data = self.scalers[subject_id].transform(new_subject_data)
            else:
                # Create new scaler for unseen subject
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(new_subject_data)
                self.scalers[subject_id] = scaler
        else:
            normalized_data = new_subject_data

        # Get or create transformation matrix
        if subject_id in self.transformation_matrices:
            W = self.transformation_matrices[subject_id]
        else:
            # Create transformation for new subject
            # Use mean common space as target
            mean_common = np.mean([cs for cs in self.common_space.values()], axis=0)
            W = self._procrustes_alignment(normalized_data, mean_common)
            self.transformation_matrices[subject_id] = W

        # Apply transformation
        aligned_data = normalized_data @ W

        # Denormalize if needed
        if self.normalize and subject_id in self.scalers:
            aligned_data = self.scalers[subject_id].inverse_transform(aligned_data)

        return aligned_data

class ProcrustesAlignment(BaseAlignment):
    """
    Simple Procrustes alignment for pairwise subject alignment
    Finds orthogonal transformation that minimizes Frobenius norm
    """

    def __init__(self, normalize=True):
        """
        Initialize Procrustes alignment

        Args:
            normalize: Whether to normalize data before alignment
        """
        super().__init__()
        self.normalize = normalize
        self.transformation_matrix = None
        self.scaler_source = None
        self.scaler_target = None

    def fit(self, source_data, target_data):
        """
        Fit Procrustes alignment

        Args:
            source_data: Source subject data (n_samples × n_voxels)
            target_data: Target subject data (n_samples × n_voxels)

        Returns:
            self: Fitted alignment object
        """
        print("Fitting Procrustes alignment...")

        # Validate input
        if source_data.shape != target_data.shape:
            raise ValueError("Source and target must have same shape for Procrustes")

        # Normalize if requested
        if self.normalize:
            self.scaler_source = StandardScaler()
            self.scaler_target = StandardScaler()
            source_norm = self.scaler_source.fit_transform(source_data)
            target_norm = self.scaler_target.fit_transform(target_data)
        else:
            source_norm = source_data
            target_norm = target_data

        # Procrustes analysis
        self.transformation_matrix, _ = orthogonal_procrustes(source_norm, target_norm)

        print(f"Transformation matrix shape: {self.transformation_matrix.shape}")

        self.fitted = True
        return self

    def transform(self, source_data, subject_id=None):
        """Transform source data using Procrustes transformation"""
        if not self.fitted:
            raise ValueError("Alignment must be fitted before transform")

        # Normalize if needed
        if self.normalize and self.scaler_source is not None:
            source_norm = self.scaler_source.transform(source_data)
        else:
            source_norm = source_data

        # Apply transformation
        aligned_data = source_norm @ self.transformation_matrix

        # Denormalize if needed
        if self.normalize and self.scaler_target is not None:
            aligned_data = self.scaler_target.inverse_transform(aligned_data)

        return aligned_data

class AlignmentEvaluator:
    """
    Comprehensive evaluation of alignment methods
    """

    def __init__(self):
        self.metrics = {}

    def evaluate_alignment(self, original_data, aligned_data, target_data=None):
        """
        Evaluate alignment quality using multiple metrics

        Args:
            original_data: Original unaligned data
            aligned_data: Aligned data
            target_data: Target data (if available)

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        metrics = {}

        # Inter-subject correlation (ISC)
        if isinstance(aligned_data, dict) and len(aligned_data) > 1:
            metrics['isc'] = self._compute_isc(aligned_data)

        # Alignment consistency
        if target_data is not None:
            metrics['mse'] = mean_squared_error(target_data, aligned_data)
            metrics['r2'] = r2_score(target_data, aligned_data)

        # Representational similarity
        if isinstance(aligned_data, dict):
            metrics['representational_similarity'] = self._compute_representational_similarity(aligned_data)

        return metrics

    def _compute_isc(self, multi_subject_data):
        """Compute inter-subject correlation"""
        subjects = list(multi_subject_data.keys())
        n_subjects = len(subjects)

        if n_subjects < 2:
            return 0.0

        correlations = []

        for i in range(n_subjects):
            for j in range(i + 1, n_subjects):
                data_i = multi_subject_data[subjects[i]]
                data_j = multi_subject_data[subjects[j]]

                # Compute correlation for each voxel
                min_samples = min(data_i.shape[0], data_j.shape[0])
                corr_per_voxel = []

                for voxel in range(data_i.shape[1]):
                    corr = np.corrcoef(data_i[:min_samples, voxel],
                                     data_j[:min_samples, voxel])[0, 1]
                    if not np.isnan(corr):
                        corr_per_voxel.append(corr)

                if corr_per_voxel:
                    correlations.append(np.mean(corr_per_voxel))

        return np.mean(correlations) if correlations else 0.0

    def _compute_representational_similarity(self, multi_subject_data):
        """Compute representational similarity across subjects"""
        subjects = list(multi_subject_data.keys())

        # Compute representational dissimilarity matrices (RDMs)
        rdms = {}
        for subject_id, data in multi_subject_data.items():
            # Compute pairwise distances between timepoints
            distances = pdist(data, metric='correlation')
            rdm = squareform(distances)
            rdms[subject_id] = rdm

        # Compute similarity between RDMs
        rdm_similarities = []
        for i, subj_i in enumerate(subjects):
            for j, subj_j in enumerate(subjects[i+1:], i+1):
                rdm_i = rdms[subj_i].flatten()
                rdm_j = rdms[subj_j].flatten()

                # Remove diagonal and NaN values
                mask = ~(np.isnan(rdm_i) | np.isnan(rdm_j))
                if np.sum(mask) > 0:
                    similarity = np.corrcoef(rdm_i[mask], rdm_j[mask])[0, 1]
                    if not np.isnan(similarity):
                        rdm_similarities.append(similarity)

        return np.mean(rdm_similarities) if rdm_similarities else 0.0

class MultiSubjectAlignmentPipeline:
    """
    Complete pipeline for multi-subject alignment and evaluation
    """

    def __init__(self, alignment_method='ridge', alignment_params=None):
        """
        Initialize alignment pipeline

        Args:
            alignment_method: 'ridge', 'hyperalignment', 'procrustes'
            alignment_params: Parameters for alignment method
        """
        self.alignment_method = alignment_method
        self.alignment_params = alignment_params or {}

        # Initialize alignment method
        if alignment_method == 'ridge':
            self.alignment = RidgeAlignment(**self.alignment_params)
        elif alignment_method == 'hyperalignment':
            self.alignment = Hyperalignment(**self.alignment_params)
        elif alignment_method == 'procrustes':
            self.alignment = ProcrustesAlignment(**self.alignment_params)
        else:
            raise ValueError(f"Unknown alignment method: {alignment_method}")

        self.evaluator = AlignmentEvaluator()
        self.fitted = False

    def fit_transform(self, multi_subject_data):
        """
        Fit alignment and transform data

        Args:
            multi_subject_data: dict {subject_id: fMRI_data}

        Returns:
            aligned_data: dict {subject_id: aligned_fMRI_data}
            evaluation_metrics: dict of evaluation metrics
        """
        print(f"Running alignment pipeline with {self.alignment_method}")

        if self.alignment_method in ['ridge', 'procrustes']:
            # Pairwise alignment - use first subject as reference
            aligned_data = self._pairwise_alignment(multi_subject_data)
        elif self.alignment_method == 'hyperalignment':
            # Multi-subject alignment
            aligned_data = self.alignment.fit_transform(multi_subject_data)
        else:
            raise ValueError(f"Unsupported alignment method: {self.alignment_method}")

        # Evaluate alignment quality
        evaluation_metrics = self.evaluator.evaluate_alignment(
            multi_subject_data, aligned_data
        )

        print("Alignment evaluation metrics:")
        for metric, value in evaluation_metrics.items():
            print(f"  {metric}: {value:.4f}")

        self.fitted = True

        return aligned_data, evaluation_metrics

    def _pairwise_alignment(self, multi_subject_data):
        """Perform pairwise alignment using reference subject"""
        subject_ids = list(multi_subject_data.keys())
        reference_id = subject_ids[0]
        reference_data = multi_subject_data[reference_id]

        aligned_data = {reference_id: reference_data}

        print(f"Using subject {reference_id} as reference")

        for subject_id in subject_ids[1:]:
            print(f"Aligning subject {subject_id} to reference")
            source_data = multi_subject_data[subject_id]

            # Fit alignment for this subject pair
            self.alignment.fit(source_data, reference_data)
            aligned_data[subject_id] = self.alignment.transform(source_data)

        return aligned_data

    def transform_new_subject(self, new_subject_data, subject_id):
        """
        Transform new subject data using fitted alignment

        Args:
            new_subject_data: fMRI data from new subject
            subject_id: Subject identifier

        Returns:
            aligned_data: Transformed data
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transforming new subjects")

        return self.alignment.transform(new_subject_data, subject_id)

def compare_alignment_methods(multi_subject_data, methods=['ridge', 'hyperalignment', 'procrustes']):
    """
    Compare different alignment methods on the same data

    Args:
        multi_subject_data: dict {subject_id: fMRI_data}
        methods: List of alignment methods to compare

    Returns:
        comparison_results: dict {method: {aligned_data, metrics}}
    """
    print(f"Comparing alignment methods: {methods}")

    results = {}

    for method in methods:
        print(f"\n--- Testing {method} alignment ---")

        try:
            # Initialize pipeline
            if method == 'ridge':
                params = {'alpha': 'auto', 'normalize': True}
            elif method == 'hyperalignment':
                params = {'max_iterations': 5, 'normalize': True}
            elif method == 'procrustes':
                params = {'normalize': True}
            else:
                params = {}

            pipeline = MultiSubjectAlignmentPipeline(
                alignment_method=method,
                alignment_params=params
            )

            # Fit and evaluate
            aligned_data, metrics = pipeline.fit_transform(multi_subject_data)

            results[method] = {
                'aligned_data': aligned_data,
                'metrics': metrics,
                'pipeline': pipeline
            }

        except Exception as e:
            print(f"Error with {method}: {str(e)}")
            results[method] = {
                'error': str(e)
            }

    return results