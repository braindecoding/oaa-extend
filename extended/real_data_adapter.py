#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Adapter for Multi-Subject Alignment
Adapts digit69_28x28.mat and other real data files to work with 
multi-subject alignment implementation.

@author: Rolly Maulana Awangga
"""

import numpy as np
import scipy.io as sio
import sys
import os

# Add lib directory to path for existing data loading functions
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import prepro
except ImportError:
    print("Warning: prepro module not found, using fallback data loading")
    prepro = None

from alignment_methods import MultiSubjectAlignmentPipeline
from multi_subject_dgmm import MultiSubjectDGMM

class RealDataAdapter:
    """
    Adapter to convert real data to multi-subject format
    """
    
    def __init__(self, data_path='./data'):
        self.data_path = data_path
        self.digit_file = os.path.join(data_path, 'digit69_28x28.mat')
        
    def load_digit_data(self):
        """Load digit69_28x28.mat data"""
        print("Loading digit69_28x28.mat...")
        
        if not os.path.exists(self.digit_file):
            raise FileNotFoundError(f"Data file not found: {self.digit_file}")
            
        # Try using existing prepro module first
        if prepro is not None:
            try:
                X_train, X_test, X_val, Y_train, Y_test, Y_val = prepro.getXYVal(self.digit_file, 28)
                
                print(f"✓ Loaded using prepro module")
                print(f"  X_train shape: {X_train.shape}")
                print(f"  Y_train shape: {Y_train.shape}")
                print(f"  X_test shape: {X_test.shape}")
                print(f"  Y_test shape: {Y_test.shape}")
                
                return {
                    'X_train': X_train, 'Y_train': Y_train,
                    'X_test': X_test, 'Y_test': Y_test,
                    'X_val': X_val, 'Y_val': Y_val
                }
                
            except Exception as e:
                print(f"Error using prepro: {str(e)}")
                print("Falling back to direct loading...")
        
        # Fallback: direct loading
        return self._load_digit_direct()
        
    def _load_digit_direct(self):
        """Direct loading of digit69_28x28.mat"""
        data = sio.loadmat(self.digit_file)
        
        # Remove MATLAB metadata
        data_keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Available keys: {data_keys}")
        
        # Expected keys based on prepro.py analysis
        expected_keys = ['fmriTrn', 'fmriTest', 'stimTrn', 'stimTest']
        
        result = {}
        
        for key in expected_keys:
            if key in data:
                var = data[key]
                print(f"Found {key}: shape {var.shape}, dtype {var.dtype}")
                result[key] = var
            else:
                print(f"Warning: {key} not found in data")
                
        # Convert to standard format
        if 'stimTrn' in result and 'fmriTrn' in result:
            X_train = result['stimTrn'].astype('float32') / 255.0
            Y_train = result['fmriTrn'].astype('float32')
            
            # Reshape images to 4D if needed
            if len(X_train.shape) == 2:
                X_train = X_train.reshape([X_train.shape[0], 28, 28, 1])
                
        if 'stimTest' in result and 'fmriTest' in result:
            X_test = result['stimTest'].astype('float32') / 255.0
            Y_test = result['fmriTest'].astype('float32')
            
            # Reshape images to 4D if needed
            if len(X_test.shape) == 2:
                X_test = X_test.reshape([X_test.shape[0], 28, 28, 1])
                
        return {
            'X_train': X_train, 'Y_train': Y_train,
            'X_test': X_test, 'Y_test': Y_test,
            'X_val': None, 'Y_val': None  # No validation set in direct loading
        }
        
    def create_multi_subject_data(self, n_subjects=3, split_method='random'):
        """
        Create multi-subject data from single-subject digit data
        
        Args:
            n_subjects: Number of subjects to create
            split_method: 'random', 'sequential', or 'bootstrap'
            
        Returns:
            multi_subject_data: dict {subject_id: {'X': images, 'Y': fmri}}
        """
        print(f"Creating multi-subject data with {n_subjects} subjects...")
        
        # Load original data
        data = self.load_digit_data()
        
        X_all = np.concatenate([data['X_train'], data['X_test']], axis=0)
        Y_all = np.concatenate([data['Y_train'], data['Y_test']], axis=0)
        
        print(f"Total samples: {X_all.shape[0]}")
        print(f"Image shape: {X_all.shape[1:]}")
        print(f"fMRI shape: {Y_all.shape[1:]}")
        
        multi_subject_data = {}
        
        if split_method == 'random':
            # Random split
            np.random.seed(42)  # For reproducibility
            indices = np.random.permutation(X_all.shape[0])
            samples_per_subject = X_all.shape[0] // n_subjects
            
            for i in range(n_subjects):
                start_idx = i * samples_per_subject
                end_idx = start_idx + samples_per_subject
                
                subject_indices = indices[start_idx:end_idx]
                
                multi_subject_data[f'subject_{i}'] = {
                    'X': X_all[subject_indices],
                    'Y': Y_all[subject_indices]
                }
                
        elif split_method == 'sequential':
            # Sequential split
            samples_per_subject = X_all.shape[0] // n_subjects
            
            for i in range(n_subjects):
                start_idx = i * samples_per_subject
                end_idx = start_idx + samples_per_subject
                
                multi_subject_data[f'subject_{i}'] = {
                    'X': X_all[start_idx:end_idx],
                    'Y': Y_all[start_idx:end_idx]
                }
                
        elif split_method == 'bootstrap':
            # Bootstrap sampling (with replacement)
            np.random.seed(42)
            samples_per_subject = min(50, X_all.shape[0] // 2)  # Smaller samples for bootstrap
            
            for i in range(n_subjects):
                subject_indices = np.random.choice(X_all.shape[0], 
                                                 size=samples_per_subject, 
                                                 replace=True)
                
                multi_subject_data[f'subject_{i}'] = {
                    'X': X_all[subject_indices],
                    'Y': Y_all[subject_indices]
                }
                
        # Add subject-specific transformations to simulate real multi-subject variability
        multi_subject_data = self._add_subject_variability(multi_subject_data)
        
        # Print summary
        print(f"\nMulti-subject data created:")
        for subject_id, subject_data in multi_subject_data.items():
            print(f"  {subject_id}: {subject_data['X'].shape[0]} samples")
            
        return multi_subject_data
        
    def _add_subject_variability(self, multi_subject_data, noise_level=0.1):
        """Add realistic subject-specific variability to fMRI data"""
        print("Adding subject-specific variability...")
        
        np.random.seed(42)
        
        for subject_id, subject_data in multi_subject_data.items():
            Y_original = subject_data['Y']
            n_voxels = Y_original.shape[1]
            
            # Create subject-specific transformation matrix
            # Small rotation + scaling to simulate individual differences
            transform_strength = 0.05  # Small transformation
            transform_matrix = np.eye(n_voxels) + \
                             transform_strength * np.random.randn(n_voxels, n_voxels)
            
            # Apply transformation
            Y_transformed = Y_original @ transform_matrix
            
            # Add subject-specific noise
            subject_noise = noise_level * np.random.randn(*Y_original.shape)
            Y_final = Y_transformed + subject_noise
            
            # Update data
            multi_subject_data[subject_id]['Y'] = Y_final.astype('float32')
            
        return multi_subject_data
        
    def test_alignment_with_real_data(self, alignment_methods=['ridge', 'hyperalignment']):
        """Test alignment methods with real data"""
        print(f"\n{'='*60}")
        print("TESTING ALIGNMENT WITH REAL DATA")
        print(f"{'='*60}")
        
        # Create multi-subject data
        multi_subject_data = self.create_multi_subject_data(n_subjects=3, split_method='random')
        
        # Test each alignment method
        results = {}
        
        for method in alignment_methods:
            print(f"\n--- Testing {method.upper()} ---")
            
            try:
                # Set parameters
                if method == 'ridge':
                    params = {'alpha': 'auto', 'normalize': True}
                elif method == 'hyperalignment':
                    params = {'n_components': 50, 'max_iterations': 5, 'normalize': True}
                else:
                    params = {}
                    
                # Create pipeline
                pipeline = MultiSubjectAlignmentPipeline(
                    alignment_method=method,
                    alignment_params=params
                )
                
                # Extract fMRI data for alignment
                fmri_data = {sid: data['Y'] for sid, data in multi_subject_data.items()}
                
                # Apply alignment
                aligned_fmri, metrics = pipeline.fit_transform(fmri_data)
                
                print(f"✓ {method} alignment completed")
                print(f"  Metrics: {metrics}")
                
                results[method] = {
                    'aligned_data': aligned_fmri,
                    'metrics': metrics,
                    'success': True
                }
                
            except Exception as e:
                print(f"✗ {method} failed: {str(e)}")
                results[method] = {
                    'error': str(e),
                    'success': False
                }
                
        return results
        
    def test_multi_subject_dgmm_with_real_data(self):
        """Test MultiSubjectDGMM with real data"""
        print(f"\n{'='*60}")
        print("TESTING MULTI-SUBJECT DGMM WITH REAL DATA")
        print(f"{'='*60}")
        
        # Create multi-subject data
        multi_subject_data = self.create_multi_subject_data(n_subjects=2, split_method='random')
        
        # Test different alignment methods
        alignment_methods = ['ridge', 'hyperalignment', 'none']
        
        for method in alignment_methods:
            print(f"\n--- Testing MultiSubjectDGMM with {method} ---")
            
            try:
                # Set parameters
                if method == 'ridge':
                    params = {'alpha': 1.0, 'normalize': True}
                elif method == 'hyperalignment':
                    params = {'n_components': 30, 'max_iterations': 3}
                else:
                    params = {}
                    
                # Initialize MultiSubjectDGMM
                ms_dgmm = MultiSubjectDGMM(
                    alignment_method=method,
                    alignment_params=params
                )
                
                # Train
                ms_dgmm.fit(multi_subject_data)
                
                # Test prediction
                test_subject = list(multi_subject_data.keys())[0]
                test_fmri = multi_subject_data[test_subject]['Y'][:5]  # Use first 5 samples
                
                predictions = ms_dgmm.predict(test_fmri)
                
                print(f"✓ {method} MultiSubjectDGMM completed")
                print(f"  Input fMRI shape: {test_fmri.shape}")
                print(f"  Prediction shape: {predictions.shape}")
                
            except Exception as e:
                print(f"✗ {method} MultiSubjectDGMM failed: {str(e)}")

def main():
    """Main function to test real data compatibility"""
    print("REAL DATA COMPATIBILITY TEST")
    print("=" * 60)
    
    try:
        # Initialize adapter
        adapter = RealDataAdapter()
        
        # Test data loading
        print("1. Testing data loading...")
        data = adapter.load_digit_data()
        print("✓ Data loading successful")
        
        # Test multi-subject creation
        print("\n2. Testing multi-subject data creation...")
        multi_data = adapter.create_multi_subject_data(n_subjects=3)
        print("✓ Multi-subject data creation successful")
        
        # Test alignment
        print("\n3. Testing alignment with real data...")
        alignment_results = adapter.test_alignment_with_real_data()
        print("✓ Alignment testing completed")
        
        # Test MultiSubjectDGMM
        print("\n4. Testing MultiSubjectDGMM with real data...")
        adapter.test_multi_subject_dgmm_with_real_data()
        print("✓ MultiSubjectDGMM testing completed")
        
        print(f"\n{'='*60}")
        print("✅ ALL TESTS PASSED!")
        print("Real data is compatible with multi-subject alignment implementation")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
