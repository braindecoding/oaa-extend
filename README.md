# Option A Implementation: Flat Structure untuk Quick Progress

## 🚀 **IMMEDIATE ACTION PLAN**

### **Step 1: Setup Directory Structure (5 minutes)**
```bash
# Create directories
mkdir -p extended
mkdir -p legacy  
mkdir -p data/multi_subject
mkdir -p results/comparison

# Preserve existing work
cp oaavangerven.py legacy/oaavangerven_original.py
cp fidvg.py legacy/fidvg_original.py
cp runvg.sh legacy/runvg_original.sh
cp -r vg/ legacy/results_690_experiments/
```

### **Step 2: Create Core Extended Files (Today)**
```bash
# Files to create in extended/ folder:
touch extended/oaavangerven_extended.py      # MAIN SCRIPT
touch extended/fidvg_extended.py             # EXTENDED EVALUATION  
touch extended/multi_subject_dgmm.py         # MULTI-SUBJECT FUNCTIONALITY
touch extended/advanced_architectures.py     # LDM, GAN, TRANSFORMER
touch extended/bayesian_optimization.py      # BAYESIAN OPT
touch extended/comprehensive_eval.py         # COMPREHENSIVE EVALUATION
```

---

## 📝 **FILE IMPLEMENTATIONS**

### **File 1: `extended/oaavangerven_extended.py` (MAIN SCRIPT)**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended DGMM Implementation - Multi-Subject & Multi-Architecture Support
Based on original oaavangerven.py with enhancements for pembimbing revisions

@author: Rolly Maulana Awangga
"""

import os
import sys
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend, optimizers, metrics
from tensorflow.python.framework.ops import disable_eager_execution

# GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['THEANO_FLAGS'] = "device=gpu"
disable_eager_execution()

# Import legacy components (REUSE EXISTING)
sys.path.append('../legacy')
from legacy.oaavangerven_original import prepro, ars, obj, init, train

# Import new extended components
from multi_subject_dgmm import MultiSubjectDGMM, RidgeAlignment
from advanced_architectures import BrainLDM, BrainStyleGAN, BrainViT
from comprehensive_eval import ComprehensiveEvaluator

class ExtendedExperiment:
    def __init__(self, architecture='dgmm', alignment_method='ridge', multi_subject=False):
        """
        Extended experiment framework
        
        Args:
            architecture: 'dgmm', 'ldm', 'stylegan', 'transformer'
            alignment_method: 'ridge', 'hyperalignment', 'none'
            multi_subject: True for multi-subject, False for single-subject
        """
        self.architecture = architecture
        self.alignment_method = alignment_method
        self.multi_subject = multi_subject
        
        # Initialize alignment if needed
        if multi_subject and alignment_method != 'none':
            if alignment_method == 'ridge':
                self.alignment = RidgeAlignment()
            else:
                raise NotImplementedError(f"Alignment method {alignment_method} not implemented yet")
        
        # Initialize evaluator
        self.evaluator = ComprehensiveEvaluator()
    
    def load_data(self, data_path='./data'):
        """Load data - single or multi-subject"""
        if self.multi_subject:
            return self._load_multi_subject_data(data_path)
        else:
            # REUSE existing data loading
            return prepro.getXYVal('./data/digit69_28x28.mat', 28)
    
    def _load_multi_subject_data(self, data_path):
        """Load multi-subject data (placeholder - to be implemented)"""
        # TODO: Implement multi-subject data loading
        print("Multi-subject data loading - TO BE IMPLEMENTED")
        # For now, use single subject data
        return prepro.getXYVal('./data/digit69_28x28.mat', 28)
    
    def create_model(self, config):
        """Create model based on architecture"""
        if self.architecture == 'dgmm':
            return self._create_dgmm_model(config)
        elif self.architecture == 'ldm':
            return BrainLDM(config)
        elif self.architecture == 'stylegan':
            return BrainStyleGAN(config)
        elif self.architecture == 'transformer':
            return BrainViT(config)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def _create_dgmm_model(self, config):
        """Create DGMM model (REUSE EXISTING LOGIC)"""
        # Extract parameters
        K = config['latent_dim']
        intermediate_dim = config['intermediate_dim']
        batch_size = config['batch_size']
        
        # REUSE existing VAE architecture
        img_rows, img_cols, img_chns = 28, 28, 1
        filters = 64
        num_conv = 3
        D2 = 3092  # fMRI dimensions
        
        if backend.image_data_format() == 'channels_first':
            original_img_size = (img_chns, img_rows, img_cols)
        else:
            original_img_size = (img_rows, img_cols, img_chns)
        
        # Build architecture (REUSE EXISTING)
        X = Input(shape=original_img_size)
        Y = Input(shape=(D2,))
        Y_mu = Input(shape=(D2,))
        Y_lsgms = Input(shape=(D2,))
        
        Z, Z_lsgms, Z_mu = ars.encoder(X, D2, img_chns, filters, num_conv, intermediate_dim, K)
        
        # REUSE existing decoder architecture
        decoder_components = ars.decoderars(intermediate_dim, filters, batch_size, num_conv, img_chns)
        X_mu, X_lsgms = ars.decoders(Z, *decoder_components)
        
        # REUSE existing custom loss function
        def custom_loss(X, X_mu):
            X = backend.flatten(X)
            X_mu = backend.flatten(X_mu) 
            Lp = 0.5 * backend.mean(1 + Z_lsgms - backend.square(Z_mu) - backend.exp(Z_lsgms), axis=-1)     
            Lx = -metrics.binary_crossentropy(X, X_mu)
            Ly = obj.Y_normal_logpdf(Y, Y_mu, Y_lsgms, backend)
            lower_bound = backend.mean(Lp + 10000 * Lx + Ly)
            cost = -lower_bound
            return cost
        
        # Build model
        DGMM = Model(inputs=[X, Y, Y_mu, Y_lsgms], outputs=X_mu)
        
        try:
            opt_method = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        except:
            opt_method = optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        DGMM.compile(optimizer=opt_method, loss=custom_loss)
        
        return {
            'model': DGMM,
            'encoder': Model(inputs=X, outputs=[Z_mu, Z_lsgms]),
            'decoder': Model(inputs=Input(shape=(K,)), outputs=ars.decoders(Input(shape=(K,)), *decoder_components)[0])
        }
    
    def train_model(self, model, data, config):
        """Train model with extended functionality"""
        if self.architecture == 'dgmm':
            return self._train_dgmm(model, data, config)
        else:
            # For other architectures, call their specific training methods
            return model.train(data, config)
    
    def _train_dgmm(self, model_dict, data, config):
        """Train DGMM (REUSE EXISTING TRAINING LOGIC)"""
        X_train, X_test, X_validation, Y_train, Y_test, Y_validation = data
        
        maxiter = config['max_iterations']
        batch_size = config['batch_size']
        K = config['latent_dim']
        D2 = Y_train.shape[1]
        C = 5
        
        # REUSE existing initialization
        numTrn = X_train.shape[0]
        Z_mu, B_mu, R_mu, H_mu = init.randombetween0and1withmatrixsize(numTrn, K, C, D2)
        Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu).astype(np.float32)
        
        sigma_r, sigma_h = init.matriksidentitasukuran(C)
        tau_alpha, tau_beta = 1, 1
        eta_alpha, eta_beta = 1, 1
        gamma_alpha, gamma_beta = 1, 1
        
        tau_mu, eta_mu, gamma_mu = init.alphabagibeta(tau_alpha, tau_beta, eta_alpha, eta_beta, gamma_alpha, gamma_beta)
        Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2))).astype(np.float32)
        
        # Build similarity matrix (REUSE EXISTING)
        from lib import calculate
        S = np.mat(calculate.S(k=10, t=10.0, Y_train, Y_validation))
        
        # REUSE existing training loop
        DGMM = model_dict['model']
        encoder = model_dict['encoder']
        
        for l in range(maxiter):
            print(f'Training iteration {l+1}/{maxiter}')
            
            # REUSE existing parameter updates
            Z_mu, Z_lsgms = train.updateZ(DGMM, X_train, Y_train, Y_mu, Y_lsgms, 1, batch_size, encoder)
            B_mu, sigma_b = train.updateB(Z_lsgms, Z_mu, K, tau_mu, gamma_mu, Y_train, R_mu, H_mu)
            H_mu, sigma_h = train.updateH(R_mu, numTrn, sigma_r, eta_mu, C, gamma_mu, Y_train, Z_mu, B_mu)
            R_mu, sigma_r = train.updateR(H_mu, D2, sigma_h, C, gamma_mu, Y_train, Z_mu, B_mu)
            
            Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu)
            
            tau_mu = train.updateTau(tau_alpha, K, D2, tau_beta, B_mu, sigma_b)
            eta_mu = train.updateEta(eta_alpha, C, D2, eta_beta, H_mu, sigma_h)
            gamma_mu = train.updateGamma(gamma_alpha, numTrn, D2, Y_train, Z_mu, B_mu, R_mu, H_mu, gamma_beta)
            
            Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2)))
        
        return {
            'model': DGMM,
            'encoder': encoder,
            'decoder': model_dict['decoder'],
            'parameters': {
                'Z_mu': Z_mu, 'B_mu': B_mu, 'H_mu': H_mu, 'R_mu': R_mu,
                'S': S, 'gamma_mu': gamma_mu
            }
        }
    
    def evaluate_model(self, trained_model, test_data, config):
        """Evaluate model with comprehensive metrics"""
        return self.evaluator.evaluate_comprehensive(trained_model, test_data, config)

def main():
    """Main execution function"""
    # Parse command line arguments
    if len(sys.argv) >= 5:
        K = int(sys.argv[1])
        intermediate_dim = int(sys.argv[2])
        batch_size = int(sys.argv[3])
        maxiter = int(sys.argv[4])
        architecture = sys.argv[5] if len(sys.argv) > 5 else 'dgmm'
        alignment = sys.argv[6] if len(sys.argv) > 6 else 'ridge'
        multi_subject = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else False
    else:
        # Default parameters for testing
        K, intermediate_dim, batch_size, maxiter = 12, 128, 10, 100
        architecture, alignment, multi_subject = 'dgmm', 'ridge', False
    
    # Configuration
    config = {
        'latent_dim': K,
        'intermediate_dim': intermediate_dim,
        'batch_size': batch_size,
        'max_iterations': maxiter
    }
    
    # Create experiment
    experiment = ExtendedExperiment(
        architecture=architecture,
        alignment_method=alignment,
        multi_subject=multi_subject
    )
    
    print(f"Running extended experiment:")
    print(f"Architecture: {architecture}")
    print(f"Alignment: {alignment}")
    print(f"Multi-subject: {multi_subject}")
    print(f"Config: {config}")
    
    # Load data
    data = experiment.load_data()
    
    # Create and train model
    model = experiment.create_model(config)
    trained_model = experiment.train_model(model, data, config)
    
    # Evaluate model
    results = experiment.evaluate_model(trained_model, data[1:3], config)  # X_test, Y_test
    
    print(f"Evaluation results: {results}")
    
    return results

if __name__ == '__main__':
    main()
```

### **File 2: `extended/fidvg_extended.py` (EXTENDED EVALUATION)**

```python
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
```

### **File 3: `extended/multi_subject_dgmm.py` (MULTI-SUBJECT FUNCTIONALITY)**

```python
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