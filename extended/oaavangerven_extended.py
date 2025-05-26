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