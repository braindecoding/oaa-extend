#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use Aligned Data for Downstream Processing
Demonstrates how to load and use saved alignment results for modular processing.

@author: Rolly Maulana Awangga
"""

import numpy as np
import sys
import os
import pickle
import json
import glob

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extended'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from extended.multi_subject_dgmm import MultiSubjectDGMM
from extended.alignment_methods import MultiSubjectAlignmentPipeline

def list_available_alignments(output_dir='./outputs'):
    """
    List all available alignment results
    
    Args:
        output_dir: Directory containing alignment results
        
    Returns:
        available_files: dict with available alignment files
    """
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return {}
    
    # Find all alignment data files
    data_files = glob.glob(os.path.join(output_dir, "alignment_*_aligned_data.npz"))
    
    available_files = {}
    
    for data_file in data_files:
        base_name = os.path.basename(data_file).replace('_aligned_data.npz', '')
        
        # Find corresponding files
        metrics_file = data_file.replace('_aligned_data.npz', '_metrics.json')
        pipeline_file = data_file.replace('_aligned_data.npz', '_pipeline.pkl')
        dgmm_file = data_file.replace('_aligned_data.npz', '_dgmm_model.pkl')
        summary_file = data_file.replace('_aligned_data.npz', '_summary.txt')
        
        available_files[base_name] = {
            'aligned_data': data_file if os.path.exists(data_file) else None,
            'metrics': metrics_file if os.path.exists(metrics_file) else None,
            'pipeline': pipeline_file if os.path.exists(pipeline_file) else None,
            'dgmm_model': dgmm_file if os.path.exists(dgmm_file) else None,
            'summary': summary_file if os.path.exists(summary_file) else None
        }
    
    return available_files

def load_alignment_session(session_name, output_dir='./outputs'):
    """
    Load a complete alignment session
    
    Args:
        session_name: Name of the alignment session
        output_dir: Directory containing alignment results
        
    Returns:
        session_data: dict with all loaded components
    """
    available = list_available_alignments(output_dir)
    
    if session_name not in available:
        print(f"Session '{session_name}' not found")
        print("Available sessions:")
        for name in available.keys():
            print(f"  - {name}")
        return None
    
    session_files = available[session_name]
    session_data = {}
    
    print(f"Loading alignment session: {session_name}")
    
    # Load aligned data
    if session_files['aligned_data']:
        loaded = np.load(session_files['aligned_data'])
        session_data['aligned_data'] = {key: loaded[key] for key in loaded.files}
        print(f"✓ Loaded aligned data: {len(session_data['aligned_data'])} subjects")
    
    # Load metrics
    if session_files['metrics']:
        with open(session_files['metrics'], 'r') as f:
            session_data['metrics'] = json.load(f)
        print(f"✓ Loaded metrics")
    
    # Load pipeline
    if session_files['pipeline']:
        with open(session_files['pipeline'], 'rb') as f:
            session_data['pipeline'] = pickle.load(f)
        print(f"✓ Loaded pipeline")
    
    # Load DGMM model
    if session_files['dgmm_model']:
        with open(session_files['dgmm_model'], 'rb') as f:
            session_data['dgmm_model'] = pickle.load(f)
        print(f"✓ Loaded DGMM model")
    
    return session_data

def demonstrate_downstream_processing(session_name=None, output_dir='./outputs'):
    """
    Demonstrate downstream processing using saved alignment results
    
    Args:
        session_name: Name of alignment session to use (None for latest)
        output_dir: Directory containing alignment results
    """
    print("DOWNSTREAM PROCESSING WITH SAVED ALIGNMENT RESULTS")
    print("=" * 60)
    
    # List available sessions
    available = list_available_alignments(output_dir)
    
    if not available:
        print("❌ No alignment results found!")
        print(f"Please run 'python run_multi_subject_alignment.py' first")
        return
    
    print(f"Found {len(available)} alignment session(s):")
    for name in available.keys():
        print(f"  - {name}")
    
    # Select session
    if session_name is None:
        # Use the latest session
        session_name = max(available.keys())
        print(f"\nUsing latest session: {session_name}")
    else:
        print(f"\nUsing specified session: {session_name}")
    
    # Load session data
    session_data = load_alignment_session(session_name, output_dir)
    
    if session_data is None:
        return
    
    # Example 1: Use aligned data for new analysis
    print(f"\n1. ANALYZING ALIGNED DATA")
    print("-" * 30)
    
    aligned_data = session_data['aligned_data']
    
    # Calculate inter-subject similarity
    subject_ids = list(aligned_data.keys())
    if len(subject_ids) >= 2:
        subj1_data = aligned_data[subject_ids[0]]
        subj2_data = aligned_data[subject_ids[1]]
        
        # Calculate correlation between subjects
        min_samples = min(subj1_data.shape[0], subj2_data.shape[0])
        correlation = np.corrcoef(
            subj1_data[:min_samples].flatten(),
            subj2_data[:min_samples].flatten()
        )[0, 1]
        
        print(f"Inter-subject correlation: {correlation:.4f}")
    
    # Example 2: Use trained DGMM for new predictions
    if 'dgmm_model' in session_data:
        print(f"\n2. USING TRAINED DGMM MODEL")
        print("-" * 30)
        
        dgmm_model = session_data['dgmm_model']
        
        # Use aligned data for prediction
        test_subject = subject_ids[0]
        test_fmri = aligned_data[test_subject][:3]  # Use first 3 samples
        
        try:
            predictions = dgmm_model.predict(test_fmri)
            print(f"✓ Generated predictions: {predictions.shape}")
            print(f"  Input fMRI: {test_fmri.shape}")
            print(f"  Output images: {predictions.shape}")
        except Exception as e:
            print(f"⚠️ Prediction failed: {str(e)}")
    
    # Example 3: Use pipeline for new subject alignment
    if 'pipeline' in session_data:
        print(f"\n3. ALIGNING NEW SUBJECT DATA")
        print("-" * 30)
        
        pipeline = session_data['pipeline']
        
        # Simulate new subject data
        reference_shape = aligned_data[subject_ids[0]].shape
        new_subject_data = np.random.randn(10, reference_shape[1])
        
        try:
            aligned_new = pipeline.transform_new_subject(new_subject_data, 'new_subject')
            print(f"✓ Aligned new subject data: {aligned_new.shape}")
        except Exception as e:
            print(f"⚠️ New subject alignment failed: {str(e)}")
    
    # Example 4: Export aligned data for external tools
    print(f"\n4. EXPORTING FOR EXTERNAL TOOLS")
    print("-" * 30)
    
    # Export to CSV for analysis in R/MATLAB
    export_dir = os.path.join(output_dir, 'exports')
    os.makedirs(export_dir, exist_ok=True)
    
    for subject_id, data in aligned_data.items():
        csv_file = os.path.join(export_dir, f"{subject_id}_aligned.csv")
        np.savetxt(csv_file, data, delimiter=',')
        print(f"✓ Exported {subject_id}: {csv_file}")
    
    # Export summary statistics
    stats_file = os.path.join(export_dir, 'alignment_stats.csv')
    with open(stats_file, 'w') as f:
        f.write("subject_id,n_timepoints,n_features,mean_activation,std_activation\n")
        for subject_id, data in aligned_data.items():
            f.write(f"{subject_id},{data.shape[0]},{data.shape[1]},{np.mean(data):.6f},{np.std(data):.6f}\n")
    print(f"✓ Exported statistics: {stats_file}")
    
    print(f"\n{'='*60}")
    print("✅ DOWNSTREAM PROCESSING COMPLETED")
    print(f"{'='*60}")
    print("Key benefits of modular approach:")
    print("1. ✅ Reuse alignment results without recomputation")
    print("2. ✅ Share aligned data with collaborators")
    print("3. ✅ Use different downstream methods on same alignment")
    print("4. ✅ Export to external analysis tools")
    print("5. ✅ Reproducible research workflow")

def compare_alignment_sessions(output_dir='./outputs'):
    """
    Compare multiple alignment sessions
    
    Args:
        output_dir: Directory containing alignment results
    """
    print("COMPARING ALIGNMENT SESSIONS")
    print("=" * 40)
    
    available = list_available_alignments(output_dir)
    
    if len(available) < 2:
        print("Need at least 2 alignment sessions for comparison")
        return
    
    print(f"Comparing {len(available)} sessions:")
    
    comparison_data = []
    
    for session_name in available.keys():
        session_data = load_alignment_session(session_name, output_dir)
        
        if session_data and 'metrics' in session_data:
            metrics = session_data['metrics']
            
            comparison_data.append({
                'session': session_name,
                'method': metrics.get('alignment_method', 'unknown'),
                'correlation': metrics['loso_results']['mean_scores']['correlation'],
                'r2': metrics['loso_results']['mean_scores']['r2'],
                'n_subjects': len(metrics.get('data_shapes', {}))
            })
    
    # Print comparison table
    print(f"\n{'Session':<25} {'Method':<15} {'Correlation':<12} {'R²':<8} {'Subjects':<8}")
    print("-" * 70)
    
    for data in comparison_data:
        print(f"{data['session']:<25} {data['method']:<15} {data['correlation']:<12.4f} {data['r2']:<8.4f} {data['n_subjects']:<8}")
    
    # Find best session
    if comparison_data:
        best_session = max(comparison_data, key=lambda x: x['correlation'])
        print(f"\n🏆 Best session (highest correlation): {best_session['session']}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Use saved alignment results')
    parser.add_argument('--session', type=str, help='Specific session name to use')
    parser.add_argument('--compare', action='store_true', help='Compare all available sessions')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_alignment_sessions(args.output_dir)
    else:
        demonstrate_downstream_processing(args.session, args.output_dir)
