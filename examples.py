#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Subject Alignment Examples
Complete examples for using the modular multi-subject alignment system.

@author: Rolly Maulana Awangga
"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extended'))

def example_1_basic_usage():
    """Example 1: Basic usage with real data"""
    print("EXAMPLE 1: BASIC USAGE")
    print("=" * 30)
    
    try:
        from extended.real_data_adapter import RealDataAdapter
        from extended.multi_subject_dgmm import MultiSubjectDGMM
        
        # Load real data
        adapter = RealDataAdapter(data_path='./data')
        multi_subject_data = adapter.create_multi_subject_data(n_subjects=3)
        
        print(f"✓ Created {len(multi_subject_data)} subjects from real data")
        
        # Train subject-agnostic model
        ms_dgmm = MultiSubjectDGMM(alignment_method='ridge')
        ms_dgmm.fit(multi_subject_data)
        
        # Test prediction
        test_fmri = multi_subject_data['subject_0']['Y'][:5]
        predictions = ms_dgmm.predict(test_fmri)
        
        print(f"✓ Prediction successful: {predictions.shape}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")

def example_2_alignment_comparison():
    """Example 2: Compare alignment methods"""
    print("\nEXAMPLE 2: ALIGNMENT COMPARISON")
    print("=" * 35)
    
    try:
        from extended.real_data_adapter import RealDataAdapter
        from extended.alignment_methods import compare_alignment_methods
        
        # Load data
        adapter = RealDataAdapter()
        multi_data = adapter.create_multi_subject_data(n_subjects=3)
        fmri_data = {sid: data['Y'] for sid, data in multi_data.items()}
        
        # Compare methods
        methods = ['ridge', 'hyperalignment']
        results = compare_alignment_methods(fmri_data, methods)
        
        print("Comparison results:")
        for method, result in results.items():
            if 'error' not in result:
                metrics = result['metrics']
                print(f"  {method}: ISC = {metrics.get('isc', 0):.4f}")
            else:
                print(f"  {method}: Error")
                
    except Exception as e:
        print(f"✗ Error: {str(e)}")

def example_3_modular_workflow():
    """Example 3: Modular workflow demonstration"""
    print("\nEXAMPLE 3: MODULAR WORKFLOW")
    print("=" * 30)
    
    print("Step 1: Run alignment pipeline")
    print("  Command: python run_multi_subject_alignment.py")
    print("  Output: Saves results to outputs/ directory")
    
    print("\nStep 2: Use saved results")
    print("  Command: python use_aligned_data.py")
    print("  Benefit: Reuse without recomputation")
    
    print("\nStep 3: Load specific results")
    print("  Code example:")
    print("    from run_multi_subject_alignment import load_alignment_results")
    print("    data = load_alignment_results('outputs/alignment_*.npz')")
    
    # Check if we have saved results
    if os.path.exists('./outputs'):
        import glob
        files = glob.glob('./outputs/*_aligned_data.npz')
        if files:
            print(f"\n✓ Found {len(files)} saved alignment result(s)")
        else:
            print("\n! No saved results found - run main pipeline first")
    else:
        print("\n! No outputs directory - run main pipeline first")

def example_4_custom_analysis():
    """Example 4: Custom analysis with aligned data"""
    print("\nEXAMPLE 4: CUSTOM ANALYSIS")
    print("=" * 28)
    
    try:
        # Check for saved results
        import glob
        if not os.path.exists('./outputs'):
            print("! Run 'python run_multi_subject_alignment.py' first")
            return
            
        files = glob.glob('./outputs/*_aligned_data.npz')
        if not files:
            print("! No alignment results found")
            return
            
        # Load most recent results
        from run_multi_subject_alignment import load_alignment_results
        latest_file = max(files, key=os.path.getctime)
        aligned_data = load_alignment_results(latest_file)
        
        print(f"✓ Loaded alignment results: {len(aligned_data)} subjects")
        
        # Custom analysis example
        print("\nCustom analysis:")
        
        # 1. Calculate mean activation per subject
        for subject_id, data in aligned_data.items():
            mean_act = np.mean(data)
            print(f"  {subject_id}: mean activation = {mean_act:.4f}")
        
        # 2. Inter-subject correlation
        if len(aligned_data) >= 2:
            subjects = list(aligned_data.keys())
            data1 = aligned_data[subjects[0]]
            data2 = aligned_data[subjects[1]]
            
            min_samples = min(data1.shape[0], data2.shape[0])
            corr = np.corrcoef(data1[:min_samples].flatten(), 
                             data2[:min_samples].flatten())[0, 1]
            
            print(f"  Inter-subject correlation: {corr:.4f}")
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")

def example_5_production_workflow():
    """Example 5: Production workflow"""
    print("\nEXAMPLE 5: PRODUCTION WORKFLOW")
    print("=" * 32)
    
    print("Complete production workflow:")
    print("\n1. Data preparation:")
    print("   ✓ Real data: data/digit69_28x28.mat")
    print("   ✓ Multi-subject conversion: automatic")
    
    print("\n2. Alignment training:")
    print("   Command: python run_multi_subject_alignment.py")
    print("   Output: Trained models + aligned data")
    
    print("\n3. Downstream processing:")
    print("   Command: python use_aligned_data.py")
    print("   Features: Load, analyze, export")
    
    print("\n4. Collaboration:")
    print("   Share: outputs/ directory")
    print("   Benefit: Reproducible results")
    
    print("\n5. Integration:")
    print("   Format: NPZ (NumPy), JSON (metadata), PKL (models)")
    print("   Export: CSV for external tools")

def main():
    """Run all examples"""
    print("MULTI-SUBJECT ALIGNMENT EXAMPLES")
    print("=" * 40)
    print("Comprehensive examples for the modular alignment system")
    
    # Run examples
    example_1_basic_usage()
    example_2_alignment_comparison()
    example_3_modular_workflow()
    example_4_custom_analysis()
    example_5_production_workflow()
    
    print(f"\n{'='*40}")
    print("🎯 SUMMARY")
    print(f"{'='*40}")
    print("Key features demonstrated:")
    print("✅ Real data integration (digit69_28x28.mat)")
    print("✅ Multiple alignment methods")
    print("✅ Modular workflow (save/load)")
    print("✅ Subject-agnostic models")
    print("✅ Production-ready pipeline")
    
    print(f"\n📚 NEXT STEPS:")
    print("1. Run: python run_multi_subject_alignment.py")
    print("2. Explore: python use_aligned_data.py")
    print("3. Customize: Modify this script for your needs")

if __name__ == '__main__':
    main()
