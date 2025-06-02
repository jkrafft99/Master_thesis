#!/usr/bin/env python3
import os
import sys
import glob
import jiwer
import re
import csv
from datetime import datetime

# Medical terms list
MEDICAL_TERMS = [
    'diarree', 'ontlasting', 'buikpijn', 'waterig', 'gastro-enteritis', 
    'maag-darmontsteking', 'paracetamol', 'dioralyte', 'ors', 'koorts',
    'astma', 'inhalator', 'virus', 'bacterie', 'ontlastingsonderzoek',
    'braaksel', 'braken', 'gebraakt', 'overgegeven',
    'atriumfibrilleren', 'boezemfibrilleren', 'cardioversie', 'thoracoscopische', 'ablatie',
    'apixaban', 'metoprolol', 'bètablokker', 'ace-remmer', 'bloeddruk',
    'bloedverdunners', 'hartinfarct', 'hartfunctie', 'hartritme', 'hartslag',
    'longaders', 'borstkas', 'ct-scan', 'mri', 'röntgenstraling', 'röntgenfoto', 
    'scan', 'echo', 'biopt', 'longen', 'longkanker', 'prostaat', 'prostaatkanker',
    'knie', 'knieprothese', 'spataderen', 'amputatie', 'infectie', 'antibiotica',
    'chemotherapie', 'palliatieve', 'operatie', 'chirurg'
]

def preprocess_text(text):
    """Preprocess text for WER calculation"""
    text = text.lower()
    text = re.sub(r'[.,;:!?()"\'_\-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_wer_metrics(ref_text, hyp_text):
    """Calculate comprehensive WER metrics"""
    ref_processed = preprocess_text(ref_text)
    hyp_processed = preprocess_text(hyp_text)
    
    # Basic WER
    wer = jiwer.wer(ref_processed, hyp_processed)
    measures = jiwer.compute_measures(ref_processed, hyp_processed)
    
    # Medical terms analysis
    ref_medical = [word for word in ref_processed.split() 
                   if any(term in word for term in MEDICAL_TERMS)]
    hyp_medical = [word for word in hyp_processed.split() 
                   if any(term in word for term in MEDICAL_TERMS)]
    
    medical_wer = 0
    if ref_medical:
        medical_ref = " ".join(ref_medical)
        medical_hyp = " ".join(hyp_medical) if hyp_medical else ""
        medical_wer = jiwer.wer(medical_ref, medical_hyp)
    
    # Count words in reference
    ref_words = ref_processed.split()
    total_words = len(ref_words)
    
    return {
        'wer': wer,
        'mer': measures.get('mer', 0),
        'wil': measures.get('wil', 0),
        'insertions': measures.get('insertions', 0),
        'deletions': measures.get('deletions', 0),
        'substitutions': measures.get('substitutions', 0),
        'total_words': total_words,
        'medical_wer': medical_wer,
        'medical_words': len(ref_medical)
    }

def main():
    # Define all ASR systems to compare
    systems = {
        'kaldi_nl': '~/Kaldi_NL/evaluation/hypotheses_clean',
        'speechmatics': '~/Kaldi_NL/evaluation/speechmatics',
        'whisper': '~/Kaldi_NL/evaluation/whisper',
        'google': '~/Kaldi_NL/evaluation/google'
    }
    
    # Expand paths and check existence
    available_systems = {}
    for system, path in systems.items():
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            files = glob.glob(os.path.join(expanded_path, '*.txt'))
            if files:
                available_systems[system] = expanded_path
                print(f"Found {len(files)} files for {system}")
        else:
            print(f"Warning: Directory not found for {system}: {path}")
            print(f"         Skipping {system}")
    
    # Reference directory
    ref_dir = os.path.expanduser('~/Kaldi_NL/evaluation/references_clean')
    ref_files = glob.glob(os.path.join(ref_dir, '*.txt'))
    
    if not ref_files:
        print("Error: No reference files found!")
        return
    
    print(f"\nFound {len(ref_files)} reference files")
    print("Starting evaluation...\n")
    
    # Calculate metrics for each system
    results = {}
    system_totals = {}
    
    # Initialize totals
    for system in available_systems:
        system_totals[system] = {
            'total_words': 0,
            'total_errors': 0,
            'medical_words': 0,
            'medical_errors': 0,
            'files': 0
        }
    
    # Process each file
    for filename in [os.path.basename(f) for f in ref_files]:
        ref_path = os.path.join(ref_dir, filename)
        
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref_text = f.read()
        
        results[filename] = {}
        
        for system, system_path in available_systems.items():
            hyp_path = os.path.join(system_path, filename)
            
            if os.path.exists(hyp_path):
                with open(hyp_path, 'r', encoding='utf-8') as f:
                    hyp_text = f.read()
                
                metrics = calculate_wer_metrics(ref_text, hyp_text)
                results[filename][system] = metrics
                
                # Update totals
                system_totals[system]['files'] += 1
                system_totals[system]['total_words'] += metrics['total_words']
                system_totals[system]['total_errors'] += (metrics['insertions'] + 
                                                         metrics['deletions'] + 
                                                         metrics['substitutions'])
                system_totals[system]['medical_words'] += metrics['medical_words']
                system_totals[system]['medical_errors'] += int(metrics['medical_wer'] * metrics['medical_words'])
    
    # Calculate aggregate metrics
    aggregate_results = {}
    for system, totals in system_totals.items():
        if totals['total_words'] > 0:
            aggregate_results[system] = {
                'wer': totals['total_errors'] / totals['total_words'],
                'medical_wer': totals['medical_errors'] / totals['medical_words'] if totals['medical_words'] > 0 else 0,
                'files': totals['files']
            }
    
    # Save results
    output_dir = os.path.expanduser('~/Kaldi_NL/evaluation/final_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Write detailed report
    report_file = os.path.join(output_dir, f"asr_comparison_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("ASR SYSTEMS COMPARISON REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reference files: {len(ref_files)}\n")
        f.write(f"Systems compared: {', '.join(aggregate_results.keys())}\n")
        f.write("="*60 + "\n\n")
        
        # Sort systems by WER
        sorted_systems = sorted(aggregate_results.items(), key=lambda x: x[1]['wer'])
        
        f.write("OVERALL PERFORMANCE RANKING\n")
        f.write("-"*30 + "\n")
        for i, (system, metrics) in enumerate(sorted_systems, 1):
            f.write(f"{i}. {system.upper()}\n")
            f.write(f"   Overall WER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)\n")
            f.write(f"   Medical Terms WER: {metrics['medical_wer']:.4f} ({metrics['medical_wer']*100:.2f}%)\n")
            f.write(f"   Files analyzed: {metrics['files']}\n\n")
        
        f.write("\nDETAILED FILE-BY-FILE RESULTS\n")
        f.write("-"*30 + "\n")
        for filename in sorted(results.keys()):
            f.write(f"\n{filename}:\n")
            if results[filename]:
                file_systems = sorted(results[filename].items(), key=lambda x: x[1]['wer'])
                for system, metrics in file_systems:
                    f.write(f"  {system}: WER {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
                    f.write(f" | Medical WER: {metrics['medical_wer']:.4f}\n")
    
    # Write CSV for data analysis
    csv_file = os.path.join(output_dir, f"asr_comparison_{timestamp}.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['filename', 'system', 'wer', 'mer', 'wil', 'insertions', 
                  'deletions', 'substitutions', 'medical_wer', 'medical_words']
        writer.writerow(header)
        
        # Data
        for filename in results:
            for system, metrics in results[filename].items():
                row = [filename, system] + [metrics.get(h, 0) for h in header[2:]]
                writer.writerow(row)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS - ASR SYSTEMS COMPARISON")
    print("="*60)
    
    for i, (system, metrics) in enumerate(sorted_systems, 1):
        print(f"{i}. {system.upper()}")
        print(f"   Overall WER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
        print(f"   Medical Terms WER: {metrics['medical_wer']:.4f} ({metrics['medical_wer']*100:.2f}%)")
        print(f"   Files analyzed: {metrics['files']}")
        print()
    
    print(f"Detailed report saved to: {report_file}")
    print(f"CSV data saved to: {csv_file}")

if __name__ == "__main__":
    main()
