import os
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from scipy.stats import wilcoxon

# Define paths
base_dir = "/home/joshua-krafft/thesis/summaries"
ai_systems = ["GPT 3.5 Turbo", "Gemini 1.5 pro"]
asr_tools = ["Whisper", "Speechmatics", "Google", "Kaldi_NL"]
consult_numbers = range(1, 11) 

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Function to read a text file
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# Prepare data structure for results
results = []

# Process each consultation
for consult_num in consult_numbers:
    print(f"Processing consultation {consult_num}...")
    # Read the reference summary
    ref_path = os.path.join(base_dir, "Reference", f"consult{consult_num}_SOEP_summary.txt")
    ref_summary = read_file(ref_path)
    
    # Skip if reference is empty
    if not ref_summary:
        print(f"Warning: Reference summary for consult{consult_num} is empty. Skipping.")
        continue
    
    # Collect all summaries for this consultation for batch BERT scoring
    batch_ai_summaries = []
    batch_metadata = []  # Store AI and ASR info for each summary
    
    # Process each AI system and ASR tool combination
    for ai in ai_systems:
        for asr in asr_tools:
            ai_path = os.path.join(base_dir, ai, asr, f"consult{consult_num}_SOEP_summary.txt")
            ai_summary = read_file(ai_path)
            
            if not ai_summary:
                print(f"Warning: {ai} + {asr} summary for consult{consult_num} is empty. Skipping.")
                continue
                
            # Calculate ROUGE scores
            scores = scorer.score(ref_summary, ai_summary)
            
            # Store the summary for batch processing
            batch_ai_summaries.append(ai_summary)
            batch_metadata.append((ai, asr, scores, len(ai_summary.split())))
    
    # Calculate BERT-Score for all summaries of this consultation at once
    if batch_ai_summaries:
        print(f"  Calculating BERT-Scores for consultation {consult_num}...")
        
        # Raw scores (without baseline rescaling)
        P_raw, R_raw, F1_raw = bert_score(
            batch_ai_summaries, 
            [ref_summary] * len(batch_ai_summaries), 
            lang="en", verbose=False, rescale_with_baseline=False
        )
        
        # Rescaled scores (with baseline rescaling)
        P, R, F1 = bert_score(
            batch_ai_summaries, 
            [ref_summary] * len(batch_ai_summaries), 
            lang="en", verbose=False, rescale_with_baseline=True
        )
        
        # Add results for each summary
        for i, (ai, asr, rouge_scores, word_count) in enumerate(batch_metadata):
            results.append({
                'Consult': consult_num,
                'AI': ai,
                'ASR': asr,
                'ROUGE-1-F': rouge_scores['rouge1'].fmeasure,
                'ROUGE-1-P': rouge_scores['rouge1'].precision,
                'ROUGE-1-R': rouge_scores['rouge1'].recall,
                'ROUGE-2-F': rouge_scores['rouge2'].fmeasure,
                'ROUGE-2-P': rouge_scores['rouge2'].precision,
                'ROUGE-2-R': rouge_scores['rouge2'].recall,
                'ROUGE-L-F': rouge_scores['rougeL'].fmeasure,
                'ROUGE-L-P': rouge_scores['rougeL'].precision,
                'ROUGE-L-R': rouge_scores['rougeL'].recall,
                'BERT-Score-P-Raw': P_raw[i].item(),
                'BERT-Score-R-Raw': R_raw[i].item(),
                'BERT-Score-F1-Raw': F1_raw[i].item(),
                'BERT-Score-P': P[i].item(),
                'BERT-Score-R': R[i].item(),
                'BERT-Score-F1': F1[i].item(),
                'AI_Word_Count': word_count,
                'Ref_Word_Count': len(ref_summary.split())
            })

# Convert to DataFrame
df = pd.DataFrame(results)

# Save raw results to CSV
df.to_csv("evaluation_scores_all.csv", index=False)

# Generate summary statistics - specify numeric columns only (INCLUDING raw BERT-Scores)
numeric_cols = ['ROUGE-1-F', 'ROUGE-1-P', 'ROUGE-1-R', 
                'ROUGE-2-F', 'ROUGE-2-P', 'ROUGE-2-R',
                'ROUGE-L-F', 'ROUGE-L-P', 'ROUGE-L-R',
                'BERT-Score-P-Raw', 'BERT-Score-R-Raw', 'BERT-Score-F1-Raw',
                'BERT-Score-P', 'BERT-Score-R', 'BERT-Score-F1',
                'AI_Word_Count', 'Ref_Word_Count']

# Calculate means AND standard deviations for numeric columns
summary_by_ai_asr = df.groupby(['AI', 'ASR'])[numeric_cols].agg(['mean', 'std']).reset_index()
summary_by_ai = df.groupby(['AI'])[numeric_cols].agg(['mean', 'std']).reset_index()
summary_by_asr = df.groupby(['ASR'])[numeric_cols].agg(['mean', 'std']).reset_index()

# Flatten the column names for easier reading
summary_by_ai_asr.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary_by_ai_asr.columns.values]
summary_by_ai.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary_by_ai.columns.values]
summary_by_asr.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary_by_asr.columns.values]

# Save summary statistics with standard deviations
summary_by_ai_asr.to_csv("scores_by_ai_asr.csv", index=False)
summary_by_ai.to_csv("scores_by_ai.csv", index=False)
summary_by_asr.to_csv("scores_by_asr.csv", index=False)

# Create detailed standard deviation analysis
print("Creating detailed standard deviation analysis...")
std_analysis = []

# By AI system
for ai in ai_systems:
    ai_data = df[df['AI'] == ai]
    for metric in numeric_cols:
        if metric not in ['AI_Word_Count', 'Ref_Word_Count']:  # Skip word count metrics
            std_analysis.append({
                'Group_Type': 'AI_System',
                'Group_Name': ai,
                'Metric': metric,
                'Mean': ai_data[metric].mean(),
                'Std': ai_data[metric].std(),
                'CV': ai_data[metric].std() / ai_data[metric].mean() if ai_data[metric].mean() != 0 else np.nan,
                'Min': ai_data[metric].min(),
                'Max': ai_data[metric].max(),
                'Sample_Size': len(ai_data)
            })

# By ASR tool
for asr in asr_tools:
    asr_data = df[df['ASR'] == asr]
    for metric in numeric_cols:
        if metric not in ['AI_Word_Count', 'Ref_Word_Count']:  # Skip word count metrics
            std_analysis.append({
                'Group_Type': 'ASR_Tool',
                'Group_Name': asr,
                'Metric': metric,
                'Mean': asr_data[metric].mean(),
                'Std': asr_data[metric].std(),
                'CV': asr_data[metric].std() / asr_data[metric].mean() if asr_data[metric].mean() != 0 else np.nan,
                'Min': asr_data[metric].min(),
                'Max': asr_data[metric].max(),
                'Sample_Size': len(asr_data)
            })

# By AI+ASR combination
for ai in ai_systems:
    for asr in asr_tools:
        combo_data = df[(df['AI'] == ai) & (df['ASR'] == asr)]
        if len(combo_data) > 0:  # Only if data exists
            for metric in numeric_cols:
                if metric not in ['AI_Word_Count', 'Ref_Word_Count']:  # Skip word count metrics
                    std_analysis.append({
                        'Group_Type': 'AI_ASR_Combo',
                        'Group_Name': f'{ai} + {asr}',
                        'Metric': metric,
                        'Mean': combo_data[metric].mean(),
                        'Std': combo_data[metric].std(),
                        'CV': combo_data[metric].std() / combo_data[metric].mean() if combo_data[metric].mean() != 0 else np.nan,
                        'Min': combo_data[metric].min(),
                        'Max': combo_data[metric].max(),
                        'Sample_Size': len(combo_data)
                    })

# Save detailed standard deviation analysis
pd.DataFrame(std_analysis).to_csv("standard_deviation_analysis.csv", index=False)

# Create a summary table with key metrics and their variability (INCLUDING raw BERT-Scores)
print("Creating variability summary...")
key_metrics = ['ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-L-F', 'BERT-Score-F1-Raw', 'BERT-Score-F1']
variability_summary = []

for metric in key_metrics:
    # Overall statistics
    overall_mean = df[metric].mean()
    overall_std = df[metric].std()
    overall_cv = overall_std / overall_mean if overall_mean != 0 else np.nan
    
    # Find most and least variable combinations
    combo_stats = df.groupby(['AI', 'ASR'])[metric].agg(['mean', 'std']).reset_index()
    combo_stats['cv'] = combo_stats['std'] / combo_stats['mean']
    combo_stats = combo_stats.dropna()
    
    if len(combo_stats) > 0:
        most_variable_idx = combo_stats['cv'].idxmax()
        least_variable_idx = combo_stats['cv'].idxmin()
        
        most_variable = f"{combo_stats.loc[most_variable_idx, 'AI']} + {combo_stats.loc[most_variable_idx, 'ASR']}"
        least_variable = f"{combo_stats.loc[least_variable_idx, 'AI']} + {combo_stats.loc[least_variable_idx, 'ASR']}"
        
        variability_summary.append({
            'Metric': metric,
            'Overall_Mean': overall_mean,
            'Overall_Std': overall_std,
            'Overall_CV': overall_cv,
            'Most_Variable_Combo': most_variable,
            'Most_Variable_CV': combo_stats.loc[most_variable_idx, 'cv'],
            'Least_Variable_Combo': least_variable,
            'Least_Variable_CV': combo_stats.loc[least_variable_idx, 'cv']
        })

pd.DataFrame(variability_summary).to_csv("variability_summary.csv", index=False)

# Create a table showing best and worst combinations for all metrics (INCLUDING raw BERT-Scores)
metrics_list = [
    ('ROUGE-1-F', 'ROUGE-1 F1'), 
    ('ROUGE-2-F', 'ROUGE-2 F1'), 
    ('ROUGE-L-F', 'ROUGE-L F1'),
    ('BERT-Score-F1-Raw', 'BERT-Score F1 (Raw)'),
    ('BERT-Score-F1', 'BERT-Score F1 (Rescaled)')
]

best_worst = []
for metric_col, metric_name in metrics_list:
    best_idx = df[metric_col].idxmax()
    worst_idx = df[metric_col].idxmin()
    
    best_worst.append({
        'Metric': metric_name,
        'Best_Score': df.loc[best_idx, metric_col],
        'Best_Combination': f"{df.loc[best_idx, 'AI']} + {df.loc[best_idx, 'ASR']}",
        'Worst_Score': df.loc[worst_idx, metric_col],
        'Worst_Combination': f"{df.loc[worst_idx, 'AI']} + {df.loc[worst_idx, 'ASR']}"
    })

best_worst_df = pd.DataFrame(best_worst)
best_worst_df.to_csv("best_worst_combinations.csv", index=False)

# Statistical significance testing between AI systems (INCLUDING raw BERT-Scores)
print("Performing statistical significance tests...")
# For each metric, test if there's a significant difference between AI systems
significance_results = []

for metric in key_metrics:  # Now includes raw BERT-Scores
    gpt_scores = df[df['AI'] == 'GPT 3.5 Turbo'][metric]
    gemini_scores = df[df['AI'] == 'Gemini 1.5 pro'][metric]
    
    # Perform Wilcoxon signed-rank test
    try:
        stat, p_value = wilcoxon(gpt_scores, gemini_scores)
        
        # Determine which is better
        if gpt_scores.mean() > gemini_scores.mean():
            better_system = 'GPT 3.5 Turbo'
            mean_diff = gpt_scores.mean() - gemini_scores.mean()
        else:
            better_system = 'Gemini 1.5 pro'
            mean_diff = gemini_scores.mean() - gpt_scores.mean()
            
        significance_results.append({
            'Metric': metric,
            'P_Value': p_value,
            'Significant': p_value < 0.05,
            'Better_System': better_system if p_value < 0.05 else 'No significant difference',
            'Mean_Difference': mean_diff,
            'GPT_Mean': gpt_scores.mean(),
            'GPT_Std': gpt_scores.std(),
            'Gemini_Mean': gemini_scores.mean(),
            'Gemini_Std': gemini_scores.std()
        })
    except Exception as e:
        significance_results.append({
            'Metric': metric,
            'P_Value': None,
            'Significant': False,
            'Better_System': f'Error in test: {str(e)}',
            'Mean_Difference': None,
            'GPT_Mean': gpt_scores.mean() if not gpt_scores.empty else None,
            'GPT_Std': gpt_scores.std() if not gpt_scores.empty else None,
            'Gemini_Mean': gemini_scores.mean() if not gemini_scores.empty else None,
            'Gemini_Std': gemini_scores.std() if not gemini_scores.empty else None
        })

# Save significance test results
pd.DataFrame(significance_results).to_csv("significance_tests.csv", index=False)

print("Analysis complete! Generated files:")
print("- evaluation_scores_all.csv: All individual scores (with raw and rescaled BERT-Scores)")
print("- scores_by_ai_asr.csv: Means and std devs by AI+ASR combination")
print("- scores_by_ai.csv: Means and std devs by AI system")
print("- scores_by_asr.csv: Means and std devs by ASR tool")
print("- standard_deviation_analysis.csv: Detailed variability analysis")
print("- variability_summary.csv: Summary of most/least variable combinations")
print("- best_worst_combinations.csv: Best and worst performing combinations (raw + rescaled)")
print("- significance_tests.csv: Statistical significance tests with std devs")