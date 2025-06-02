import os
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from bert_score import score as bert_score
import torch  # Needed for bert-score

# Set a less memory-intensive BERT model for bert-score if needed
# os.environ["BERT_SCORE_MODEL"] = "distilbert-base-uncased"

# Define paths
base_dir = "/home/joshua-krafft/thesis/summaries"
ai_systems = ["GPT 3.5 Turbo", "Gemini 1.5 pro"]
asr_tools = ["Whisper", "Speechmatics", "Google", "Kaldi_NL"]
consult_numbers = range(1, 11)  # 1 through 10

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
print("Starting evaluation...")
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
        P, R, F1 = bert_score(batch_ai_summaries, [ref_summary] * len(batch_ai_summaries), 
                             lang="en", verbose=False, rescale_with_baseline=True)
        
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

# Generate summary statistics - specify numeric columns only
numeric_cols = ['ROUGE-1-F', 'ROUGE-1-P', 'ROUGE-1-R', 
                'ROUGE-2-F', 'ROUGE-2-P', 'ROUGE-2-R',
                'ROUGE-L-F', 'ROUGE-L-P', 'ROUGE-L-R',
                'BERT-Score-P', 'BERT-Score-R', 'BERT-Score-F1',
                'AI_Word_Count', 'Ref_Word_Count']

# Calculate means for numeric columns only
summary_by_ai_asr = df.groupby(['AI', 'ASR'])[numeric_cols].mean().reset_index()
summary_by_ai = df.groupby(['AI'])[numeric_cols].mean().reset_index()
summary_by_asr = df.groupby(['ASR'])[numeric_cols].mean().reset_index()

# Save summary statistics
summary_by_ai_asr.to_csv("scores_by_ai_asr.csv", index=False)
summary_by_ai.to_csv("scores_by_ai.csv", index=False)
summary_by_asr.to_csv("scores_by_asr.csv", index=False)

print("Creating visualizations...")

# Create visualizations
# 1. Heatmap of BERT-Score F1 scores by AI and ASR
plt.figure(figsize=(10, 6))
heatmap_data = df.pivot_table(index='AI', columns='ASR', values='BERT-Score-F1')
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('BERT-Score F1 by AI System and ASR Tool')
plt.tight_layout()
plt.savefig('bertscore_heatmap.png')

# 2. Heatmap of ROUGE-1 F1 scores by AI and ASR
plt.figure(figsize=(10, 6))
heatmap_data = df.pivot_table(index='AI', columns='ASR', values='ROUGE-1-F')
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('ROUGE-1 F1 Scores by AI System and ASR Tool')
plt.tight_layout()
plt.savefig('rouge1_heatmap.png')

# 3. Bar chart comparing AI systems across all metrics
plt.figure(figsize=(14, 8))
metrics = ['ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-L-F', 'BERT-Score-F1']
ai_data = summary_by_ai[['AI'] + metrics]
ai_data_melted = pd.melt(ai_data, id_vars=['AI'], value_vars=metrics, 
                          var_name='Metric', value_name='Score')
sns.barplot(data=ai_data_melted, x='AI', y='Score', hue='Metric')
plt.title('Average Performance Metrics by AI System')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('ai_comparison.png')

# 4. Bar chart comparing ASR tools across all metrics
plt.figure(figsize=(14, 8))
asr_data = summary_by_asr[['ASR'] + metrics]
asr_data_melted = pd.melt(asr_data, id_vars=['ASR'], value_vars=metrics, 
                           var_name='Metric', value_name='Score')
sns.barplot(data=asr_data_melted, x='ASR', y='Score', hue='Metric')
plt.title('Average Performance Metrics by ASR Tool')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('asr_comparison.png')

# 5. Box plots to show distribution of BERT-Score F1 by AI system
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='AI', y='BERT-Score-F1')
plt.title('Distribution of BERT-Score F1 by AI System')
plt.tight_layout()
plt.savefig('bertscore_distribution_by_ai.png')

# 6. Box plots to show distribution of BERT-Score F1 by ASR tool
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='ASR', y='BERT-Score-F1')
plt.title('Distribution of BERT-Score F1 by ASR Tool')
plt.tight_layout()
plt.savefig('bertscore_distribution_by_asr.png')

# 7. Scatter plot comparing ROUGE-1 vs BERT-Score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='ROUGE-1-F', y='BERT-Score-F1', hue='AI', style='ASR')
plt.title('ROUGE-1 F1 vs BERT-Score F1')
plt.xlabel('ROUGE-1 F1')
plt.ylabel('BERT-Score F1')
plt.tight_layout()
plt.savefig('rouge_vs_bertscore.png')

# 8. Create a table showing best and worst combinations for all metrics
metrics_list = [
    ('ROUGE-1-F', 'ROUGE-1 F1'), 
    ('ROUGE-2-F', 'ROUGE-2 F1'), 
    ('ROUGE-L-F', 'ROUGE-L F1'),
    ('BERT-Score-F1', 'BERT-Score F1')
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

# 9. Statistical significance testing between AI systems
from scipy.stats import wilcoxon

# For each metric, test if there's a significant difference between AI systems
significance_results = []

for metric in metrics:
    gpt_scores = df[df['AI'] == 'GPT 3.5 Turbo'][metric]
    gemini_scores = df[df['AI'] == 'Gemini 1.5 pro'][metric]
    
    # Perform Wilcoxon signed-rank test
    try:
        stat, p_value = wilcoxon(gpt_scores, gemini_scores)
        
        # Determine which is better
        if gpt_scores.mean() > gemini_scores.mean():
            better_system = 'GPT 3.5 Turbo'
        else:
            better_system = 'Gemini 1.5 pro'
            
        significance_results.append({
            'Metric': metric,
            'P_Value': p_value,
            'Significant': p_value < 0.05,
            'Better_System': better_system if p_value < 0.05 else 'No significant difference'
        })
    except Exception as e:
        significance_results.append({
            'Metric': metric,
            'P_Value': None,
            'Significant': False,
            'Better_System': f'Error in test: {str(e)}'
        })

# Save significance test results
pd.DataFrame(significance_results).to_csv("significance_tests.csv", index=False)

print("Analysis complete. Results saved to CSVs and PNGs.")