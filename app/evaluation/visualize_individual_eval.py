import json
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

# Either manually list files or use glob to auto-find them
json_folder = '../../data/individual_evaluation_results/'  # Folder containing JSON files
json_files = glob(os.path.join(json_folder, '*.json'))

def plot_scores_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Remove total score dict if present
    if "total_output_score" in data[-1]:
        data = data[:-1]

    # Extract data
    indices = [item['index'] for item in data]
    output_scores = [item['output_score'] for item in data]
    retrieval_scores = [item['retrieval_score'] for item in data]

    x = np.arange(len(indices))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.bar(x, output_scores, width, label='Output Score', color='#1f77b4')
    ax.bar(x, retrieval_scores, width, bottom=output_scores, label='Retrieval Score', color='#ff7f0e')

    # Labels & Titles
    ax.set_xlabel('Index')
    ax.set_ylabel('Score')
    ax.set_title(f'Stacked Scores: {os.path.basename(json_path)}')
    ax.set_xticks(x)
    ax.set_xticklabels(indices)
    ax.legend()

    plt.tight_layout()

    # Save image
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_path = f'{base_name}.png'
    plt.savefig(output_path, dpi=300)
    plt.close()  

    print(f"Saved plot: {output_path}")


for file_path in json_files:
    plot_scores_from_json(file_path)
