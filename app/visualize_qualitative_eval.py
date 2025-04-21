import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
from matplotlib.gridspec import GridSpec

# Suppress warnings
warnings.filterwarnings('ignore')

# Set the style for our plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Define file paths
CLAUDE_FILE = '../data/evaluation_results_final/results_evaluated_by_claude_sonnet37.csv'
CHATGPT_FILE = '../data/evaluation_results_final/results_evaluated_by_chatgpt_o4mini.csv'
OUTPUT_DIR = '../data/evaluation_results_final/'


# Function to load the CSV files with validation
def load_data():
    """
    Load both CSV files into pandas DataFrames with data validation
    """
    print("Loading data...")

    try:
        claude_df = pd.read_csv(CLAUDE_FILE)
        chatgpt_df = pd.read_csv(CHATGPT_FILE)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None, None

    # Add a column to identify the evaluator
    claude_df['Evaluator'] = 'Claude 3.7 Sonnet'
    chatgpt_df['Evaluator'] = 'ChatGPT o4-mini'

    # Convert data types with validation
    for df in [claude_df, chatgpt_df]:
        # Convert Chunk Size to numeric
        df['Chunk Size'] = pd.to_numeric(df['Chunk Size'], errors='coerce')

        # Convert metric columns to numeric
        metric_columns = ['Context Quality', 'Answer Relevance', 'Answer Correctness', 'Faithfulness to Context']
        for col in metric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN values in critical columns
        df.dropna(subset=['Chunk Size'] + metric_columns, inplace=True)

    return claude_df, chatgpt_df


# Function to aggregate scores by chunk size
def aggregate_by_chunk_size(df, group_by=None):
    """
    Group by chunk size and optionally another column,
    then compute average scores for all metrics
    """
    metrics = ['Context Quality', 'Answer Relevance', 'Answer Correctness', 'Faithfulness to Context']

    try:
        if group_by:
            return df.groupby(['Chunk Size', group_by]).agg({
                'Context Quality': 'mean',
                'Answer Relevance': 'mean',
                'Answer Correctness': 'mean',
                'Faithfulness to Context': 'mean'
            }).reset_index()
        else:
            return df.groupby('Chunk Size').agg({
                'Context Quality': 'mean',
                'Answer Relevance': 'mean',
                'Answer Correctness': 'mean',
                'Faithfulness to Context': 'mean'
            }).reset_index()
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return None


# Function to calculate a combined score
def calculate_combined_score(df):
    """
    Calculate a combined score from all metrics
    """
    df_copy = df.copy()

    metrics = ['Context Quality', 'Answer Relevance', 'Answer Correctness', 'Faithfulness to Context']
    df_copy['Combined Score'] = df_copy[metrics].mean(axis=1)

    return df_copy


# APPROACH 1: Side-by-side plots for better comparison
def create_side_by_side_plots(claude_df, chatgpt_df, output_dir):
    """
    Create side-by-side plots of LLM models and embedding models
    for both Claude and ChatGPT evaluations
    """
    print("Creating side-by-side comparison plots...")

    # Create subdirectory for side-by-side plots
    side_dir = f"{output_dir}/side_by_side"
    if not os.path.exists(side_dir):
        os.makedirs(side_dir)

    # Add combined score to both dataframes
    claude_df = calculate_combined_score(claude_df)
    chatgpt_df = calculate_combined_score(chatgpt_df)

    # Get unique models
    llm_models = sorted(pd.concat([claude_df['LLM Model'], chatgpt_df['LLM Model']]).unique())
    embedding_models = sorted(pd.concat([claude_df['Embedding Model'], chatgpt_df['Embedding Model']]).unique())

    # Metrics to compare
    metrics = ['Context Quality', 'Answer Relevance', 'Answer Correctness',
               'Faithfulness to Context', 'Combined Score']

    # 1. Create side-by-side LLM comparison plots
    for metric in metrics:
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

        # Plot Claude data on the left
        for model in llm_models:
            model_df = claude_df[claude_df['LLM Model'] == model]
            if len(model_df) > 0:
                agg_data = model_df.groupby('Chunk Size')[metric].mean().reset_index()
                ax1.plot(agg_data['Chunk Size'], agg_data[metric], 'o-',
                         linewidth=2, label=f"{model}")

        # Plot ChatGPT data on the right
        for model in llm_models:
            model_df = chatgpt_df[chatgpt_df['LLM Model'] == model]
            if len(model_df) > 0:
                agg_data = model_df.groupby('Chunk Size')[metric].mean().reset_index()
                ax2.plot(agg_data['Chunk Size'], agg_data[metric], 'o-',
                         linewidth=2, label=f"{model}")

        # Set up the left subplot
        ax1.set_xscale('log', base=2)
        ax1.set_xlabel('Chunk Size (KB)')
        ax1.set_ylabel(f'Average {metric} Score')
        ax1.set_title(f'Claude 3.7 Sonnet Evaluation')
        ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylim(0, 5.5)
        ax1.set_xticks(sorted(claude_df['Chunk Size'].unique()))
        ax1.set_xticklabels(sorted(claude_df['Chunk Size'].unique()), rotation=45)
        ax1.legend()

        # Set up the right subplot
        ax2.set_xscale('log', base=2)
        ax2.set_xlabel('Chunk Size (KB)')
        ax2.set_title(f'ChatGPT o4-mini Evaluation')
        ax2.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylim(0, 5.5)
        ax2.set_xticks(sorted(chatgpt_df['Chunk Size'].unique()))
        ax2.set_xticklabels(sorted(chatgpt_df['Chunk Size'].unique()), rotation=45)
        ax2.legend()

        # Add main title
        plt.suptitle(f'LLM Model Comparison: {metric}', fontsize=16, y=1.02)

        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{side_dir}/llm_comparison_{metric.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Create side-by-side embedding model comparison plots
    for metric in metrics:
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

        # Plot Claude data on the left
        for model in embedding_models:
            model_df = claude_df[claude_df['Embedding Model'] == model]
            if len(model_df) > 0:
                agg_data = model_df.groupby('Chunk Size')[metric].mean().reset_index()
                model_name = model.split('/')[-1] if '/' in model else model
                ax1.plot(agg_data['Chunk Size'], agg_data[metric], 'o-',
                         linewidth=2, label=f"{model_name}")

        # Plot ChatGPT data on the right
        for model in embedding_models:
            model_df = chatgpt_df[chatgpt_df['Embedding Model'] == model]
            if len(model_df) > 0:
                agg_data = model_df.groupby('Chunk Size')[metric].mean().reset_index()
                model_name = model.split('/')[-1] if '/' in model else model
                ax2.plot(agg_data['Chunk Size'], agg_data[metric], 'o-',
                         linewidth=2, label=f"{model_name}")

        # Set up the left subplot
        ax1.set_xscale('log', base=2)
        ax1.set_xlabel('Chunk Size (KB)')
        ax1.set_ylabel(f'Average {metric} Score')
        ax1.set_title(f'Claude 3.7 Sonnet Evaluation')
        ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylim(0, 5.5)
        ax1.set_xticks(sorted(claude_df['Chunk Size'].unique()))
        ax1.set_xticklabels(sorted(claude_df['Chunk Size'].unique()), rotation=45)
        ax1.legend()

        # Set up the right subplot
        ax2.set_xscale('log', base=2)
        ax2.set_xlabel('Chunk Size (KB)')
        ax2.set_title(f'ChatGPT o4-mini Evaluation')
        ax2.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylim(0, 5.5)
        ax2.set_xticks(sorted(chatgpt_df['Chunk Size'].unique()))
        ax2.set_xticklabels(sorted(chatgpt_df['Chunk Size'].unique()), rotation=45)
        ax2.legend()

        # Add main title
        plt.suptitle(f'Embedding Model Comparison: {metric}', fontsize=16, y=1.02)

        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{side_dir}/embedding_comparison_{metric.replace(' ', '_').lower()}.png", dpi=300,
                    bbox_inches='tight')
        plt.close()


# APPROACH 2: Combined dataset plots
def create_combined_evaluator_plots(claude_df, chatgpt_df, output_dir):
    """
    Create plots that show both evaluators' assessments on the same graph
    """
    print("Creating combined evaluator plots...")

    # Create subdirectory
    combined_dir = f"{output_dir}/combined_evaluator"
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    # Combine the dataframes
    combined_df = pd.concat([claude_df, chatgpt_df])

    # Add combined score
    combined_df = calculate_combined_score(combined_df)

    # Get unique models
    llm_models = combined_df['LLM Model'].unique()
    embedding_models = combined_df['Embedding Model'].unique()

    # Metrics to compare
    metrics = ['Context Quality', 'Answer Relevance', 'Answer Correctness',
               'Faithfulness to Context', 'Combined Score']

    # 1. LLM Model comparison with both evaluators
    for metric in metrics:
        plt.figure(figsize=(14, 10))

        # Setup color palette
        model_colors = {llm_models[i]: plt.cm.tab10(i) for i in range(len(llm_models))}
        evaluator_markers = {'Claude 3.7 Sonnet': 'o', 'ChatGPT o4-mini': 's'}
        evaluator_linestyles = {'Claude 3.7 Sonnet': '-', 'ChatGPT o4-mini': '--'}

        # For each LLM model and evaluator combination
        for llm in llm_models:
            for evaluator in ['Claude 3.7 Sonnet', 'ChatGPT o4-mini']:
                # Filter data
                filtered_df = combined_df[(combined_df['LLM Model'] == llm) &
                                          (combined_df['Evaluator'] == evaluator)]

                if len(filtered_df) > 0:
                    # Aggregate by chunk size
                    agg_data = filtered_df.groupby('Chunk Size')[metric].mean().reset_index()

                    # Plot
                    plt.plot(agg_data['Chunk Size'], agg_data[metric],
                             marker=evaluator_markers[evaluator],
                             linestyle=evaluator_linestyles[evaluator],
                             linewidth=2,
                             color=model_colors[llm],
                             label=f"{llm} ({evaluator})")

        # Set plot properties
        plt.xscale('log', base=2)
        plt.xlabel('Chunk Size (KB)')
        plt.ylabel(f'Average {metric} Score')
        plt.title(f'LLM Model Comparison: {metric} (Both Evaluators)')
        plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
        plt.ylim(0, 5.5)
        plt.xticks(sorted(combined_df['Chunk Size'].unique()),
                   sorted(combined_df['Chunk Size'].unique()), rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{combined_dir}/llm_comparison_{metric.replace(' ', '_').lower()}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Embedding Model comparison with both evaluators
    for metric in metrics:
        plt.figure(figsize=(14, 10))

        # Setup color palette
        model_colors = {embedding_models[i]: plt.cm.tab10(i) for i in range(len(embedding_models))}
        evaluator_markers = {'Claude 3.7 Sonnet': 'o', 'ChatGPT o4-mini': 's'}
        evaluator_linestyles = {'Claude 3.7 Sonnet': '-', 'ChatGPT o4-mini': '--'}

        # For each embedding model and evaluator combination
        for emb in embedding_models:
            for evaluator in ['Claude 3.7 Sonnet', 'ChatGPT o4-mini']:
                # Filter data
                filtered_df = combined_df[(combined_df['Embedding Model'] == emb) &
                                          (combined_df['Evaluator'] == evaluator)]

                if len(filtered_df) > 0:
                    # Aggregate by chunk size
                    agg_data = filtered_df.groupby('Chunk Size')[metric].mean().reset_index()

                    # Get simplified model name
                    emb_name = emb.split('/')[-1] if '/' in emb else emb

                    # Plot
                    plt.plot(agg_data['Chunk Size'], agg_data[metric],
                             marker=evaluator_markers[evaluator],
                             linestyle=evaluator_linestyles[evaluator],
                             linewidth=2,
                             color=model_colors[emb],
                             label=f"{emb_name} ({evaluator})")

        # Set plot properties
        plt.xscale('log', base=2)
        plt.xlabel('Chunk Size (KB)')
        plt.ylabel(f'Average {metric} Score')
        plt.title(f'Embedding Model Comparison: {metric} (Both Evaluators)')
        plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
        plt.ylim(0, 5.5)
        plt.xticks(sorted(combined_df['Chunk Size'].unique()),
                   sorted(combined_df['Chunk Size'].unique()), rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{combined_dir}/embedding_comparison_{metric.replace(' ', '_').lower()}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()


# APPROACH 3: Grid plot comparison for complete overview - FIXED VERSION
def create_grid_plots(claude_df, chatgpt_df, output_dir):
    """
    Create a comprehensive grid of plots showing all combinations
    of models, evaluators, and metrics
    """
    print("Creating grid plots for comprehensive comparison...")

    # Create subdirectory
    grid_dir = f"{output_dir}/grid_plots"
    if not os.path.exists(grid_dir):
        os.makedirs(grid_dir)

    # Add combined score
    claude_df = calculate_combined_score(claude_df)
    chatgpt_df = calculate_combined_score(chatgpt_df)

    # Combine the dataframes
    combined_df = pd.concat([claude_df, chatgpt_df])

    # Get unique values
    llm_models = sorted(combined_df['LLM Model'].unique())
    embedding_models = sorted(combined_df['Embedding Model'].unique())
    evaluators = ['Claude 3.7 Sonnet', 'ChatGPT o4-mini']
    metrics = ['Context Quality', 'Answer Relevance', 'Answer Correctness',
               'Faithfulness to Context', 'Combined Score']

    # Create a matrix grid plot for all metrics, with LLM models
    # Using subplots instead of GridSpec with subplot_spec
    fig, axes = plt.subplots(len(metrics), 2, figsize=(20, 25), sharex=True)

    # Adjust the vertical spacing
    plt.subplots_adjust(hspace=0.4)

    for i, metric in enumerate(metrics):
        # Add a title for this metric
        fig.text(0.5, 0.95 - (i * 0.95 / len(metrics)), f"{metric}",
                 ha='center', fontsize=16)

        # Create subplots for Claude and ChatGPT
        for j, evaluator in enumerate(evaluators):
            ax = axes[i, j]

            # Filter data for this evaluator
            eval_df = combined_df[combined_df['Evaluator'] == evaluator]

            # Plot each LLM model
            for llm in llm_models:
                llm_df = eval_df[eval_df['LLM Model'] == llm]
                if len(llm_df) > 0:
                    agg_data = llm_df.groupby('Chunk Size')[metric].mean().reset_index()
                    ax.plot(agg_data['Chunk Size'], agg_data[metric], 'o-',
                            linewidth=2, label=f"{llm}")

            # Configure subplot
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Chunk Size (KB)')
            if j == 0:  # Only add y-label on the leftmost plot
                ax.set_ylabel(f'Score')
            ax.set_title(f"{evaluator}")
            ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(0, 5.5)
            ax.set_xticks(sorted(eval_df['Chunk Size'].unique()))
            ax.set_xticklabels(sorted(eval_df['Chunk Size'].unique()), rotation=45)
            ax.legend(loc='lower right')

    # Add main title
    fig.suptitle('LLM Model Comparison Across All Metrics', fontsize=20, y=0.98)

    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    plt.savefig(f"{grid_dir}/llm_model_grid_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Now do the same for embedding models
    fig, axes = plt.subplots(len(metrics), 2, figsize=(20, 25), sharex=True)

    # Adjust the vertical spacing
    plt.subplots_adjust(hspace=0.4)

    for i, metric in enumerate(metrics):
        # Add a title for this metric
        fig.text(0.5, 0.95 - (i * 0.95 / len(metrics)), f"{metric}",
                 ha='center', fontsize=16)

        # Create subplots for Claude and ChatGPT
        for j, evaluator in enumerate(evaluators):
            ax = axes[i, j]

            # Filter data for this evaluator
            eval_df = combined_df[combined_df['Evaluator'] == evaluator]

            # Plot each embedding model
            for emb in embedding_models:
                emb_df = eval_df[eval_df['Embedding Model'] == emb]
                if len(emb_df) > 0:
                    agg_data = emb_df.groupby('Chunk Size')[metric].mean().reset_index()
                    emb_name = emb.split('/')[-1] if '/' in emb else emb
                    ax.plot(agg_data['Chunk Size'], agg_data[metric], 'o-',
                            linewidth=2, label=f"{emb_name}")

            # Configure subplot
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Chunk Size (KB)')
            if j == 0:  # Only add y-label on the leftmost plot
                ax.set_ylabel(f'Score')
            ax.set_title(f"{evaluator}")
            ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(0, 5.5)
            ax.set_xticks(sorted(eval_df['Chunk Size'].unique()))
            ax.set_xticklabels(sorted(eval_df['Chunk Size'].unique()), rotation=45)
            ax.legend(loc='lower right')

    # Add main title
    fig.suptitle('Embedding Model Comparison Across All Metrics', fontsize=20, y=0.98)

    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    plt.savefig(f"{grid_dir}/embedding_model_grid_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


# Main function to run the analysis
def run_analysis():
    """
    Run the improved comparison analyses
    """
    print("Loading data...")
    claude_df, chatgpt_df = load_data()

    if claude_df is None or chatgpt_df is None:
        print("Failed to load data. Exiting.")
        return

    # Create side-by-side plots
    create_side_by_side_plots(claude_df, chatgpt_df, OUTPUT_DIR)

    # Create combined evaluator plots
    create_combined_evaluator_plots(claude_df, chatgpt_df, OUTPUT_DIR)

    # Create grid plots for comprehensive overview
    create_grid_plots(claude_df, chatgpt_df, OUTPUT_DIR)

    print(f"All analyses complete! Results saved to {OUTPUT_DIR}")


# Run the analysis if this script is run directly
if __name__ == "__main__":
    run_analysis()