import os
import json
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


# Function to load evaluation results
def load_evaluation_results(eval_dir):
    """Load all evaluation result files from the specified directory."""
    result_files = list(Path(eval_dir).glob("evaluation_results_*.json"))

    if not result_files:
        print(f"No evaluation result files found in {eval_dir}")
        return None

    all_results = []

    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                all_results.extend(results)
        except Exception as e:
            print(f"Error loading {result_file}: {str(e)}")

    return all_results


# Function to convert results to DataFrame
def results_to_dataframe(results):
    """Convert evaluation results to a pandas DataFrame for easier analysis."""
    if not results:
        return None

    rows = []

    for result in results:
        config = result.get("config", {})
        metrics = result.get("metrics", {})

        # Basic info
        row = {
            "question": result.get("question", ""),
            "llm_model": config.get("llm_model", ""),
            "embedding_model": config.get("embedding_model", "").split("/")[-1],
            "chunk_size": config.get("chunk_size", ""),
            "spacy_model": config.get("spacy_model", "")
        }

        # Add metrics
        for metric_name, metric_data in metrics.items():
            row[f"{metric_name}_score"] = metric_data.get("score")

        rows.append(row)

    return pd.DataFrame(rows)


# Function to generate visualizations
def generate_visualizations(df, output_dir):
    """Generate various visualizations from the results DataFrame."""
    if df is None or df.empty:
        print("No data available for visualization")
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set the style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    # 1. Bar chart comparing metrics across LLM models
    plot_metric_comparison_by_llm(df, output_dir, timestamp)

    # 2. Line charts showing impact of chunk size
    plot_chunk_size_impact(df, output_dir, timestamp)

    # 3. Heatmap of metrics by embedding model and chunk size
    plot_heatmap_by_embedding_and_chunk(df, output_dir, timestamp)

    # 4. Box plots of metric distributions
    plot_metric_distributions(df, output_dir, timestamp)

    # 5. Generate summary tables
    generate_summary_tables(df, output_dir, timestamp)

    print(f"All visualizations saved to {output_dir}")


def plot_metric_comparison_by_llm(df, output_dir, timestamp):
    """Plot bar charts comparing metrics across different LLM models."""
    metric_cols = [col for col in df.columns if col.endswith('_score')]

    plt.figure(figsize=(12, 8))

    grouped = df.groupby('llm_model')[metric_cols].mean().reset_index()
    melted = pd.melt(grouped, id_vars=['llm_model'], value_vars=metric_cols,
                     var_name='Metric', value_name='Score')
    melted['Metric'] = melted['Metric'].str.replace('_score', '')

    g = sns.barplot(x='llm_model', y='Score', hue='Metric', data=melted)
    g.set_title('Average Metric Scores by LLM Model')
    g.set_xlabel('LLM Model')
    g.set_ylabel('Average Score')
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'llm_comparison_{timestamp}.png'), dpi=300)
    plt.close()


def plot_chunk_size_impact(df, output_dir, timestamp):
    """Plot line charts showing the impact of chunk size on metrics."""
    metric_cols = [col for col in df.columns if col.endswith('_score')]

    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metric_cols):
        plt.subplot(2, 3, i + 1)

        for llm in df['llm_model'].unique():
            for emb in df['embedding_model'].unique():
                subset = df[(df['llm_model'] == llm) & (df['embedding_model'] == emb)]
                grouped = subset.groupby('chunk_size')[metric].mean().reset_index()
                plt.plot(grouped['chunk_size'], grouped[metric],
                         marker='o', label=f'{llm}, {emb}')

        metric_name = metric.replace('_score', '')
        plt.title(f'Impact of Chunk Size on {metric_name}')
        plt.xlabel('Chunk Size (KB)')
        plt.ylabel('Average Score')
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'chunk_size_impact_{timestamp}.png'), dpi=300)
    plt.close()


def plot_heatmap_by_embedding_and_chunk(df, output_dir, timestamp):
    """Plot heatmaps of metrics by embedding model and chunk size."""
    metric_cols = [col for col in df.columns if col.endswith('_score')]

    for llm in df['llm_model'].unique():
        llm_df = df[df['llm_model'] == llm]

        fig, axes = plt.subplots(nrows=len(metric_cols), figsize=(12, 4 * len(metric_cols)))

        for i, metric in enumerate(metric_cols):
            pivot = pd.pivot_table(
                llm_df,
                values=metric,
                index='embedding_model',
                columns='chunk_size',
                aggfunc='mean'
            )

            ax = axes[i] if len(metric_cols) > 1 else axes
            sns.heatmap(pivot, annot=True, cmap='viridis', vmin=0, vmax=1, ax=ax)

            metric_name = metric.replace('_score', '')
            ax.set_title(f'{metric_name} by Embedding Model and Chunk Size (LLM: {llm})')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_{llm}_{timestamp}.png'), dpi=300)
        plt.close()


def plot_metric_distributions(df, output_dir, timestamp):
    """Plot box plots of metric score distributions."""
    metric_cols = [col for col in df.columns if col.endswith('_score')]

    plt.figure(figsize=(12, 8))

    melted = pd.melt(df, id_vars=['llm_model', 'embedding_model', 'chunk_size'],
                     value_vars=metric_cols, var_name='Metric', value_name='Score')
    melted['Metric'] = melted['Metric'].str.replace('_score', '')

    sns.boxplot(x='Metric', y='Score', hue='llm_model', data=melted)
    plt.title('Distribution of Metric Scores by LLM Model')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.legend(title='LLM Model')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'metric_distributions_{timestamp}.png'), dpi=300)
    plt.close()


def generate_summary_tables(df, output_dir, timestamp):
    """Generate summary tables of the results."""
    # 1. Overall average metrics by configuration
    overall_avg = df.groupby(['llm_model', 'embedding_model', 'chunk_size'])[
        [col for col in df.columns if col.endswith('_score')]
    ].mean().reset_index()

    # Add an "average_score" column that averages all metrics
    score_cols = [col for col in overall_avg.columns if col.endswith('_score')]
    overall_avg['average_score'] = overall_avg[score_cols].mean(axis=1)

    # Sort by average score
    overall_avg = overall_avg.sort_values('average_score', ascending=False)

    # Save to CSV
    overall_avg.to_csv(os.path.join(output_dir, f'overall_avg_metrics_{timestamp}.csv'), index=False)

    # 2. Best configuration for each metric
    best_configs = pd.DataFrame()

    for metric in [col for col in df.columns if col.endswith('_score')]:
        metric_name = metric.replace('_score', '')
        best_config = df.loc[df[metric].idxmax()]

        best_configs = pd.concat([
            best_configs,
            pd.DataFrame({
                'Metric': [metric_name],
                'Best Score': [best_config[metric]],
                'LLM Model': [best_config['llm_model']],
                'Embedding Model': [best_config['embedding_model']],
                'Chunk Size': [best_config['chunk_size']]
            })
        ])

    best_configs.to_csv(os.path.join(output_dir, f'best_configs_{timestamp}.csv'), index=False)

    # 3. Generate HTML report with tables
    html_report = f"""
    <html>
    <head>
        <title>RAG Evaluation Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>RAG Evaluation Results Summary</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>Top 10 Configurations by Average Score</h2>
        {overall_avg.head(10).to_html(index=False)}

        <h2>Best Configuration for Each Metric</h2>
        {best_configs.to_html(index=False)}

        <h2>Average Scores by LLM Model</h2>
        {df.groupby('llm_model')[[col for col in df.columns if col.endswith('_score')]].mean().to_html()}

        <h2>Average Scores by Embedding Model</h2>
        {df.groupby('embedding_model')[[col for col in df.columns if col.endswith('_score')]].mean().to_html()}

        <h2>Average Scores by Chunk Size</h2>
        {df.groupby('chunk_size')[[col for col in df.columns if col.endswith('_score')]].mean().to_html()}
    </body>
    </html>
    """

    with open(os.path.join(output_dir, f'summary_report_{timestamp}.html'), 'w') as f:
        f.write(html_report)


# Main function
def main(eval_dir, output_dir):
    """Load evaluation results, generate visualizations, and create reports."""
    # Load evaluation results
    results = load_evaluation_results(eval_dir)

    if not results:
        print("No results found to visualize")
        return

    # Convert to DataFrame
    df = results_to_dataframe(results)

    if df is None or df.empty:
        print("Failed to convert results to DataFrame")
        return

    # Generate visualizations
    generate_visualizations(df, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RAG evaluation results")

    parser.add_argument('--eval-dir', type=str, default='../data/evaluation_results',
                        help='Directory containing evaluation results')

    parser.add_argument('--output-dir', type=str, default='../data/visualizations',
                        help='Directory to store visualizations')

    args = parser.parse_args()

    main(args.eval_dir, args.output_dir)