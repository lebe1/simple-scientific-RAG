import json
import csv
import os
import argparse
from datetime import datetime
import glob


def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def load_benchmark_results(file_path):
    """Load benchmark results from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Successfully loaded {len(results)} results from {file_path}")
        return results
    except Exception as e:
        print(f"Error loading benchmark results from {file_path}: {str(e)}")
        return None


def extract_config_from_filename(filename):
    """Extract configuration details from the benchmark result filename."""
    try:
        # results_gemma3:12b_jina-embeddings-v2-base-de_4kb_de_core_news_lg_20250413_083826.json
        parts = os.path.basename(filename).split('_')
        if len(parts) >= 5:
            return {
                "llm_model": parts[1],
                "embedding_model": parts[2],
                "chunk_size": parts[3].replace('kb', ''),
                "spacy_model": parts[4]
            }
    except Exception as e:
        print(f"Error extracting config from filename: {str(e)}")

    return None


def create_evaluation_csv(results, output_path, config=None):
    """Create CSV file for manual evaluation from benchmark results."""
    if not results:
        print("No results to process")
        return

    # Extract configuration from first result if not provided
    if not config and 'config' in results[0]:
        config = results[0]['config']

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'Question ID',
            'Question',
            'RAG Answer',
            'Reference Answer',
            'LLM Model',
            'Embedding Model',
            'Chunk Size',
            'Context Quality (1-5)',
            'Answer Relevance (1-5)',
            'Answer Correctness (1-5)',
            'Faithfulness to Context (1-5)',
            'Notes'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, result in enumerate(results):
            # Get configuration details
            result_config = result.get('config', config)

            writer.writerow({
                'Question ID': result.get('question_id', i + 1),
                'Question': result.get('question', 'No question'),
                'RAG Answer': result.get('answer', 'No answer'),
                'Reference Answer': result.get('reference', ''),
                'LLM Model': result_config.get('llm_model', ''),
                'Embedding Model': result_config.get('embedding_model', ''),
                'Chunk Size': result_config.get('chunk_size', ''),
                'Context Quality (1-5)': '',
                'Answer Relevance (1-5)': '',
                'Answer Correctness (1-5)': '',
                'Faithfulness to Context (1-5)': '',
                'Notes': ''
            })

    print(f"Created evaluation CSV at {output_path}")
    return output_path


def process_benchmark_files(input_dir, output_dir, pattern="results_*.json"):
    """Process all benchmark result files in the input directory."""
    ensure_directory(output_dir)

    # Find all benchmark files matching the pattern
    file_paths = glob.glob(os.path.join(input_dir, pattern))

    if not file_paths:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return []

    processed_files = []

    for file_path in file_paths:
        # Load benchmark results
        results = load_benchmark_results(file_path)
        if not results:
            continue

        # Extract config from filename
        config = extract_config_from_filename(file_path)

        # Create output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        basename = os.path.basename(file_path).replace('.json', '')
        output_path = os.path.join(output_dir, f"eval_{basename}_{timestamp}.csv")

        # Create evaluation CSV
        csv_path = create_evaluation_csv(results, output_path, config)
        if csv_path:
            processed_files.append(csv_path)

    return processed_files


def create_evaluation_guide(output_dir):
    """Create evaluation guide file in the output directory."""
    guide_path = os.path.join(output_dir, "evaluation_guide.txt")

    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("""
RAG EVALUATION CRITERIA GUIDE
=============================

When evaluating the RAG outputs in the CSV file, use these criteria:

1. CONTEXT QUALITY (1-5)
   1 - Completely irrelevant context
   2 - Mostly irrelevant with some useful information
   3 - Contains partial relevant information
   4 - Contains most relevant information
   5 - Perfect context retrieval

2. ANSWER RELEVANCE (1-5)
   1 - Answer doesn't address the question at all
   2 - Answer is tangentially related to the question
   3 - Answer partially addresses the question
   4 - Answer addresses the question but misses minor points
   5 - Answer perfectly addresses the question

3. ANSWER CORRECTNESS (1-5)
   1 - Answer is completely incorrect
   2 - Answer contains mostly incorrect information
   3 - Answer contains a mix of correct and incorrect information
   4 - Answer is mostly correct with minor errors
   5 - Answer is completely correct

4. FAITHFULNESS TO CONTEXT (1-5)
   1 - Answer contradicts or fabricates beyond the context
   2 - Answer mostly contains information not in the context
   3 - Answer uses some information from context but adds unsupported details
   4 - Answer mostly sticks to context with minimal extrapolation
   5 - Answer completely faithful to the provided context

NOTES:
- Use this column for qualitative observations
- Note any specific issues, patterns or ideas for improvement
- For legal texts, pay special attention to accuracy of legal references
        """)

    print(f"Created evaluation guide at {guide_path}")
    return guide_path


def combine_benchmark_files(input_dir, output_dir, pattern="results_*.json"):
    """Combine results from multiple benchmark files into a single CSV for comprehensive evaluation."""
    ensure_directory(output_dir)

    # Find all benchmark files matching the pattern
    file_paths = glob.glob(os.path.join(input_dir, pattern))

    if not file_paths:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return None

    combined_results = []

    for file_path in file_paths:
        # Load benchmark results
        results = load_benchmark_results(file_path)
        if not results:
            continue

        # Extract config from filename
        file_config = extract_config_from_filename(file_path)

        # Update each result with the config from filename if needed
        for result in results:
            if 'config' not in result and file_config:
                result['config'] = file_config
            combined_results.append(result)

    if not combined_results:
        print("No results to combine")
        return None

    # Create output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f"combined_evaluation_{timestamp}.csv")

    # Create combined evaluation CSV
    return create_evaluation_csv(combined_results, output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate manual evaluation files from benchmark results")

    parser.add_argument('--input', type=str, default='../data/benchmark_results',
                        help='Input directory containing benchmark result files')

    parser.add_argument('--output', type=str, default='../data/evaluation',
                        help='Output directory for evaluation files')

    parser.add_argument('--mode', choices=['individual', 'combine', 'single'],
                        default='individual',
                        help='Processing mode: individual (process each benchmark file separately), combine (combine all files into one CSV), or single (process a single file)')

    parser.add_argument('--file', type=str, default=None,
                        help='Path to a specific benchmark file (for single mode)')

    parser.add_argument('--pattern', type=str, default='results_*.json',
                        help='Filename pattern to match benchmark files')

    args = parser.parse_args()

    # Create output directory
    ensure_directory(args.output)

    # Create evaluation guide
    create_evaluation_guide(args.output)

    # Process files based on selected mode
    if args.mode == 'individual':
        processed_files = process_benchmark_files(args.input, args.output, args.pattern)

        if processed_files:
            print(f"\nSuccessfully processed {len(processed_files)} benchmark files:")
            for file_path in processed_files:
                print(f"  - {file_path}")
        else:
            print("\nNo benchmark files were processed.")

    elif args.mode == 'combine':
        combined_file = combine_benchmark_files(args.input, args.output, args.pattern)

        if combined_file:
            print(f"\nSuccessfully combined benchmark results into: {combined_file}")
        else:
            print("\nFailed to combine benchmark results.")

    elif args.mode == 'single':
        if not args.file:
            print("Error: --file parameter is required for single mode")
            return

        # Load benchmark results
        results = load_benchmark_results(args.file)
        if not results:
            return

        # Extract config from filename
        config = extract_config_from_filename(args.file)

        # Create output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        basename = os.path.basename(args.file).replace('.json', '')
        output_path = os.path.join(args.output, f"eval_{basename}_{timestamp}.csv")

        # Create evaluation CSV
        csv_path = create_evaluation_csv(results, output_path, config)
        if csv_path:
            print(f"\nSuccessfully processed benchmark file into: {csv_path}")

    print("\nNext steps:")
    print("1. Open the generated CSV file(s) in a spreadsheet application")
    print("2. Fill in the evaluation scores (1-5) for each answer")
    print("3. Save the filled CSV file")
    print("4. Run the analyze_eval_results.py script on the filled CSV")


if __name__ == "__main__":
    main()