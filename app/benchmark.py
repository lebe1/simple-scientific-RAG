import os
import json
import requests
from datetime import datetime
import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration options for benchmarking
CONFIGURATIONS = {
    "llm_models": ["gemma3:12b"],  # Add other models you have in Ollama
    "embedding_models": [
        "jinaai/jina-embeddings-v2-base-de",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Add other embedding models
    ],
    "chunk_sizes": [0.125],  # Chunk sizes in KB
    "spacy_models": ["de_core_news_lg"]  # You could add more if needed
}


# Function to read questions from a file
def read_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]


# Function to read reference answers
def read_references(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]


# Function to query the RAG API with a specific configuration
def query_rag(question, llm_model, embedding_model, spacy_model, chunk_size):
    url = "http://localhost:8000/api/rag"
    headers = {'Content-Type': 'application/json'}

    # Modify the API call to include the LLM model
    payload = {
        'question': question,
        'model': embedding_model,
        'spacy_model': spacy_model,
        'chunk_size_in_kb': chunk_size,
        'llm_model': llm_model  # You'll need to modify the API to accept this parameter
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Received status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None


# Function to evaluate a single configuration
def evaluate_configuration(config, questions, references, output_dir):
    llm_model, embedding_model, chunk_size, spacy_model = config

    # Create a unique identifier for this configuration
    config_id = f"{llm_model}_{embedding_model.split('/')[-1]}_{chunk_size}kb_{spacy_model}"
    print(f"Evaluating configuration: {config_id}")

    results = []

    # Process each question with the current configuration
    for i, (question, reference) in enumerate(zip(questions, references)):
        print(f"  Processing question {i + 1}/{len(questions)}: {question[:50]}...")

        # Query the RAG system
        rag_output = query_rag(
            question=question,
            llm_model=llm_model,
            embedding_model=embedding_model,
            spacy_model=spacy_model,
            chunk_size=chunk_size
        )

        if rag_output:
            # Store the results
            results.append({
                "question_id": i + 1,
                "question": question,
                "reference": reference,
                "answer": rag_output.get('answer', 'No answer provided'),
                "context": rag_output.get('context', 'No context provided'),
                "config": {
                    "llm_model": llm_model,
                    "embedding_model": embedding_model,
                    "chunk_size": chunk_size,
                    "spacy_model": spacy_model
                }
            })

    # Save the results for this configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"results_{config_id}_{timestamp}.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results for configuration {config_id} saved to {output_file}")
    return output_file


# Function to run all benchmarks
def run_benchmarks(questions_file, references_file, output_dir, configs=None, max_workers=2):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read questions and references
    questions = read_questions(questions_file)
    references = read_references(references_file)

    if len(questions) != len(references):
        print("Warning: Number of questions doesn't match number of references")

    # Generate all possible configurations or use the provided ones
    if configs is None:
        configs = list(itertools.product(
            CONFIGURATIONS["llm_models"],
            CONFIGURATIONS["embedding_models"],
            CONFIGURATIONS["chunk_sizes"],
            CONFIGURATIONS["spacy_models"]
        ))

    result_files = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_config = {
            executor.submit(
                evaluate_configuration, config, questions, references, output_dir
            ): config for config in configs
        }

        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result_file = future.result()
                result_files.append(result_file)
            except Exception as e:
                print(f"Configuration {config} generated an exception: {e}")

    # Generate a summary of all runs
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_configurations": len(configs),
        "total_questions": len(questions),
        "result_files": result_files
    }

    summary_file = os.path.join(output_dir, f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Benchmark summary saved to {summary_file}")
    return summary_file, result_files


# Main function with CLI arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG benchmarks across different configurations")

    parser.add_argument('--questions', type=str, default='../data/sample_questions.txt',
                        help='Path to the questions file')

    parser.add_argument('--references', type=str, default='../data/sample_answers.txt',
                        help='Path to the reference answers file')

    parser.add_argument('--output-dir', type=str, default='../data/benchmark_results',
                        help='Directory to store benchmark results')

    parser.add_argument('--max-workers', type=int, default=2,
                        help='Maximum number of parallel workers')

    parser.add_argument('--llm-models', type=str, nargs='+',
                        help='Specific LLM models to test')

    parser.add_argument('--embedding-models', type=str, nargs='+',
                        help='Specific embedding models to test')

    parser.add_argument('--chunk-sizes', type=int, nargs='+',
                        help='Specific chunk sizes to test')

    parser.add_argument('--spacy-models', type=str, nargs='+',
                        help='Specific spaCy models to test')

    args = parser.parse_args()

    # Filter configurations based on arguments
    selected_configs = None
    if any([args.llm_models, args.embedding_models, args.chunk_sizes, args.spacy_models]):
        llm_models = args.llm_models or CONFIGURATIONS["llm_models"]
        embedding_models = args.embedding_models or CONFIGURATIONS["embedding_models"]
        chunk_sizes = args.chunk_sizes or CONFIGURATIONS["chunk_sizes"]
        spacy_models = args.spacy_models or CONFIGURATIONS["spacy_models"]

        selected_configs = list(itertools.product(
            llm_models, embedding_models, chunk_sizes, spacy_models
        ))

    # Run the benchmarks
    run_benchmarks(
        questions_file=args.questions,
        references_file=args.references,
        output_dir=args.output_dir,
        configs=selected_configs,
        max_workers=args.max_workers
    )