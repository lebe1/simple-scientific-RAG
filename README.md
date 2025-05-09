# Simple scientific RAG

## Requirements

Make sure you have Git, Python 3.12+ and Docker installed on your machine.

## Setup Instructions

1. **Clone the repository**

    ```bash
    git clone git@github.com:lebe1/simple-scientific-RAG.git
    cd simple-scientific-RAG
    ```

2. **Create a virtual environment** (optional but recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**

    Before running the project, install all the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

4. **Build Dockerfile**

    Copy the example environment file into the `env`

    ```bash
    cp .env.example .env
    ```

    ```bash
    docker compose build
    ```

5. **Run docker container**

   Run docker-compose.yml to pull the required image:

   ```bash
   docker compose --profile cpu up -d
   ```
   If you have GPU available, run it with the GPU profile enabled:

   ```bash
   docker compose --profile gpu up -d
   ```

6. **Install Ollama model**

   Pull the required model by running, choose either `ollama-cpu` or `ollama-gpu`:

   ```bash
   docker exec ollama-cpu ollama run llama3-chatqa:8b
   docker exec ollama-cpu ollama run gemma3:27b
   ```

7. **Install model for chunking**

    ```bash
    python -m spacy download de_core_news_lg
    ```

8. **Creating the embedding vectors and their index**

    You have several arguments to pass here. You can decide between three splitting criteria.
    1. `create-embeddings` splits the text until the maximum chunk size is reached
    2. `create-embeddings-by-article` splits the text at each article
    3. `create-embeddings-by-subarticle` splits the text at each subarticle
    4. `--chunk-size` defines the chunk size
    5. `--model` defines the embedding model
    
    In the following several sample arguments are provided 
    ```bash
    python app/workflow.py create-embeddings --chunk-size 0.5 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    python app/workflow.py create-embeddings-by-article --chunk-size 4.0 --model jinaai/jina-embeddings-v2-base-de
    python app/workflow.py create-embeddings-by-subarticle --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    ```

    For the experiments, you can also create several indeces with one of the following Scripts

    ```bash
    ./run_workflow_create_index_small
    ./run_workflow_create_index
    ```

    

## Running the application after setup instructions

**Note:** If you want to improve your runtime and you have access to a GPU, comment out the commented lines of code in the `docker-compose.yml`

Run fastapi server locally:

```bash
cd app;
uvicorn main:app --reload
```

### Testing the API

There are two ways for testing the API.  
Either by sending the following POST-request using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":4}'
```
```bash
curl -X POST "http://127.0.0.1:8000/api/search" -H "Content-Type: application/json" -d '{"query": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":4}'
```

Or by opening the user interface via `http://127.0.0.1:8000`

## Running the automated benchmark execution

Assuming again that you executed the `uvicorn` command above, execute:

```bash
cd app/evaluation;
python benchmark.py --questions ../../data/sample_questions.txt --references ../../data/sample_answers.txt --output-dir ../../data/benchmark_results_final
./cleanup_empty_json.sh -d data/benchmark_results_final/
```

Which will execute the following combinations of multiple llm-model, chunk-size, and model embedding:
```python
CONFIGURATIONS = {
    "llm_models": ["llama3-chatqa:8b", "gemma3:27b"],  # Add other models you have in Ollama
    "embedding_models": [
        "jinaai/jina-embeddings-v2-base-de",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Add other embedding models
    ],
    "chunk_sizes": [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128],  # Chunk sizes in KB
    "spacy_models": ["de_core_news_lg"]  # You could add more if needed
}
```

The `cleanup_empty_json.sh` script cleans empty JSON files that get created during the benchmark workflow.

## Running the quantitative evaluation

To run the quantitative evaluation using the LLM-as-a-judge approach, you can execute the evaluation script:

```bash
python evaluate_benchmarks.py --benchmark-dir ../data/benchmark_results --output-dir ../data/evaluation_results --eval-model gemma3:12b --max-retries 2
```

> Please note that you could use any LLM to evaluate, you just need to run it in the ollama container first. The `max-retries` was added to try multiple times in case the LLM does not provide a proper JSON structure.

Finally, you can run the visualization pipeline:

```bash
python visualize_results.py --eval-dir ../data/evaluation_results --output-dir ../data/visualizations
```

## Running the qualitative evaluation

To qualitatively evaluate the benchmark results, collect the results into a single CSV file:

```bash
python prepare_qualitative_eval.py --input ../data/benchmark_results_final/ --output ../data/evaluation_results_final/ --mode combine
```

Next, you need to give scores for each answer and combination of LLM, chunk size, embedding model, and spacy model. Either you manually fill it out, or you let other LLMs do it like ChatGPT or Claude. Finally, you execute the following visualization pipeline:

```bash
python visualize_qualitative_eval.py
```

Which will store the evaluation results under `data/evaluation_results_final`.

## Running the keyword evaluation

IMPORTANT: Make sure all docker container as well as the FastAPI App is running in a terminal via `uvicorn main:app --reload`.

Only the arguments `--llm-model` and `select-top-k` will affect the experiment.
The other two arguments are there to record the embedding chunking method that has been executed beforehand.

```bash
cd app/evaluation;
python question_query.py   --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2   --llm-model llama3.2   --select-top-k 3 5   --splitting-method SUBARTICLE
```

### Visualizing the results

Make sure the PATH variable is set correctly

```bash
python visualize_keyword_eval.py
```


## Data
The data of the legal basis can be found under https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=LrW&Gesetzesnummer=20000006