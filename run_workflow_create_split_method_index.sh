# Default is --model jinaai/jina-embeddings-v2-base-de
python app/workflow.py create-embeddings --chunk-size 0.5 
python app/workflow.py create-embeddings-by-article 
python app/workflow.py create-embeddings-by-subarticle 

# Second model is --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py create-embeddings --chunk-size 0.5 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py create-embeddings-by-article --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py create-embeddings-by-subarticle --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2