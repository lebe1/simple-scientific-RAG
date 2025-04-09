import os
import argparse
from processor import Processor
from embedding import Embedding
from search import Search
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = '../data/legal-basis.txt'

class Workflow:
    def __init__(self, model='jinaai/jina-embeddings-v2-base-de', spacy_model='de_core_news_lg', chunk_size_in_kb=4, index_name="documents"):
        self.processor = Processor(spacy_model=spacy_model, chunk_size_in_kb=chunk_size_in_kb)
        self.embedding = Embedding(spacy_model=spacy_model, chunk_size_in_kb=chunk_size_in_kb, model=model)
        self.es = Search(embedding=self.embedding)

    def create_new_embeddings(self, split_by_article=False, split_by_subarticle=False):
        # Read the legal basis text
        legal_text_path = os.path.join(FILE_PATH, DATA_PATH)
        with open(legal_text_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Chunk the text
        if split_by_article:
            chunks = self.processor.chunk_by_article(text=text)
        elif split_by_subarticle:
            chunks = self.processor.chunk_by_article(text=text, split_into_subarticles=True)
        else:
            chunks = self.processor.chunk_text(text=text)
        self.processor.save(chunks)

        # Generate embeddings for each chunk
        embeddings = self.embedding.create_embeddings(chunks)
        self.embedding.save(embeddings)

    def update_es_index(self):
        chunks = self.processor.load()
        embeddings = self.embedding.load()
        self.es.index_chunks(chunks, embeddings, self.embedding.index_name)

        return chunks, embeddings


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Choose workflow operation")

    # Required argument for operation
    parser.add_argument(
        'operation',
        choices=['create-embeddings', 'update-es-index'],
        help="Operation to perform: 'create-embeddings' or 'update-es-index'"
    )

    # Optional arguments with defaults
    parser.add_argument(
        '--model',
        type=str,
        default='jinaai/jina-embeddings-v2-base-de',
        help="Name of the embedding model (default: jinaai/jina-embeddings-v2-base-de)"
    )

    parser.add_argument(
        '--spacy-model',
        type=str,
        default='de_core_news_lg',
        help="Name of the spaCy model (default: de_core_news_lg)"
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=4,
        help="Chunk size in KB (default: 4)"
    )

    args = parser.parse_args()

    # Initialize workflow with provided or default arguments
    workflow = Workflow(
        model=args.model,
        spacy_model=args.spacy_model,
        chunk_size_in_kb=args.chunk_size
    )

    # Run the appropriate method based on user input
    if args.operation == 'create-embeddings':
        workflow.create_new_embeddings()
    elif args.operation == 'update-es-index':
        workflow.update_es_index()
    elif args.operation == 'create-embeddings-by-article':
        workflow.create_new_embeddings(split_by_article=True, split_by_subarticle=True)
    elif args.operation == 'create-embeddings-by-subarticle':
        workflow.create_new_embeddings(split_by_article=True, split_by_subarticle=True)
