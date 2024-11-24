import os
import argparse
from processor import Processor
from embedding import Embedding
from search import Search
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

class Workflow:
    def __init__(self, model_name='jinaai/jina-embeddings-v2-base-de', spacy_model='de_core_news_lg'):
        self.processor = Processor(spacy_model=spacy_model)
        self.embedding = Embedding(model_name=model_name)
        self.es = Search()

    def create_new_embeddings(self):
        # Read the legal basis text
        legal_text_path = os.path.join(FILE_PATH, '../data/legal-basis.txt')
        with open(legal_text_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Chunk the text
        chunks = self.processor.chunk_text(text=text, max_size_kb=4)
        self.processor.save(chunks)

        # Generate embeddings for each chunk
        embeddings = self.embedding.create_embeddings(chunks)
        self.embedding.save(embeddings)

    def update_es_index(self):
        chunks = self.processor.load()
        embeddings = self.embedding.load()
        self.es.index_chunks(chunks, embeddings)

        return chunks, embeddings

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Choose workflow operation")
    parser.add_argument('operation', choices=['create-embeddings', 'update-es-index'], help="Operation to perform: 'create' or 'update'")

    args = parser.parse_args()

    # English
    # model_name='jinaai/jina-embeddings-v2-small-en'
    # spacy_model = 'en_core_web_sm'

    # German
    model_name = 'jinaai/jina-embeddings-v2-base-de'
    spacy_model = 'de_core_news_lg'
    workflow = Workflow(model_name=model_name, spacy_model=spacy_model)

    # Run the appropriate method based on user input
    if args.operation == 'create-embeddings':
        workflow.create_new_embeddings()
    elif args.operation == 'update-es-index':
        workflow.update_es_index()