from elasticsearch import Elasticsearch
from pprint import pprint
from sentence_transformers import SentenceTransformer, CrossEncoder
import gc
import os
from pathlib import Path
import torch
from embedding import Embedding

def load_env_file(filepath):
    with open(filepath) as f:
        for line in f:
            # Ignore comments and empty lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Parse the key-value pair
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

class Search:
    def __init__(self):
        self.embedding = Embedding(model_name='jinaai/jina-embeddings-v2-base-de')
        # Define the path to the .env file
        env_path = Path(__file__).resolve().parent.parent / '.env'

        # Load the .env file
        load_env_file(env_path)

        # Load password from environment variables
        password = os.getenv('ES_LOCAL_PASSWORD')

        # Initialize Elasticsearch client
        self.es = Elasticsearch('http://localhost:9200', http_auth=('elastic', password))  
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)
    
    def index_chunks(self, chunks, embeddings):
        """Index the chunks into Elasticsearch with their embeddings."""
        for i, chunk in enumerate(chunks):
            doc = {
                'text': chunk,
                'embedding': embeddings[i].tolist()  # Convert to list for JSON serialization
            }
            self.es.index(index='documents', id=i, document=doc)

    def search(self, query, top_k=30):
        """Search for relevant chunks in Elasticsearch."""
        # Generate embedding for the query
        query_embedding = self.embedding.model.encode(query).tolist()

        # Perform a vector search on Elasticsearch
        response = self.es.search(index='documents', body={
            'query': {
                'script_score': {
                    'query': {
                        'match_all': {}
                    },
                    'script': {
                        'source': "cosineSimilarity(params.queryVector, 'embedding') + 1.0",  # +1.0 to make it positive
                        'params': {
                            'queryVector': query_embedding
                        }
                    }
                }
            },
            'size': top_k
        })

        # Extract and return the relevant chunks
        retrieved_chunks = [hit['_source']['text'] for hit in response['hits']['hits']]

        best_chunk, score = self.rank_chunks_with_cross_encoder(query, retrieved_chunks)

        # Save memory by clearing the embeddings
        del response
        gc.collect()

        # Save best chunk into text file
        with open('../data/best_chunk.txt', 'w', encoding='utf-8') as file:
            file.write(best_chunk)

        return best_chunk
    
    def rank_chunks_with_cross_encoder(self, query, retrieved_chunks):
        """Rank the retrieved chunks based on their relevance to the query using SBERT Cross-Encoder."""
        # Load the Cross-Encoder model for scoring
        cross_encoder = CrossEncoder('cross-encoder/msmarco-MiniLM-L12-en-de-v1', max_length=512)

        # Create (query, chunk) pairs for Cross-Encoder scoring
        query_chunk_pairs = [[query, chunk] for chunk in retrieved_chunks]

        # Score the pairs using the Cross-Encoder
        scores = cross_encoder.predict(query_chunk_pairs)

        # Combine chunks with their scores and sort by relevance
        chunk_score_pairs = list(zip(retrieved_chunks, scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by score in descending order

        chunks_subset = chunk_score_pairs[:10]
        # Merge best chunks
        best_chunks = " ".join(text for text, _ in chunks_subset)
        # Calculate the average score
        average_score = sum(score for _, score in chunks_subset) / len(chunks_subset)

        # Return the top-ranked chunk and its score
        return best_chunks, average_score  # Return the best chunk
    
    def __del__(self):
        """Close the Elasticsearch connection."""
        self.es.close()
        print('Connection to Elasticsearch closed!')