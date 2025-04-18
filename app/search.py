from elasticsearch import Elasticsearch
from pprint import pprint
from sentence_transformers import SentenceTransformer, CrossEncoder
import gc
import os
from pathlib import Path
from embedding import Embedding
import numpy as np

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
    def __init__(self, embedding):
        self.embedding = embedding
        # Define the path to the .env file
        env_path = Path(__file__).resolve().parent.parent / '.env'

        # Load the Cross-Encoder model for scoring
        self.cross_encoder = CrossEncoder('cross-encoder/msmarco-MiniLM-L12-en-de-v1', max_length=512)


        # Load the .env file
        load_env_file(env_path)

        # Load password from environment variables
        password = os.getenv('ES_LOCAL_PASSWORD')

        # Initialize Elasticsearch client
        self.es = Elasticsearch('http://localhost:9200', http_auth=('elastic', password))  
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

    def index_chunks(self, chunks, embeddings, index_name):
        """Index the chunks into Elasticsearch with their embeddings."""
        # Create index with mapping if it doesn't exist
        if not self.es.indices.exists(index=index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {"type": "dense_vector", "dims": len(embeddings[0])}
                    }
                }
            }
            self.es.indices.create(index=index_name, body=mapping)
            print(f"Created new index: {index_name}")

        skipped_chunks = 0
        indexed_chunks = 0

        # Index the documents
        for i, chunk in enumerate(chunks):
            # Convert embedding to list
            embedding_list = embeddings[i].tolist()

            # Check if embedding is valid (not all zeros and no NaN values)
            if np.any(embeddings[i]) and not np.any(np.isnan(embeddings[i])):
                doc = {
                    'text': chunk,
                    'embedding': embedding_list
                }
                try:
                    self.es.index(index=index_name, id=indexed_chunks, document=doc)
                    indexed_chunks += 1
                    if indexed_chunks % 100 == 0:  # Print progress every 100 documents
                        print(f"Indexed {indexed_chunks}/{len(chunks)} valid chunks")
                except Exception as e:
                    print(f"Error indexing chunk {i}: {str(e)}")
                    skipped_chunks += 1
            else:
                print(f"Skipping chunk {i} due to invalid embedding (zero magnitude or NaN values)")
                skipped_chunks += 1

    def search(self, query, top_k=30):
        """Search for relevant chunks in Elasticsearch."""
        # Generate embedding for the query
        print(f"Generating embedding for the query...")
        query_embedding = self.embedding.create_embeddings([query], batch_size=1)[0].tolist()

        print(f"Embedding creation finished, performing vector search...")
        # Perform a vector search on Elasticsearch
        response = self.es.search(index=self.embedding.index_name, body={
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

        print(f"Reranking relevant chunks...")
        best_chunk, _ = self.rank_chunks_with_cross_encoder(query, retrieved_chunks)

        # Save memory by clearing the embeddings
        del response
        gc.collect()

        # Save best chunk into text file
        with open('../data/best_chunk.txt', 'w', encoding='utf-8') as file:
            file.write(best_chunk)

        return best_chunk
    
    def rank_chunks_with_cross_encoder(self, query, retrieved_chunks):
        """Rank the retrieved chunks based on their relevance to the query using SBERT Cross-Encoder."""

        # Create (query, chunk) pairs for Cross-Encoder scoring
        query_chunk_pairs = [[query, chunk] for chunk in retrieved_chunks]

        # Score the pairs using the Cross-Encoder
        scores = self.cross_encoder.predict(query_chunk_pairs)

        # Combine chunks with their scores and sort by relevance
        chunk_score_pairs = list(zip(retrieved_chunks, scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by score in descending order

        # Important parameter here: Set the top k chunks here considered for retrieval
        chunks_subset = chunk_score_pairs[:3]
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