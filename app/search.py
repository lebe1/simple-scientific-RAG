from elasticsearch import Elasticsearch
from pprint import pprint
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder
import gc
import os
from pathlib import Path

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

        # Load the Sentence-BERT model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Read the legal basis text
        with open('../data/legal-basis.txt', 'r', encoding='utf-8') as file:
            text = file.read()

        # Chunk the text
        chunks = self.chunk_text(text)

        # Generate embeddings for each chunk
        embeddings = self.create_embeddings(chunks)

        # Index the chunks into Elasticsearch
        self.index_chunks(chunks, embeddings)


    def chunk_text(self, text, max_size_kb=128):
        """Chunk the input text into smaller parts of max_size_kb."""
        # Load the spaCy model
        nlp = spacy.load("en_core_web_sm")
        max_size_bytes = max_size_kb * 1024  # Convert KB to Bytes
        chunks = []
        current_chunk = []

        for sentence in nlp(text).sents:
            current_chunk.append(sentence.text)
            current_chunk_size = sum(len(s.encode('utf-8')) for s in current_chunk)

            if current_chunk_size >= max_size_bytes:
                # Append all but the last sentence
                chunks.append(" ".join(current_chunk[:-1]))  
                # Start a new chunk with the last sentence
                current_chunk = [current_chunk[-1]]  

            # Add any remaining sentences as a final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks        

    def create_embeddings(self, chunks):
        """Generate embeddings for each chunk using Sentence-BERT.""" 

        # Catch edge case if there are no chunks
        if not chunks:
            return []  

        embeddings = self.model.encode(chunks, show_progress_bar=True)

        return embeddings   
    
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
        query_embedding = self.model.encode(query).tolist()

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
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Create (query, chunk) pairs for Cross-Encoder scoring
        query_chunk_pairs = [[query, chunk] for chunk in retrieved_chunks]

        # Score the pairs using the Cross-Encoder
        scores = cross_encoder.predict(query_chunk_pairs)

        # Combine chunks with their scores and sort by relevance
        chunk_score_pairs = list(zip(retrieved_chunks, scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by score in descending order

        # Return the top-ranked chunk and its score
        return chunk_score_pairs[0]  # Return the best chunk
    
    def __del__(self):
        """Close the Elasticsearch connection."""
        self.es.close()
        print('Connection to Elasticsearch closed!')