from elasticsearch import Elasticsearch
from pprint import pprint
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder
import gc

class Search:
    def __init__(self):
        # Initialize Elasticsearch client
        # TODO: Take API key from .env file
        self.es = Elasticsearch('http://localhost:9200', api_key="c3VDSHFwSUJLNC1XWnBHYUUtaEo6RnZ6RUVYY19SODZHY1pCeTFJRzNwQQ==")  
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

        # Load the Sentence-BERT model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Read the legal basis text
        with open('../data/legal-basis.txt', 'r', encoding='utf-8') as file:
            text = file.read()

        print(f"Read {len(text)} characters from the legal basis text.")
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
        retrieved_chunks = [hit['_source'] for hit in response['hits']['hits']]

        print(f"Found {len(retrieved_chunks)} relevant chunks!")

        best_chunk, score = self.rank_chunks_with_cross_encoder(query, retrieved_chunks)

        print(f"Best chunk score: {score:.2f}")

        return best_chunk['text']
    
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