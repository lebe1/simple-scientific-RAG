import os
import torch
from torch.cuda.amp import autocast
import pickle
import numpy as np
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from processor import Processor

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

class Embedding:
    def __init__(self):
        # Load the Sentence-BERT model
        self.model = SentenceTransformer('jinaai/jina-embeddings-v2-base-de')


        # Read the legal basis text
        legal_text_path = os.path.join(FILE_PATH, '../data/legal-basis.txt')
        with open(legal_text_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Chunk the text
        chunks = Processor().chunk_text(text, spacy_model='de_core_news_lg')

        # Generate embeddings for each chunk
        embeddings = self.create(chunks)

        # Store chunks and embeddings
        self.save_chunks(chunks)
        self.save_embeddings(embeddings)


    def create(self, chunks, batch_size=8):
        """Generate embeddings for each chunk using Sentence-BERT."""

        # Catch edge case if there are no chunks
        if not chunks:
            return []

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:256'

        # model_name = 'jinaai/jina-embeddings-v2-base-de'
        model_name='jinaai/jina-embeddings-v2-small-en'
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                          torch_dtype=torch.bfloat16)

        embeddings = []
        with torch.no_grad():  # Disable gradient calculations
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                with autocast():  # Enable mixed precision
                    batch_embeddings = model.encode(batch_chunks)  # Generate embeddings for the batch
                embeddings.extend(batch_embeddings)  # Append batch embeddings to the main list

                # Clear CUDA cache after each batch to free memory
                torch.cuda.empty_cache()

        return embeddings

    @staticmethod
    def save_chunks(chunks):
        with open(os.path.join(FILE_PATH, '../data/chunks.pkl'), 'wb') as f:
            pickle.dump(chunks, f)

    @staticmethod
    def save_embeddings(embeddings):
        np.save(os.path.join(FILE_PATH, '../data/embeddings.npy'), embeddings)

    @staticmethod
    def load_chunks(path= os.path.join(FILE_PATH, '../data/chunks.pkl')):
        with open(path, 'rb') as f:
            chunks = pickle.load(f)
        return chunks

    @staticmethod
    def load_embeddings(path= os.path.join(FILE_PATH, '../data/embeddings.npy')):
        embeddings = np.load(path)
        return embeddings

if __name__ == "__main__":
    Embedding()


