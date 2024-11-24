import os
import torch
from torch.cuda.amp import autocast
import pickle
import numpy as np
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from processor import Processor

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:256'

class Embedding:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                               torch_dtype=torch.bfloat16)

    def main(self):

        # Chunk the text
        chunks = Processor().chunk_text(self.text, spacy_model='de_core_news_lg')

        # Generate embeddings for each chunk
        embeddings = self.create_embeddings(chunks)

        # Store chunks and embeddings
        self.save_chunks(chunks)
        self.save_embeddings(embeddings)


    def create_embeddings(self, chunks, batch_size=8):
        """Generate embeddings for each chunk using Sentence-BERT."""

        # Catch edge case if there are no chunks
        if not chunks:
            return []

        embeddings = []
        with torch.no_grad():  # Disable gradient calculations
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                with autocast():  # Enable mixed precision
                    batch_embeddings = self.model.encode(batch_chunks)  # Generate embeddings for the batch
                embeddings.extend(batch_embeddings)  # Append batch embeddings to the main list

                # Clear CUDA cache after each batch to free memory
                torch.cuda.empty_cache()

        return embeddings

    @staticmethod
    def save(embeddings):
        np.save(os.path.join(FILE_PATH, '../data/embeddings.npy'), embeddings)

    @staticmethod
    def load(path= os.path.join(FILE_PATH, '../data/embeddings.npy')):
        embeddings = np.load(path)
        return embeddings

if __name__ == "__main__":
    # model_name='jinaai/jina-embeddings-v2-small-en'
    model_name = 'jinaai/jina-embeddings-v2-base-de'
    embedding = Embedding(model_name=model_name)
    embedding.main()


