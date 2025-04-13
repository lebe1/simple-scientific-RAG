import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:256'


class Embedding:
    def __init__(self, model, spacy_model, chunk_size_in_kb):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model and tokenizer
        self.model = AutoModel.from_pretrained(
            model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # Set model to evaluation mode
        self.model.eval()

        # Store configuration
        self.spacy_model = spacy_model
        self.chunk_size_in_kb = chunk_size_in_kb
        self.model_escaped = model.replace('/', '-').lower()
        self.file_name = f'../data/{self.model_escaped}__with{self.chunk_size_in_kb}kbchunks__spacymodel_{self.spacy_model}.npy'
        self.index_name = f"{self.model_escaped}__chunks{chunk_size_in_kb}kb__{spacy_model}"

    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]  # First element of model_output contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def create_embeddings(self, chunks, batch_size=8):
        """Generate embeddings for each chunk using the transformer model."""
        if not chunks:
            return []

        embeddings = []
        total_chunks = len(chunks)

        with torch.no_grad():
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]

                # Print progress
                print(f"Processing batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size}")

                # Tokenize the batch
                encoded_input = self.tokenizer(
                    batch_chunks,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)

                # Generate embeddings
                outputs = self.model(**encoded_input)

                # Mean pooling
                batch_embeddings = self.mean_pooling(
                    outputs,
                    encoded_input['attention_mask']
                )

                # Move to CPU and convert to numpy
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.extend(batch_embeddings)

                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"Created embeddings for {len(embeddings)} chunks")
        # Quick verification
        print(f"Sample embedding shape: {embeddings[0].shape}")
        print(f"Sample embedding non-zero values: {np.count_nonzero(embeddings[0])}")

        return np.array(embeddings)

    def save(self, embeddings):
        path = os.path.join(FILE_PATH, self.file_name)
        np.save(path, embeddings)
        print(f"Embeddings saved to: {path}")
        # Verify saved embeddings
        print(f"Saved embedding array shape: {embeddings.shape}")

    def load(self):
        path = os.path.join(FILE_PATH, self.file_name)
        embeddings = np.load(path)
        print(f"Loaded embedding array shape: {embeddings.shape}")
        return embeddings