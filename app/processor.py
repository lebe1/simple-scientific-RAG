import spacy
import os
from pathlib import Path
import pickle

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

class Processor:
    def __init__(self, spacy_model='de_core_news_lg', chunk_size_in_kb=4):
        self.spacy_model = spacy_model
        self.chunk_size_in_kb = chunk_size_in_kb

    def chunk_text(self, text):
        """Chunk the input text into smaller parts of self.chunk_size_in_kb."""
        # Load the spaCy model
        nlp = spacy.load(self.spacy_model)
        max_size_bytes = self.chunk_size_in_kb * 1024  # Convert KB to Bytes
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

        self.save_chunks_to_output_dir(chunks)

        return chunks

    def save(self, chunks):
        file_name = f'../data/{self.spacy_model}_{self.chunk_size_in_kb}kb_chunks.pkl'
        path = os.path.join(FILE_PATH, file_name)
        with open(path, 'wb') as f:
            pickle.dump(chunks, f)

    def load(self):
        file_name = f'../data/{self.spacy_model}_{self.chunk_size_in_kb}kb_chunks.pkl'
        path = os.path.join(FILE_PATH, file_name)
        with open(path, 'rb') as f:
            chunks = pickle.load(f)
        return chunks

    def save_chunks_to_output_dir(self, chunks):
        file_name = f"../data/{self.spacy_model}_{self.chunk_size_in_kb}kb_chunks.txt"
        path = os.path.join(FILE_PATH, file_name)
        with open(path, "w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(chunk + "\n\n")
        print(f"Chunks saved to {path}")
