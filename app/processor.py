import spacy
import os
from pathlib import Path
import pickle

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

class Processor:
    def __init__(self, spacy_model='de_core_news_lg'):
        self.spacy_model = spacy_model

    def chunk_text(self, text, max_size_kb=128):
        """Chunk the input text into smaller parts of max_size_kb."""
        # Load the spaCy model
        nlp = spacy.load(self.spacy_model)
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

        self.save_chunks_to_output_dir(chunks, self.spacy_model)

        return chunks

    @staticmethod
    def save(chunks):
        with open(os.path.join(FILE_PATH, '../data/chunks.pkl'), 'wb') as f:
            pickle.dump(chunks, f)

    @staticmethod
    def load(path=os.path.join(FILE_PATH, '../data/chunks.pkl')):
        with open(path, 'rb') as f:
            chunks = pickle.load(f)
        return chunks

    def save_chunks_to_output_dir(self, chunks):
        """Save the list of chunks to a file in the ../output directory."""
        output_dir = Path(os.path.join(FILE_PATH, '../output'))
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{self.spacy_model}_chunks.txt"
        with open(output_file, "w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(chunk + "\n\n")
        print(f"Chunks saved to {output_file}")
