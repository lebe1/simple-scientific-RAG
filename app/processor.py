import spacy
import os
from pathlib import Path
import pickle
import re

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

class Processor:
    def __init__(self, spacy_model='de_core_news_lg', chunk_size_in_kb=4.0):
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
    
    def chunk_by_article(self, text, split_into_subarticles=False):
        # Split the text into chunks starting with "ARTIKEL"
        articles = []
        current_article = []
        inside_paragraph = False
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith("ARTIKEL") or re.match(r"§ \d+(?!\.)", line): # Matches § 1, § 2 etc.
                if current_article:  # Save the previous article if it exists
                    articles.append('\n'.join(current_article))
                    current_article = []
                current_article.append(line)  
            elif split_into_subarticles and re.match(r"\(\d+[a-z]?\)", line):  # Matches (1), (2a), (5b), etc.
                if current_article:  
                    articles.append('\n'.join(current_article))
                    current_article = []
                current_article.append(line) 
            elif line == "Text" or re.match(r"§ \d+.", line) or not line: # Matches § 1., § 22. etc.
                continue  
            # Only add lines if we're inside an article
            elif current_article:  
                current_article.append(line)
        
        # Important: Add last article before returning
        if current_article:  
            articles.append('\n'.join(current_article))

        self.save_chunks_to_output_dir(articles)

        return articles

    def save(self, chunks):
        file_name = f'../data/chunks/{self.spacy_model}_{self.chunk_size_in_kb}kb_chunks.pkl'
        path = os.path.join(FILE_PATH, file_name)
        with open(path, 'wb') as f:
            pickle.dump(chunks, f)

    def load(self):
        file_name = f'../data/chunks/{self.spacy_model}_{self.chunk_size_in_kb}kb_chunks.pkl'
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
