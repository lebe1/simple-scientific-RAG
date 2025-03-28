from fastapi import FastAPI
from pydantic import BaseModel
from model import Model
from search import Search
from embedding import Embedding

# Create FastAPI instance
app = FastAPI()
model = Model()

# Define requests body model
class PromptRequest(BaseModel):
    question: str
    model_name: str
    spacy_model: str
    chunk_size_in_kb: int

class SearchQuery(BaseModel):
    query: str
    model_name: str
    spacy_model: str
    chunk_size_in_kb: int

# POST route for prompt
@app.post("/api/prompt")
async def handle_prompt(request: PromptRequest):
    answer = model.chat(request.question)
    return {"answer": f"{answer}"}

# POST route for search
@app.post("/api/search")
async def handle_search(search: SearchQuery):
    embedding = Embedding(spacy_model=search.spacy_model, chunk_size_in_kb=search.chunk_size_in_kb, model_name=search.model_name)
    es = Search(embedding=embedding)
    results = es.search(search.query)
    return {"results": results}

# POST route for RAG prompt
@app.post("/api/rag")
async def handle_rag(request: PromptRequest):
    embedding = Embedding(spacy_model=request.spacy_model, chunk_size_in_kb=request.chunk_size_in_kb, model_name=request.model_name)
    es = Search(embedding=embedding)
    answer = model.rag(question=request.question, es=es)
    return {"answer": f"{answer}"}
