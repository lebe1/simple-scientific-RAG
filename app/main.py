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
    model: str
    spacy_model: str
    chunk_size_in_kb: int
    llm_model: str = "llama3-chatqa:8b"  # Default LLM model

class SearchQuery(BaseModel):
    query: str
    model: str
    spacy_model: str
    chunk_size_in_kb: int

# POST route for prompt
@app.post("/api/prompt")
async def handle_prompt(request: PromptRequest):
    answer = model.chat(request.question, model=request.llm_model)
    return {"answer": f"{answer}"}

# POST route for search
@app.post("/api/search")
async def handle_search(search: SearchQuery):
    embedding = Embedding(spacy_model=search.spacy_model, chunk_size_in_kb=search.chunk_size_in_kb, model=search.model)
    es = Search(embedding=embedding)
    results = es.search(search.query)
    return {"results": results}

# POST route for RAG prompt
@app.post("/api/rag")
async def handle_rag(request: PromptRequest):
    embedding = Embedding(spacy_model=request.spacy_model, chunk_size_in_kb=request.chunk_size_in_kb, model=request.model)
    es = Search(embedding=embedding)
    rag_output = model.rag(question=request.question, es=es, model=request.llm_model)
    return {"context": f"{rag_output[1]}", "answer": f"{rag_output[0]}"}