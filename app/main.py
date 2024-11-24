from fastapi import FastAPI
from pydantic import BaseModel
from model import Model
from search import Search

# Create FastAPI instance
app = FastAPI()
model = Model()

# Define requests body model
class PromptRequest(BaseModel):
    question: str

# POST route for prompt
@app.post("/api/prompt")
async def handle_prompt(request: PromptRequest):
    answer = model.chat(request.question)
    return {"answer": f"{answer}"}

# POST route for search
@app.post("/api/search")
async def handle_search(search: PromptRequest):
    es = Search()
    results = es.search(search.question)
    return {"results": results}

# POST route for RAG prompt
@app.post("/api/rag")
async def handle_rag(request: PromptRequest):
    answer = model.rag(request.question)
    return {"answer": f"{answer}"}
