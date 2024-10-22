from fastapi import FastAPI
from pydantic import BaseModel
from model import chat
from search import Search

# Create FastAPI instance
app = FastAPI()
es = Search()

# Define requests body model
class PromptRequest(BaseModel):
    question: str

class SearchQuery(BaseModel):
    query: str

# POST route for prompt
@app.post("/api/prompt")
async def handle_prompt(request: PromptRequest):
    answer = chat(request.question)
    return {"answer": f"{answer}"}

# POST route for search
@app.post("/api/search")
async def handle_search(search: SearchQuery):
    results = es.search(search.query)
    return {"results": results}
