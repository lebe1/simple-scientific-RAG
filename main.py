from fastapi import FastAPI
from pydantic import BaseModel
from model import chat

# Create FastAPI instance
app = FastAPI()

# Define the request body model
class PromptRequest(BaseModel):
    question: str

# POST route for prompt
@app.post("/api/prompt")
async def handle_prompt(request: PromptRequest):
    answer = chat(request.question)
    return {"answer": f"{answer}"}

