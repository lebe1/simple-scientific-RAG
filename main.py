from fastapi import FastAPI
from pydantic import BaseModel

# Create FastAPI instance
app = FastAPI()

# Define the request body model
class PromptRequest(BaseModel):
    prompt: str

# POST route for prompt
@app.post("/api/prompt")
async def handle_prompt(request: PromptRequest):
    return {"message": f"Prompt output: {request.prompt}"}

