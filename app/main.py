from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from model import Model
from search import Search
from embedding import Embedding

app = FastAPI()
model = Model()


templates = Jinja2Templates(directory="templates")

# Data models
class PromptRequest(BaseModel):
    question: str
    model: str
    spacy_model: str
    chunk_size_in_kb: float
    llm_model: str = "llama3.2"

class SearchQuery(BaseModel):
    query: str
    model: str
    spacy_model: str
    chunk_size_in_kb: float

# API Endpoints
@app.post("/api/prompt")
async def handle_prompt(request: PromptRequest):
    answer = model.chat(request.question, model=request.llm_model)
    return {"answer": f"{answer}"}

@app.post("/api/search")
async def handle_search(search: SearchQuery):
    embedding = Embedding(spacy_model=search.spacy_model, chunk_size_in_kb=search.chunk_size_in_kb, model=search.model)
    es = Search(embedding=embedding)
    results = es.search(search.query)
    return {"results": results}

@app.post("/api/rag")
async def handle_rag(request: PromptRequest):
    embedding = Embedding(spacy_model=request.spacy_model, chunk_size_in_kb=request.chunk_size_in_kb, model=request.model)
    es = Search(embedding=embedding)
    rag_output = model.rag(question=request.question, es=es, model=request.llm_model)
    return {"context": f"{rag_output[1]}", "answer": f"{rag_output[0]}"}

# Web UI Route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def form_handler(
    request: Request,
    question: str = Form(""),
    task: str = Form(...),
    model: str = Form(...),
    spacy_model: str = Form(...),
    chunk_size_in_kb: float = Form(...),
    llm_model: str = Form("llama3.2")
):
    result = ""
    context = ""
    if task == "chat":
        response = await handle_prompt(PromptRequest(
            question=question,
            model=model,
            spacy_model=spacy_model,
            chunk_size_in_kb=chunk_size_in_kb,
            llm_model=llm_model
        ))
        result = response["answer"]
    elif task == "rag":
        response = await handle_rag(PromptRequest(
            question=question,
            model=model,
            spacy_model=spacy_model,
            chunk_size_in_kb=chunk_size_in_kb,
            llm_model=llm_model
        ))
        result = response["answer"]
        context = response["context"]
    elif task == "search":
        response = await handle_search(SearchQuery(
            query=question,
            model=model,
            spacy_model=spacy_model,
            chunk_size_in_kb=chunk_size_in_kb
        ))
        result = "\n".join(response["results"])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": question,
        "task": task,
        "model": model,
        "spacy_model": spacy_model,
        "chunk_size_in_kb": chunk_size_in_kb,
        "llm_model": llm_model,
        "result": result,
        "context": context
    })
