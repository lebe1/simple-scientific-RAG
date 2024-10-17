# Simple scientific RAG

## Requirements

Make sure you have Python 3.7+ installed on your machine.

## Setup Instructions

1. **Clone the repository** (if applicable):

    ```bash
    git clone git@github.com:lebe1/simple-scientific-RAG.git
    cd simple-scientific-RAG
    ```

2. **Create a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:

    Before running the project, install all the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run Ollama on docker and install llama3.2 model**:  

   Run docker-compose.yml to pull the required image:
   ```bash
   docker compose up -d
   ```

   Pull the required model llama3.2 by running:
   ```bash
   docker exec -it ollama ollama run llama3.2
   ```

## Running the application after setup instructions

To start the FastAPI development server, run the following command:

```bash
fastapi dev main.py
```


### Testing the API

There are two ways for testing the API.  
Either by sending the following POST-request using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/api/prompt" -H "Content-Type: application/json" -d '{"question": "What is a book?"}'
```
Or by opening the built-in Swagger of FastAPI via `http://127.0.0.1:8000/docs`



## Running the automated question query

Open a second terminal in the same directory and run:

```bash
python question_query.py
```

## Data
The data of the legal basis can be found under https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=LrW&Gesetzesnummer=20000006


