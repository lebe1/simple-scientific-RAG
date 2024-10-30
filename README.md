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

4. **Build Dockerfile**:

   > **Note**: To use the docker-compose command below, you need to have a valid `.env` file in the directory. You can use the `env.example` as template (`cp .env.example .env`)
   
   ```bash
   docker compose build
   ```

   > **Note**: In some Linux distributions and depending on the docker version, `docker compose` is not recognized as a command. In such case, download the binary and proceed with `docker-compose`:

   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.29.2/docker-compose-$(uname -s)-$(uname -m)"  -o /usr/local/bin/docker-compose
   sudo mv /usr/local/bin/docker-compose /usr/bin/docker-compose
   sudo chmod +x /usr/bin/docker-compose   
   docker-compose build   
   ```

5. **Run docker container**:  

   Run docker-compose.yml to pull the required image:
   ```bash
   docker compose up -d
   ```
6. **Install Ollama model llama3.2**:

   Pull the required model llama3.2 by running:

   ```bash
   docker exec ollama ollama run llama3.2
   ```

   If you would like to step inside the container, you can add the `-it` flags to the `docker exec` command.

7. **Install model for chunking**:

    ```bash
    python -m spacy download en_core_web_sm
    ```

## Running the application after setup instructions

Once everything is set up, simply start the whole application with:

```bash
docker compose up -d
```

Run fastapi server locally:
```bash
cd app;
uvicorn main:app --reload
```

### Testing the API

There are two ways for testing the API.  
Either by sending the following POST-request using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?"}'
```
```bash
curl -X POST "http://127.0.0.1:8000/api/search" -H "Content-Type: application/json" -d '{"query": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?"}'
```
Or by opening the built-in Swagger of FastAPI via `http://127.0.0.1:8000/docs`



## Running the automated question query

For now, this only works when the fastapi server is called outside of docker. If the fastapi server is running on docker, you need to stop it first, to be able to execute the following command:

```bash
cd app
```

```bash
uvicorn main:app --reload
```

Open a second terminal in the same directory and run:

```bash
python question_query.py
```

## Data
The data of the legal basis can be found under https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=LrW&Gesetzesnummer=20000006

# Fancy TODOs
- Store everything in the database to use variables in RAM as less as possible
- Mount data directory in fastapi server container
- Create API call to trigger question_query.py
- Clean up starting processes into one single docker-compose.yml
- Work with logs instead of prints
- Use try catch phrases for better error detection
- Add CI for linting
- Add tests?

# Actual TODOs
- Finish dataset with 10 questions
- Fix question_query.py