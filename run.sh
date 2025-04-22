python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
cp .env.example .env
sudo curl -L "https://github.com/docker/compose/releases/download/v2.29.2/docker-compose-$(uname -s)-$(uname -m)"  -o /usr/local/bin/docker-compose
sudo mv /usr/local/bin/docker-compose /usr/bin/docker-compose
sudo chmod +x /usr/bin/docker-compose
docker-compose build
docker-compose up -d
docker exec ollama ollama run llama3-chatqa:8b
docker exec ollama ollama run gemma3:12b
python -m spacy download de_core_news_lg
docker-compose up -d
python app/workflow.py update-es-index
cd app;
uvicorn main:app --reload
