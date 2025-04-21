# Default is --model jinaai/jina-embeddings-v2-base-de
python app/workflow.py create-embeddings --chunk-size 0.125
python app/workflow.py update-es-index --chunk-size 0.125
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":0.125}'
python app/workflow.py create-embeddings --chunk-size 0.25
python app/workflow.py update-es-index --chunk-size 0.25
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":0.25}'
python app/workflow.py create-embeddings --chunk-size 0.5
python app/workflow.py update-es-index --chunk-size 0.5
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":0.5}'
python app/workflow.py create-embeddings --chunk-size 1
python app/workflow.py update-es-index --chunk-size 1
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":1}'
python app/workflow.py create-embeddings --chunk-size 2
python app/workflow.py update-es-index --chunk-size 2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":2}'

# Second model is --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py create-embeddings --chunk-size 0.125 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py update-es-index --chunk-size 0.125 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":0.125}'
python app/workflow.py create-embeddings --chunk-size 0.25 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py update-es-index --chunk-size 0.25 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":0.25}'
python app/workflow.py create-embeddings --chunk-size 0.5 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py update-es-index --chunk-size 0.5 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":0.5}'
python app/workflow.py create-embeddings --chunk-size 1 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py update-es-index --chunk-size 1 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":1}'
python app/workflow.py create-embeddings --chunk-size 2 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py update-es-index --chunk-size 2 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":2}'