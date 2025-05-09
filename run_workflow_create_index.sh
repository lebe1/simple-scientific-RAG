# Default is --model jinaai/jina-embeddings-v2-base-de
python app/workflow.py create-embeddings --chunk-size 4
python app/workflow.py update-es-index --chunk-size 4
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":4}'
python app/workflow.py create-embeddings --chunk-size 8
python app/workflow.py update-es-index --chunk-size 8
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":8}'
python app/workflow.py create-embeddings --chunk-size 16
python app/workflow.py update-es-index --chunk-size 16
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":16}'
python app/workflow.py create-embeddings --chunk-size 32
python app/workflow.py update-es-index --chunk-size 32
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":32}'
python app/workflow.py create-embeddings --chunk-size 64
python app/workflow.py update-es-index --chunk-size 64
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":64}'
python app/workflow.py create-embeddings --chunk-size 128
python app/workflow.py update-es-index --chunk-size 128
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":128}'

# Second model is --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py create-embeddings --chunk-size 4 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py update-es-index --chunk-size 4 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":4}'
python app/workflow.py create-embeddings --chunk-size 8 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py update-es-index --chunk-size 8 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":8}'
python app/workflow.py create-embeddings --chunk-size 16 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py update-es-index --chunk-size 16 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":16}'
python app/workflow.py create-embeddings --chunk-size 32 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py update-es-index --chunk-size 32 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":32}'
python app/workflow.py create-embeddings --chunk-size 64 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py update-es-index --chunk-size 64 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":64}'
python app/workflow.py create-embeddings --chunk-size 128 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python app/workflow.py update-es-index --chunk-size 128 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
curl -X POST "http://127.0.0.1:8000/api/rag" -H "Content-Type: application/json" -d '{"question": "Wie hoch darf ein Gebäude in Bauklasse I gemäß Artikel IV in Wien sein?", "model":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":128}'
