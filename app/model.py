import ollama
from search import Search

class Model:

    def chat(self, question):
        response = ollama.chat(model='llama3.2', messages=[{ 'role': 'user', 'content': question }])
        return response['message']['content']

    def rag(self, question):
        es = Search()
        retrieved_text = es.search(question)
        response = ollama.chat(model='llama3.2', messages=[{ 'role': 'user', 'content': f"Beantworte die folgende Frage {question}. \n Nutze dafür folgende Information: {retrieved_text}" }])
        return response['message']['content']
