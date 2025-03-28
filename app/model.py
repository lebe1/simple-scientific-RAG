import ollama
class Model:

    def chat(self, question):
        response = ollama.chat(model='llama3.2', messages=[{ 'role': 'user', 'content': question }])
        return response['message']['content']

    def rag(self, question, es):
        print(f"Accessing RAG system...")
        retrieved_text = es.search(question)
        print(f"Passing the text to the LLM...")
        response = ollama.chat(model='llama3.2', options={'temperature': 0.0}, messages=[{ 'role': 'user', 'content': f"Du hast folgende Information: {retrieved_text}. \n Antworte auf die folgende Frage sehr präzise in nur einem Satz und mit so wenigen Worten wie möglich. {question}" }])
        return response['message']['content']
