import ollama
class Model:

    def chat(self, question):
        response = ollama.chat(model='llama3.2', messages=[{ 'role': 'user', 'content': question }])
        return response['message']['content']

    def rag(self, question, es):
        context = es.search(question)
        print(f"Passing the text to the LLM...")
        response = ollama.chat(model='llama3.2', options={'temperature': 0}, messages=[{ 'role': 'user', 'content': f"Du hast folgende Information: {context}. \n Antworte auf die folgende Frage in einem vollen Satz mit mindestens 5 Wörtern und maximal 15 Wörtern. {question}" }])
        return [response['message']['content'], context]
