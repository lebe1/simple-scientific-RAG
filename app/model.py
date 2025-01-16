import ollama
class Model:

    def chat(self, question):
        response = ollama.chat(model='llama3.2', messages=[{ 'role': 'user', 'content': question }])
        return response['message']['content']

    def rag(self, question, es):
        retrieved_text = es.search(question)
        response = ollama.chat(model='llama3.2', messages=[{ 'role': 'user', 'content': f"Du hast folgende Information: {retrieved_text}. \n Welcher Teil davon ist relevant, um folgende Frage zu beantworten: {question}. " }])
        return response['message']['content']
