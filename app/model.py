import ollama


class Model:
    def __init__(self, default_model='llama3.2'):
        self.default_model = default_model

    def chat(self, question, model=None):
        model_name = model or self.default_model
        response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': question}])
        return response['message']['content']

    def rag(self, question, es, model=None, temperature=0):
        context = es.search(question)
        print(f"Passing the text to the LLM...")
        model_name = model or self.default_model

        response = ollama.chat(
            model=model_name,
            options={'temperature': temperature},
            messages=[{
                'role': 'user',
                'content': f"Du hast folgende Information: {context}. \n Antworte auf die folgende Frage in einem vollen Satz mit mindestens 5 Wörtern und maximal 15 Wörtern. Gebe dabei immer deine Quellen an. {question}"
            }]
        )
        return [response['message']['content'], context]