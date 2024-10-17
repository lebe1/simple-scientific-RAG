import ollama

def chat(question):
    response = ollama.chat(model='llama3.2', messages=[{ 'role': 'user', 'content': question }])
    return response['message']['content']