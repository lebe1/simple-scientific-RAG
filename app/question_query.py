import requests
from datetime import datetime, timedelta, timezone
import nltk


# Function to read questions file
def read_lines_from_file(filename):
    questions = []
    try:
        with open(filename, 'r') as file:
            questions = [line.strip() for line in file.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    return questions

# Function to make POST request with the question and return the response
def post_question(question):
    url = "http://localhost:8000/api/rag"  
    headers = {'Content-Type': 'application/json'} 
    payload = {'question': question, "model":"jinaai/jina-embeddings-v2-base-de", "spacy_model":"de_core_news_lg", "chunk_size_in_kb":4}  
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get('answer', 'No Answer provided')  
        else:
            print(f"Error: Received status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None
    
# Main function to read questions, send requests, and store answers
def query():
    questions_file = '../data/sample_questions2.txt' 
    answers_file = '../data/sample_answers2.txt'
    local_time = datetime.now(timezone.utc) + timedelta(hours=2)

    output_file = f"../data/generated_answers{local_time.strftime('%Y-%m-%d %H:%M:%S')}.txt" 

    # Read questions from the file
    questions = read_lines_from_file(questions_file)
    references = read_lines_from_file(answers_file)

    if not questions:
        print("No questions found to process.")
        return
    
    if not references:
        print("No answers found to process.")
        return 

    answers = []
    scores = {}

    # Iterate through each question and store the response
    for i, question_reference in enumerate(zip(questions, references)):
        print(f"Sending Question {i + 1}: {question_reference[0]}")
        answer = post_question(question_reference[0])
        if answer:
            answers.append(f"{i + 1}. {answer}")
            scores[i+1] = nltk.translate.bleu_score.sentence_bleu([question_reference[1].split()], answer.split())
            
        else:
            answers.append(f"{i + 1}. No Answer Received")

    # Write the answers to a text file
    with open(output_file, 'w') as file:
        for answer, scores in zip(answers,scores):
            file.write(answer + '\n')
            file.write(str(scores) + '\n')

    print(f"Questions and responses saved to {output_file}")

query()