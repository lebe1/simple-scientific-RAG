import requests
from datetime import datetime, timedelta, timezone

# Function to read questions file
def read_questions_from_file(filename):
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
    payload = {'question': question}  
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
    questions_file = '../data/sample_questions.txt' 
    local_time = datetime.now(timezone.utc) + timedelta(hours=2)

    output_file = f"../data/generated_answers{local_time.strftime('%Y-%m-%d %H:%M:%S')}.txt" 

    # Read questions from the file
    questions = read_questions_from_file(questions_file)

    if not questions:
        print("No questions found to process.")
        return

    answers = []  # List to store answers

    # Iterate through each question and store the response
    for i, question in enumerate(questions):
        print(f"Sending Question {i + 1}: {question}")
        answer = post_question(question)
        if answer:
            answers.append(f"{i + 1}. {answer}")
        else:
            answers.append(f"{i + 1}. No Answer Received")

    # Write the answers to a text file
    with open(output_file, 'w') as file:
        for answer in answers:
            file.write(answer + '\n')

    print(f"Questions and responses saved to {output_file}")

query()