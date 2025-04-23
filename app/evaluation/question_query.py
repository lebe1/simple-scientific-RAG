import requests
from datetime import datetime, timedelta, timezone
import ollama
import json
from individual_eval import IndividualEval


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
            response_data = response.json()
            return {
                "context": response_data.get('context', 'No context provided'),
                "answer": response_data.get('answer', 'No answer provided')
            }
        else:
            print(f"Error: Received status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None
    
# Main function to read questions, send requests, and store answers
if __name__ == "__main__":
    questions_file = '../../data/sample_questions.txt' 
    answers_file = '../../data/sample_answers.txt'
    local_time = datetime.now(timezone.utc) + timedelta(hours=2)

    output_file = f"../../data/generated_answers{local_time.strftime('%Y-%m-%d %H:%M:%S')}.txt" 

    # Read questions from the file
    questions = read_lines_from_file(questions_file)
    references = read_lines_from_file(answers_file)

    if not questions:
        print("No questions found to process.")       
    
    if not references:
        print("No answers found to process.") 

    answers = []
    contexts = []


    # Iterate through each question and store each response and its context
    for i, question_reference in enumerate(zip(questions, references)):
        print(f"Sending Question {i + 1}: {question_reference[0]}")
        rag_output = post_question(question_reference[0])
        if rag_output:
            answers.append(f"{i + 1}. {rag_output['answer']}")
            contexts.append(f"{i + 1}. {rag_output['context']}")
        else:
            answers.append(f"{i + 1}. No Answer Received")

    # Write the answers to a text file
    with open(output_file, 'w') as file:
        for answer, context in zip(answers,contexts):
            file.write('ANTWORT\n' + answer + '\n')
            file.write('KONTEXT\n' + context + '\n')

    print(f"Questions and responses saved to {output_file}")

    
    results = []

    for index, _ in enumerate(questions):
        individual_eval = IndividualEval(
            input=questions[index],
            actual_output=answers[index],
            expected_output=references[index],
            retrieval_context=contexts[index],
            index=index+1
        )

        metrics_result = individual_eval.evaluate()

        results.append(metrics_result)

    total_output_score = sum(r["output_score"] for r in results)
    total_retrieval_score = sum(r["retrieval_score"] for r in results)

    results.append({"total_output_score":total_output_score, "total_retrieval_score":total_retrieval_score})

    # Save results to JSON file
    with open(f"../../data/evaluation_results{local_time.strftime('%Y-%m-%d %H:%M:%S')}.json", "w") as f:
        json.dump(results, f, indent=4)
