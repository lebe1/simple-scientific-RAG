import argparse
import requests
from datetime import datetime, timedelta, timezone
import json
from individual_eval import IndividualEval

def read_lines_from_file(filename):
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return []

def post_question(question, embedding_model, spacy_model, top_k_chunks, llm_model, chunk_size=4):
    url = "http://localhost:8000/api/rag"
    headers = {'Content-Type': 'application/json'}
    payload = {
        'question': question,
        "model": embedding_model,
        "spacy_model": spacy_model,
        "chunk_size_in_kb": chunk_size,
        "top_k_chunks":top_k_chunks,
        "llm_model": llm_model
    }
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

def run_eval(top_k, embedding_model, llm_model, split_method):
    questions = read_lines_from_file('../../data/sample_questions.txt')
    references = read_lines_from_file('../../data/sample_answers.txt')
    
    if not questions or not references:
        print("Missing questions or reference answers.")
        return
    
    for k in top_k:
        now = datetime.now(timezone.utc) + timedelta(hours=2)
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        label = embedding_model.split("/")[-1].replace("-", "_").upper()
        answer_file = f"../../data/{label}_TOP{k}_{split_method}_{llm_model}_generated_answers{timestamp}.txt"
        eval_file = f"../../data/{label}_TOP{k}_{split_method}_{llm_model}_evaluation_results{timestamp}.json"

        answers = []
        contexts = []

        for i, (question, ref) in enumerate(zip(questions, references)):
            print(f"TOP{k} - Sending Q{i+1}: {question}")
            result = post_question(question, embedding_model, "de_core_news_lg", k, llm_model)
            if result:
                answers.append(f"{i+1}. {result['answer']}")
                contexts.append(f"{i+1}. {result['context']}")
            else:
                answers.append(f"{i+1}. No Answer Received")
                contexts.append(f"{i+1}. No Context Received")

        with open(answer_file, 'w') as f:
            for a, c in zip(answers, contexts):
                f.write("ANTWORT\n" + a + "\n")
                f.write("KONTEXT\n" + c + "\n")

        results = []
        for index in range(len(questions)):
            eval = IndividualEval(
                input=questions[index],
                actual_output=answers[index],
                expected_output=references[index],
                retrieval_context=contexts[index],
                index=index+1
            )
            results.append(eval.evaluate())

        total_output = sum(r["output_score"] for r in results)
        total_retrieval = sum(r["retrieval_score"] for r in results)
        results.append({"total_output_score": total_output, "total_retrieval_score": total_retrieval})

        with open(eval_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"TOP{k} results saved to {answer_file} and {eval_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-model', required=True, help='Embedding model name')
    parser.add_argument('--llm-model', required=True, help='LLM model name')
    parser.add_argument('--select-top-k', nargs='+', type=int, required=True, help='Top K values (e.g. 3 5)')
    parser.add_argument('--splitting-method', choices=["BASELINE", "ARTICLE", "SUBARTICLE"], required=True, help='Document splitting method')
    args = parser.parse_args()

    run_eval(args.select_top_k, args.embedding_model, args.llm_model, args.splitting_method)
