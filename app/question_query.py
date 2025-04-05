import requests
from datetime import datetime, timedelta, timezone
import ollama
import json
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from templates.answer_relevancy_template import AnswerRelevancyTemplate
from templates.contextual_precision_template import ContextualPrecisionTemplate
from templates.contextual_recall_template import ContextualRecallTemplate
from templates.contextual_relevancy_template import ContextualRelevancyTemplate
from templates.faithfulness_template import FaithfulnessTemplate


class CustomOllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={'num_predict': 100, 'temperature':0}
        )
        return response['response']

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"Ollama {self.model_name}"


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
def query():
    questions_file = '../data/sample_questions.txt' 
    answers_file = '../data/sample_answers.txt'
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
            file.write(answer + '\n')
            file.write(context + '\n')

    print(f"Questions and responses saved to {output_file}")

    # Evaluate via LLM-as-a-judge approach
    llama3 = CustomOllamaModel()

    answer_relevancy_metric = AnswerRelevancyMetric(model=llama3, threshold=0.7, evaluation_template=AnswerRelevancyTemplate, strict_mode=True)

    contextual_recall_metric = ContextualRecallMetric(
        threshold=0.7,
        model=llama3,
        include_reason=True,
        evaluation_template=ContextualRecallTemplate,
        strict_mode=True
    )

    contextual_precision_metric = ContextualPrecisionMetric(
        threshold=0.7,
        model=llama3,
        include_reason=True,
        evaluation_template=ContextualPrecisionTemplate,
        strict_mode=True
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=llama3,
        include_reason=True,
        evaluation_template=FaithfulnessTemplate,
        strict_mode=True
    )

    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold=0.7,
        model=llama3,
        include_reason=True,
        evaluation_template=ContextualRelevancyTemplate,
        strict_mode=True
    )


    results = []

    for index, _ in enumerate(questions):
        test_case = LLMTestCase(
            input=questions[index],
            actual_output=answers[index],
            expected_output=references[index],
            retrieval_context=[contexts[index]]
        )

        metrics_result = {
            "question": questions[index],
            "answer": answers[index],
            "reference": references[index],
            "context": contexts[index],
            "metrics": {}
        }

        # Measure and store all metrics
        for metric_name, metric in {
            "answer_relevancy": answer_relevancy_metric,
            "contextual_recall": contextual_recall_metric,
            "contextual_precision": contextual_precision_metric,
            "faithfulness": faithfulness_metric,
            "contextual_relevancy": contextual_relevancy_metric
        }.items():
            try:
                metric.measure(test_case)
                score = metric.score
                reason = metric.reason
            except Exception as e:
                score = ""
                reason = ""
                print(f"[ERROR] {metric_name} failed for index {index}: {e}")

            metrics_result["metrics"][metric_name] = {
                "score": score,
                "reason": reason
            }

        results.append(metrics_result)

    # Save results to JSON file
    with open(f"../data/evaluation_results{local_time.strftime('%Y-%m-%d %H:%M:%S')}.json", "w") as f:
        json.dump(results, f, indent=4)


query()