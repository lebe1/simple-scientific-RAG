import ollama
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.metrics.faithfulness import FaithfulnessTemplate
from templates.answer_relevancy_template import AnswerRelevancyTemplate
from templates.contextual_precision_template import ContextualPrecisionTemplate
from templates.contextual_recall_template import ContextualRecallTemplate
from templates.faithfulness_template import FaithfulnessTemplate

class OllamaLlama3(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={'num_predict': 200, 'temperature':0}, # num_predict to set maximum number of tokens to predict
            format='json'
        )
        print("Response generated", response['response'])
        return response['response']

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"Ollama {self.model_name}"
    

# Make sure Ollama is running and you've pulled the model first:
# ollama pull llama3
llama3 = OllamaLlama3()


test_case = LLMTestCase(input="Was ist die Hauptstadt von Rumänien?", actual_output="Paris ist die Hauptstadt von Rumänien.", expected_output="Bukarest ist die Hauptstadt von Rumänien.", retrieval_context=["Bukarest, die zweitgrößte Stadt Rumäniens und bekannt als ihre Hauptstadt, hat 50000 Einwohner."])

answer_relevancy_metric = AnswerRelevancyMetric(model=llama3, threshold=0.7, evaluation_template=AnswerRelevancyTemplate, strict_mode=True)
answer_relevancy_metric.measure(test_case)
print("Score: ", answer_relevancy_metric.score)
print("Reason: ", answer_relevancy_metric.reason)

contextual_recall_metric = ContextualRecallMetric(
    threshold=0.7,
    model=llama3,
    include_reason=True,
    evaluation_template=ContextualRecallTemplate,
    strict_mode=True
)
contextual_recall_metric.measure(test_case)
print("Score: ", contextual_recall_metric.score)
print("Reason: ", contextual_recall_metric.reason)

contextual_precision_metric = ContextualPrecisionMetric(
    threshold=0.7,
    model=llama3,
    include_reason=True,
    evaluation_template=ContextualPrecisionTemplate,
    strict_mode=True
)
contextual_precision_metric.measure(test_case)
print("Score: ", contextual_precision_metric.score)
print("Reason: ", contextual_precision_metric.reason)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=llama3,
    include_reason=True,
    evaluation_template=FaithfulnessTemplate,
    strict_mode=True
)

faithfulness_metric.measure(test_case)
print("Score: ", faithfulness_metric.score)
print("Reason: ", faithfulness_metric.reason)

# In case, I want to evaluate all 10 Q&A's at once, the line below is preferable
# evaluate(test_cases=[test_case], metrics=[answer_relevancy_metric, contextual_recall_metric, contextual_precision_metric, faithfulness_metric])