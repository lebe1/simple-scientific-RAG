import ollama
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.metrics.faithfulness import FaithfulnessTemplate

# Define custom template
class CustomTemplate(FaithfulnessTemplate):
    @staticmethod
    def generate_claims(actual_output: str):
        return f"""Erstellen Sie auf der Grundlage des gegebenen Textes eine umfassende Liste von Fakten, die sich aus den gegebenen Text ableiten lässt.

Beispiel:
Beispieltext:
„CNN behauptet, dass die Sonne dreimal kleiner als die Erde ist.“

Beispiel JSON:
{{
 "claims": []
}}
===== ENDE DES BEISPIELS ======

Text:
{actual_output}

JSON:
"""
    
# Define custom template
class CustomTemplate(AnswerRelevancyTemplate):
    @staticmethod
    def generate_statements(actual_output: str):
        return f"""Gliedern Sie den Text auf und erstellen Sie eine Liste der dargestellten Aussagen.

            Beispiel:
            Unser neues Laptop-Modell verfügt über ein hochauflösendes Retina-Display für eine kristallklare Darstellung.

            {{
            "statements": [
                "Das neue Laptop-Modell hat ein hochauflösendes Retina-Display."
            ]
            }}
            ===== ENDE DES BEISPIELS ======

            Text:
            {actual_output}

            JSON:
        """

class OllamaLlama3(DeepEvalBaseLLM):
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
answer_relevancy_metric = AnswerRelevancyMetric(model=llama3, threshold=0.7)

contextual_recall_metric = ContextualRecallMetric(
    threshold=0.7,
    model=llama3,
    include_reason=True
)

contextual_precision_metric = ContextualPrecisionMetric(
    threshold=0.7,
    model=llama3,
    include_reason=True
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=llama3,
    include_reason=True
)


evaluate(test_cases=[test_case], metrics=[answer_relevancy_metric, contextual_recall_metric, contextual_precision_metric, faithfulness_metric])