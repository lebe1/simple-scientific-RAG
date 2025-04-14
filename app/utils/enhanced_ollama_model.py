import ollama
import json
import os
from deepeval.models.base_model import DeepEvalBaseLLM
from utils.json_fix import sanitize_llm_response


class EnhancedOllamaModel(DeepEvalBaseLLM):
    """
    Enhanced Ollama model with improved JSON handling for evaluation metrics.
    """

    def __init__(self, model_name: str = "llama3.2", max_retries: int = 2, enforce_json: bool = True):
        self.model_name = model_name
        self.max_retries = max_retries
        self.enforce_json = enforce_json

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        """
        Generates a response from Ollama with improved JSON handling.

        If enforce_json is True, it will:
        1. First try to generate with format='json'
        2. If that fails, try to generate normally and process the output
        3. Retry up to max_retries times if JSON is still invalid
        """
        # Try using format=json first if enforce_json is True
        if self.enforce_json:
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'num_predict': 1000,  # Larger token limit for complex responses
                        'temperature': 0,
                        'format': 'json'  # Request JSON format explicitly
                    }
                )

                # Try to parse the JSON directly
                try:
                    json_data = sanitize_llm_response(response['response'])
                    return json.dumps(json_data)
                except Exception:
                    # If JSON parsing fails, we'll try again without format=json
                    pass
            except Exception as e:
                print(f"JSON format generation failed: {e}")

        # If the json format option failed or wasn't used, try normal generation
        retry_count = 0
        last_error = None

        while retry_count <= self.max_retries:
            try:
                # Add explicit instructions to format as JSON
                enhanced_prompt = (
                    f"{prompt}\n\n"
                    """WICHTIG: Ihre Antwort MUSS ein gültiges JSON sein. Formatieren Sie Ihre gesamte Antwort als ein
                    einzelnes JSON-Objekt ohne zusätzlichen Text, Markdown oder Erklärungen davor oder danach. 
                    Achten Sie darauf, dass alle Anführungszeichen doppelte Anführungszeichen sind und alle Schlüssel korrekt in Anführungszeichen gesetzt sind.
                    Die Ausgabe soll direkt in Python geladen werden können und nicht in Markdown sein.
                    Beginnen Sie das JSON-Objekt nicht mit ```json."""
                )

                # Generate response
                response = ollama.generate(
                    model=self.model_name,
                    prompt=enhanced_prompt,
                    options={
                        'num_predict': 1000,
                        'temperature': 0,
                        'format': "json",
                        'raw': True
                    }
                )
                # Try to extract valid JSON
                json_data = sanitize_llm_response(response['response'])
                return json.dumps(json_data)

            except Exception as e:
                print(f"Attempt {retry_count + 1} failed: {e}")
                last_error = e
                retry_count += 1

        # If we've exhausted all retries, raise an error
        raise ValueError(f"Failed to generate valid JSON after {self.max_retries} retries: {last_error}")

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"Ollama {self.model_name}"