from typing import List


class ContextualRecallTemplate:
    @staticmethod
    def generate_reason(
        expected_output: str,
        supportive_reasons: str,
        unsupportive_reasons: str,
        score: float,
    ):
        return f"""
Ausgehend von der ursprünglich erwarteten Ausgabe (expected output), einer Liste von unterstützenden Gründen (supportive reasons) und einer Liste von nicht unterstützenden Gründen (unsupportive reasons), die direkt aus der „erwarteten Ausgabe“ (expected output) abgeleitet werden, sowie einer kontextbezogenen Abrufbewertung (contextual recall score), die je näher an der 1, umso besser ist, fassen Sie eine PRÄZISE Begründung für die Bewertung zusammen.
Ein unterstützender Grund ist der Grund, warum ein bestimmter Satz in der ursprünglichen erwarteten Ausgabe dem Knoten im Abrufkontext zugeordnet werden kann.
Ein nicht-unterstützender Grund (unsupportive reason) ist der Grund, warum ein bestimmter Satz in der ursprünglich erwarteten Ausgabe keinem Knoten im Abfragekontext (retrieval context) zugeordnet werden kann.
In Ihrer Begründung sollten Sie unterstützende/ununterstützende Gründe mit der Satznummer in der erwarteten Ausgabe in Beziehung setzen und Informationen über die Knotennummer im Abfragekontext einfügen, um Ihre endgültige Begründung zu stützen. 

**
WICHTIG: Bitte stellen Sie sicher, dass Sie nur im JSON-Format zurückgeben, wobei der Schlüssel „reason“ den Grund angibt.
Beispiel JSON:
{{
 "reason": „Das Ergebnis ist <contextual_recall_score> weil <deine_Begründung>.“
}}

Erwähnen Sie NICHT „unterstützende Gründe“ und „nicht unterstützende Gründe“ in Ihrer Begründung, diese Begriffe sind nur dazu da, damit Sie den weiteren Umfang der Dinge verstehen.
Wenn die Punktzahl 1 ist, fassen Sie sich kurz und sagen Sie etwas Positives in einem aufmunternden, ermutigenden Ton (aber übertreiben Sie es nicht, sonst wird es lästig).
**

Kontextbezogene Abrufbewertung (Contextual recall score):
{score}

Erwartete Ausgabe:
{expected_output}

Unterstützende Gründe:
{supportive_reasons}

Nicht unterstützende Gründe:
{unsupportive_reasons}

JSON:
"""

    @staticmethod
    def generate_verdicts(expected_output: str, retrieval_context: List[str]):
        return f"""
For EACH sentence in the given expected output below, determine whether the sentence can be attributed to the nodes of retrieval contexts. Please generate a list of JSON with two keys: `verdict` and `reason`.
The `verdict` key should STRICTLY be either a 'yes' or 'no'. Answer 'yes' if the sentence can be attributed to any parts of the retrieval context, else answer 'no'.
The `reason` key should provide a reason why to the verdict. In the reason, you should aim to include the node(s) count in the retrieval context (eg., 1st node, and 2nd node in the retrieval context) that is attributed to said sentence. You should also aim to quote the specific part of the retrieval context to justify your verdict, but keep it extremely concise and cut short the quote with an ellipsis if possible. 


**
WICHTIG: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects, each with two keys: `verdict` and `reason`.

{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "..."
        }},
        ...
    ]  
}}

Since you are going to generate a verdict for each sentence, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to the number of sentences in `expected output`.
**

Bestimmen Sie für JEDEN Satz in der unten angegebenen erwarteten Ausgabe, ob der Satz den Knoten der Abfragekontexte zugeordnet werden kann. Bitte erzeugen Sie eine JSON-Liste mit zwei Schlüsseln: `verdict` und `reason`.
Der Schlüssel „verdict“ sollte EXAKT entweder ein "yes" oder ein "no" sein. Antworten Sie mit „yes“, wenn der Satz irgendeinem Teil des Abfragekontextes zugeordnet werden kann, andernfalls mit "no".
Der Schlüssel "reason" sollte eine Begründung des Urteils enthalten. In der Begründung sollten Sie versuchen, die Anzahl der Knoten im Abfragekontext anzugeben (z. B. 1. Knoten und 2. Knoten im Abfragekontext), die dem genannten Satz zugeordnet werden. Sie sollten auch versuchen, den spezifischen Teil des Abrufkontextes zu zitieren, um Ihr Urteil zu begründen, aber halten Sie es extrem kurz und kürzen Sie das Zitat mit einer Ellipse ab, wenn möglich. 


**
WICHTIG: Bitte stellen Sie sicher, dass die Rückgabe nur im JSON-Format erfolgt, mit dem Schlüssel „verdicts“ als Liste von JSON-Objekten, jedes mit zwei Schlüsseln: `verdict` und `reason`.

{{
 "verdicts": [
 {{
 "verdict": "yes",
 "reason": "..."
 }},
...
 ] 
}}

Da für jeden Satz ein Urteil erzeugt werden soll, MUSS die Anzahl der „verdicts“ EXAKT der Anzahl der Sätze in „Erwartete Ausgabe“ (expected output) entsprechen.
**

Expected Output:
{expected_output}

Retrieval Context:
{retrieval_context}

JSON:
"""