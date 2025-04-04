from typing import List, Dict


class ContextualPrecisionTemplate:
    @staticmethod
    def generate_verdicts(
        input: str, expected_output: str, retrieval_context: List[str]
    ):
        document_count_str = f" ({len(retrieval_context)} document{'s' if len(retrieval_context) > 1 else ''})"
        return f"""Erzeugen Sie anhand der Eingabe (input), der erwarteten Ausgabe (expected output) und des Abfragekontexts (retrieval context) eine Liste von JSON-Objekten, um festzustellen, ob jeder Knoten (node) im Abfragekontext auch nur im Entferntesten nützlich war, um die erwartete Ausgabe zu erhalten.

**
WICHTIG: Bitte stellen Sie sicher, dass die Rückgabe nur im JSON-Format erfolgt, mit dem Schlüssel 'verdicts' als Liste von JSON. Diese JSON enthalten nur den „verdict“-Schlüssel, der nur „yes“ oder ‚no‘ ausgibt, und einen „reason“-Schlüssel zur Begründung des Urteils. In Ihrer Begründung sollten Sie versuchen, Teile des Kontexts zu zitieren.
Beispiel Abfragekontext (retrieval context): [„Einstein erhielt den Nobelpreis für seine Entdeckung des photoelektrischen Effekts“, „Er erhielt den Nobelpreis 1968.“, „Es gab eine Katze.“]
Beispiel Eingabe (input): „Wer gewann 1968 den Nobelpreis und wofür?“
Beispiel Erwartete Ausgabe (expected output): „Einstein gewann 1968 den Nobelpreis für seine Entdeckung des photoelektrischen Effekts.“

Beispiel:
{{
 "verdicts": [
    {{
    "verdict": "yes",
    "reason": "Die Frage wird eindeutig beantwortet, indem gesagt wird, dass ‚Einstein den Nobelpreis für seine Entdeckung des photoelektrischen Effekts erhalten hat.‘“
            }},
    {{
    "verdict": "yes",
    "reason": "Der Text verifiziert, dass der Preis tatsächlich 1968 gewonnen wurde.“
            }},
    {{
    "verdict": "no",
    "reason": „‚Es gab eine Katze‘ ist für das Thema Nobelpreisverleihung überhaupt nicht relevant.“
            }}
    ]  
}}
Da für jeden Kontext ein Urteil erstellt werden soll, MUSS die Anzahl der "verdicts", also der Urteile, EXAKT gleich der Anzahl der Kontexte sein.
**

Eingabe:
{input}

Erwartete Ausgabe:
{expected_output}

Abfragekontext{document_count_str}:
{retrieval_context}

JSON:
"""

    @staticmethod
    def generate_reason(
        input: str, score: float, verdicts: List[Dict[str, str]]
    ):
        # given the input and retrieval context for this input, where the verdict is whether ... and the node is the ..., give a reason for the score
        return f"""Geben Sie angesichts der Eingabe (input), der Abfragekontexte (retrieval contexts) und der kontextbezogenen Präzisionsbewertung (contextual precision score) eine PRÄZISE Zusammenfassung der Bewertung. Erläutern Sie, warum der Wert nicht höher ist, aber auch, warum er den aktuellen Wert erreicht hat.
Die Abfragekontexte sind eine JSON-Liste mit drei Schlüsseln: "verdict", "reson" (Grund für das Urteil) und "node". Der Wert "verdict" ist entweder "yes" oder "no", was bedeutet, dass der entsprechende "node" (Knoten) im Abfragekontext für die Eingabe (input) relevant ist. 
Die Kontextpräzision gibt an, ob die relevanten Knoten höher eingestuft werden als die irrelevanten Knoten. Beachten Sie auch, dass die Abfragekontexte in der Reihenfolge ihrer Rangfolge angegeben werden.

**
WICHTIG: Bitte stellen Sie sicher, dass die Rückgabe nur im JSON-Format erfolgt, wobei der Schlüssel „reason“ den Grund angibt.
Beispiel JSON:
{{
 "reason": „Die Punktzahl ist <contextual_precision_score> weil <deine_begründung>.“
}}


Erwähnen Sie in Ihrer Begründung NICHT den Begriff „verdict“ (Urteil), sondern formulieren Sie ihn als irrelevanten Knotenpunkt. Der Begriff „verdict,“ dient nur dazu, dass Sie den weiteren Rahmen der Dinge verstehen.
Erwähnen Sie auch NICHT, dass es in den Abfragekontexten, die Ihnen vorgelegt werden, Felder „reason“ (Grund) gibt, sondern verwenden Sie nur die Informationen im Feld „reason“ (Grund).
In Ihrer Begründung MÜSSEN Sie den „Grund“, die ZITATE in der „Begründung“ und den Knoten RANK (beginnend bei 1, z. B. erster Knoten) verwenden, um zu erklären, warum die ‚Nein‘-Urteile niedriger eingestuft werden sollten als die „Ja“-Urteile.
Wenn Sie Knoten ansprechen, machen Sie deutlich, dass es sich um Knoten in Abfragekontexten handelt.
Wenn das Ergebnis 1 ist, fassen Sie sich kurz und sagen Sie etwas Positives in einem positiven Ton (aber übertreiben Sie es nicht, sonst wird es lästig).
**

Kontextbezogene Präzisionsbewertung:
{score}

Eingabe:
{input}

Abfragekontexte:
{verdicts}

JSON:
"""