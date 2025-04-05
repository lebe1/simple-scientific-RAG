from typing import List


class ContextualRelevancyTemplate:
    @staticmethod
    def generate_reason(
        input: str,
        irrelevant_statements: List[str],
        relevant_statements: List[str],
        score: float,
    ):
        return f"""Basierend auf dem gegebenen Input, den Gründen dafür, warum der Abfragekontext irrelevant für den Input ist, den Aussagen im Abfragekontext, die tatsächlich relevant sind, und dem Kontextrelevanz-Score (je näher an 1, desto besser), erstelle bitte einen **KURZEN** Grund für den Score.
In deiner Begründung solltest du Daten zitieren, die in den Gründen für Irrelevanz und in den relevanten Aussagen genannt werden, um deinen Punkt zu untermauern.

**
WICHTIG: Bitte stelle sicher, dass du nur im JSON-Format antwortest, wobei der Schlüssel 'reason' die Begründung enthält.
Beispiel-JSON:
{{
    "reason": "Der Score beträgt <contextual_relevancy_score>, weil <deine_Begründung>."
}}

Wenn der Score 1 beträgt, halte es kurz und sag etwas Positives mit einem motivierenden, freundlichen Ton (aber übertreibe es nicht, sonst wirkt es nervig).
**

Kontextrelevanz-Score:
{score}

Input:
{input}

Gründe dafür, warum der Abfragekontext irrelevant für den Input ist:
{irrelevant_statements}

Aussagen im Abfragekontext, die relevant für den Input sind:
{relevant_statements}

JSON:
"""

    @staticmethod
    def generate_verdicts(input: str, context: str):
        return f"""Basierend auf dem Input und dem Kontext, erstelle bitte ein JSON-Objekt, das angibt, ob jede im Kontext gefundene Aussage relevant für den gegebenen Input ist. Das JSON ist eine Liste von 'verdicts', mit zwei verpflichtenden Feldern: 'verdict' und 'statement', sowie einem optionalen Feld: 'reason'.
Du solltest zunächst die im Kontext enthaltenen Aussagen extrahieren, also die wesentlichen Informationen im Text, bevor du für jede Aussage ein Urteil sowie ggf. eine Begründung gibst.
Der Schlüssel 'verdict' MUSS ENTWEDER 'yes' ODER 'no' sein und gibt an, ob die Aussage relevant zum Input ist.
Gib eine 'reason' NUR AN, WENN das Urteil 'no' lautet. Du MUSST die irrelevanten Teile der Aussage zitieren, um deine Begründung zu stützen.

**
WICHTIG: Bitte stelle sicher, dass du nur im JSON-Format antwortest.
Beispiel-Kontext: "Einstein erhielt den Nobelpreis für seine Entdeckung des photoelektrischen Effekts. Er gewann den Nobelpreis 1968. Es gab eine Katze."
Beispiel-Input: "Was waren einige von Einsteins Errungenschaften?"

Beispiel:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "statement": "Einstein erhielt den Nobelpreis für seine Entdeckung des photoelektrischen Effekts im Jahr 1968"
        }},
        {{
            "verdict": "no",
            "statement": "Es gab eine Katze.",
            "reason": "Der Abfragekontext enthielt die Information 'Es gab eine Katze', die nichts mit Einsteins Errungenschaften zu tun hat."
        }}
    ]
}}
**

Input:
{input}

Kontext:
{context}

JSON:
"""
