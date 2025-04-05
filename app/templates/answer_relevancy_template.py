from typing import List


class AnswerRelevancyTemplate:
    @staticmethod
    def generate_statements(actual_output: str):
        return f"""Gliedern Sie den Text auf und erstellen Sie eine Liste der dargestellten Aussagen. Zweideutige Aussagen und einzelne Wörter können ebenfalls als Aussagen betrachtet werden.

Beispiel:
Beispieltext: 
Unser neues Notebook-Modell verfügt über ein hochauflösendes Retina-Display für eine kristallklare Darstellung. Außerdem verfügt es über einen Schnelllade-Akku, mit dem Sie mit einer einzigen Ladung bis zu 12 Stunden arbeiten können. Für die Sicherheit haben wir eine Fingerabdruck-Authentifizierung und eine verschlüsselte SSD eingebaut. Außerdem erhalten Sie bei jedem Kauf eine einjährige Garantie und einen 24/7-Kundensupport.
WICHTIG: Benutze den gleichen Schlüssel, wie unten vorgegeben "statements".

{{
 "statements": [
        "Das neue Laptop-Modell hat ein hochauflösendes Retina-Display.",
        "Es verfügt über einen Schnelllade-Akku mit bis zu 12 Stunden Betriebsdauer.",
        "Zu den Sicherheitsmerkmalen gehören Fingerabdruck-Authentifizierung und eine verschlüsselte SSD.",
        "Jeder Kauf wird mit einer einjährigen Garantie geliefert.",
        "24/7-Kundensupport ist inklusive."
    ]
}}
===== ENDE DES BEISPIELS ======
        
**
WICHTIG: Bitte stellen Sie sicher, dass die Rückgabe nur im JSON-Format erfolgt, wobei der Schlüssel "statements" auf eine Liste von Strings abgebildet wird. Es werden keine Worte oder Erklärungen benötigt.
**

Text:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_verdicts(input: str, statements: str):
        return f"""Bestimmen Sie für die bereitgestellte Liste von Aussagen, ob jede Aussage relevant ist, um die Eingabe zu bearbeiten.
Bitte erzeugen Sie eine JSON-Liste mit zwei Schlüsseln: „verdict“ und „reason“.
Der Schlüssel „verdict“ sollte STRIKT entweder „yes“, ‚idk‘ oder „no“ lauten. Antworten Sie mit „yes“, wenn die Aussage für die ursprüngliche Eingabe relevant ist, mit ‚no‘, wenn die Aussage irrelevant ist, und mit „idk“, wenn sie mehrdeutig ist (z. B. nicht direkt relevant ist, aber als Stützpunkt für die Eingabe verwendet werden könnte).
Der Schlüssel „reason“ ist der Grund für das Urteil.
Geben Sie den Schlüssel „reason“ NUR hinzu, wenn die Antwort „nein“ lautet im Schlüssel "verdict" . 
Die angegebenen Aussagen sind Aussagen, die in der aktuellen Ausgabe gemacht werden.

**
WICHTIG: Bitte stellen Sie sicher, dass die Rückgabe nur im JSON-Format erfolgt, wobei der Schlüssel „verdicts“ auf eine Liste von JSON-Objekten abgebildet wird.
Beispiel-Eingabe: 
Welche Funktionen hat der neue Laptop?

Beispielhafte Aussagen: 
[
 "Das neue Laptop-Modell hat ein hochauflösendes Retina-Display.",
 "Es verfügt über einen Schnelllade-Akku mit einer Betriebsdauer von bis zu 12 Stunden.",
 "Zu den Sicherheitsmerkmalen gehören Fingerabdruck-Authentifizierung und eine verschlüsselte SSD.",
 "Jeder Kauf wird mit einer einjährigen Garantie geliefert.",
 "Ein 24/7-Kundensupport ist inbegriffen.",
]

Beispiel JSON:
{{
 "verdicts": [
        {{
        "verdict": "yes"
        }},
        {{
        "verdict": "yes"
        }},
        {{
        "verdict": "yes"
        }},
        {{
        "verdict": "no",
        "reason": "Eine einjährige Garantie ist ein Kaufvorteil, keine Eigenschaft des Laptops selbst.“
        }},
        {{
        "verdict": "idk",
        "reason": "Der Kundensupport ist eine Dienstleistung und kein Merkmal des Laptops.“
        }}
    ]  
}}

Da für jede Aussage ein Urteil erstellt werden soll, MUSS die Anzahl der "verdicts“ also der Urteile EXAKT gleich der Anzahl der "statements" also der Aussage sein.
**    

Eingabe:
{input}

Aussagen:
{statements}

JSON:
"""

    @staticmethod
    def generate_reason(
        irrelevant_statements: List[str], input: str, score: float
    ):
        return f"""Geben Sie anhand des answer relevancy score, also der Relevanzbewertung der Antwort, der Liste der Gründe für irrelevante Aussagen in der eigentlichen Textausgabe, dem Output, und der Eingabe eine PRÄZISE Begründung für die Bewertung. Erläutern Sie, warum die Punktzahl nicht höher ist, aber auch, warum sie so ist, wie sie ist.
Die irrelevanten Aussagen stehen für Dinge in der eigentlichen Ausgabe, die für die Beantwortung der Frage/des Themas im Input irrelevant sind.
Wenn es nichts Irrelevantes gibt, sagen Sie einfach etwas Positives in einem aufmunternden, ermutigenden Ton (aber übertreiben Sie es nicht, sonst wird es lästig).


**
WICHTIG: Bitte stellen Sie sicher, dass Sie nur im JSON-Format zurückgeben, wobei der Schlüssel „reason“ den Grund angibt.
Beispiel JSON:
{{
 "reason": „Die Punktzahl ist <answer_relevancy_score> weil <deine_Begründung>.“
}}
**

Antwort Relevanz Score:
{score}

Gründe, warum die Punktzahl nicht höher sein kann, basierend auf irrelevanten Aussagen in der eigentlichen Ausgabe:
{irrelevant_statements}

Eingabe:
{input}

JSON:
"""