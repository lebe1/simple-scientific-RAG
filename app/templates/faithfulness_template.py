from typing import Optional, List


class FaithfulnessTemplate:
    @staticmethod
    def generate_claims(actual_output: str):
        return f"""Erstellen Sie auf der Grundlage des angegebenen Textes eine umfassende Liste von TATSACHEN, unbestrittenen Wahrheiten, die sich aus dem angegebenen Text ableiten lassen.
Diese Wahrheiten MÜSSEN ZUSAMMENHÄNGEND sein und dürfen NICHT aus dem Zusammenhang gerissen werden.
    
Beispiel:
Beispieltext: 
"Albert Einstein, das Genie, das oft mit wilden Haaren und verrückten Theorien in Verbindung gebracht wird, erhielt den Nobelpreis für Physik - allerdings nicht für seine bahnbrechenden Arbeiten zur Relativitätstheorie, wie viele annehmen. Stattdessen wurde er 1968 für seine Entdeckung des photoelektrischen Effekts geehrt, ein Phänomen, das die Grundlage für die Quantenmechanik bildete."

Beispiel JSON: 
{{
 "claims": [
        "Einstein erhielt 1968 den Nobelpreis für seine Entdeckung des photoelektrischen Effekts."
        "Der photoelektrische Effekt ist ein Phänomen, das den Grundstein für die Quantenmechanik legte.“
    ]  
}}
===== ENDE DES BEISPIELS ======

**
WICHTIG: Bitte stellen Sie sicher, dass die Rückgabe nur im JSON-Format erfolgt, mit dem Schlüssel "claims" als Liste von Strings. Es werden keine Worte oder Erklärungen benötigt.
Fügen Sie nur Behauptungen ein, die sachlich sind, ABER ES SPIELT KEINE ROLLE, OB SIE FAKTISCH RICHTIG SIND. Die Behauptungen, die Sie extrahieren, sollten den vollständigen Kontext enthalten, in dem sie präsentiert wurden, und KEINE zusammengewürfelten Fakten.
Sie sollten KEIN Vorwissen einbeziehen und den Text für bare Münze nehmen, wenn Sie Behauptungen extrahieren.
**

Text:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_truths(
        retrieval_context: str, extraction_limit: Optional[int] = None
    ):
        if extraction_limit is None:
            limit = " FACTUAL, undisputed truths"
        elif extraction_limit == 1:
            limit = " the single most important FACTUAL, undisputed truth"
        else:
            limit = f" the {extraction_limit} most important FACTUAL, undisputed truths per document"
        return f"""Erstellen Sie bitte auf der Grundlage des gegebenen Textes eine umfassende Liste von {limit} (den Grenzen), die sich aus dem gegebenen Text ableiten lassen.
Diese Wahrheiten MÜSSEN ZUSAMMENHÄNGEND sein. Sie dürfen NICHT aus dem Zusammenhang gerissen werden.
        
Beispiel:
Beispieltext: 
"Albert Einstein, das Genie, das oft mit wilden Haaren und verrückten Theorien in Verbindung gebracht wird, erhielt den Nobelpreis für Physik - allerdings nicht für seine bahnbrechenden Arbeiten zur Relativitätstheorie, wie viele annehmen. Stattdessen wurde er 1968 für seine Entdeckung des photoelektrischen Effekts geehrt, ein Phänomen, das die Grundlage für die Quantenmechanik bildete."

Beispiel JSON: 
{{
 "truths": [
        "Einstein erhielt 1968 den Nobelpreis für seine Entdeckung des photoelektrischen Effekts."
        "Der photoelektrische Effekt ist ein Phänomen, das den Grundstein für die Quantenmechanik gelegt hat."
    ]  
}}
===== ENDE DES BEISPIELS ======
**
WICHTIG: Bitte stellen Sie sicher, dass die Rückgabe nur im JSON-Format erfolgt, mit dem Schlüssel "truths" als Liste von Strings. Es werden keine Worte oder Erklärungen benötigt.
Fügen Sie nur Wahrheiten ein, die faktisch sind, ABER ES SPIELT KEINE ROLLE, OB SIE FAKTISCH RICHTIG SIND.
**

Text:
{retrieval_context}

JSON:
"""

    @staticmethod
    def generate_verdicts(claims: List[str], retrieval_context: str):
        return f"""Auf der Grundlage der gegebenen Behauptungen, d.h. einer Liste von Zeichenketten, wird eine Liste von JSON-Objekten erstellt, um anzugeben, ob JEDE Behauptung den Fakten im Abfragekontext widerspricht. Das JSON hat 2 Felder: „verdict“ und „reason“.
Der Schlüssel ‚verdict‘ sollte EXAKT entweder ‚yes‘, ‚no‘ oder ‚idk‘ sein, was angibt, ob die gegebene Behauptung mit dem Kontext übereinstimmt. 
Geben Sie NUR dann ein Feld mit dem Schlüssel "reason" an, wenn die Antwort "no" lautet. 
Die angegebene Behauptung stammt aus der tatsächlichen Ausgabe. Versuchen Sie, in der Begründung eine Korrektur anhand der Fakten im Abfragekontext vorzunehmen.

**
WICHTIG: Bitte stellen Sie sicher, dass die Rückgabe nur im JSON-Format erfolgt, wobei der Schlüssel „verdicts“ eine Liste von JSON-Objekten ist!!!
Beispiel für Abfragekontexte: "Einstein erhielt den Nobelpreis für seine Entdeckung des photoelektrischen Effekts. Einstein erhielt den Nobelpreis im Jahr 1968. Einstein ist ein deutscher Wissenschaftler."
Beispiel für Behauptungen: [„Barack Obama ist ein kaukasischer Mann.“, „Zürich ist eine Stadt in London“, „Einstein gewann den Nobelpreis für die Entdeckung des photoelektrischen Effekts, was zu seinem Ruhm beigetragen haben könnte.“, „Einstein gewann den Nobelpreis 1969 für seine Entdeckung des photoelektrischen Effekts.“, „Einstein war ein deutscher Koch.“]

Beispiel:
{{
 "verdicts": [
        {{
        "verdict": "idk"
        }},
        {{
        "verdict": "idk"
        }},
        {{
        "verdict": "yes"
        }},
        {{
        "verdict": "no",
        "reason": „Die aktuelle Ausgabe behauptet, Einstein habe den Nobelpreis 1969 erhalten, was nicht stimmt, da der Abfragekontext stattdessen 1968 angibt.“
        }},
        {{
        "verdict": "no",
        "reason": „Die aktuelle Ausgabe behauptet, Einstein sei ein deutscher Koch, was nicht korrekt ist, da der Abfragekontext stattdessen angibt, er sei ein deutscher Wissenschaftler.“
        }},
    ]  
}}
===== ENDE DES BEISPIELS ======

Die Länge von "verdicts" SOLLTE EXAKT GLEICH zu der von "claims" sein.
Du musst KEINE Begründung angeben, wenn die Antwort "yes" oder „idk“ lautet.
Geben Sie NUR dann eine „no“-Antwort, wenn der Abfragekontext den Ansprüchen DIREKT WIDERSPRICHT. SIE SOLLTEN NIEMALS IHR VORWISSEN IN IHR URTEIL EINFLIESSEN LASSEN.
Behauptungen, die unter Verwendung von vagen, suggestiven, spekulativen Formulierungen wie „könnte sein“, „Möglichkeit aufgrund von“ gemacht werden, gelten NICHT als Widerspruch.
Behauptungen, die aufgrund mangelnder Informationen nicht belegt sind/im Kontext der Suche nicht erwähnt werden, MÜSSEN mit „idk“ beantwortet werden, sonst werde ich sterben.
**

Abfragekontext (retrieval context):
{retrieval_context}

Behauptungen (claims):
{claims}

JSON:
"""

    @staticmethod
    def generate_reason(score: float, contradictions: List[str]):
        return f"""Below is a list of Contradictions. It is a list of strings explaining why the 'actual output' does not align with the information presented in the 'retrieval context'. Contradictions happen in the 'actual output', NOT the 'retrieval context'.
Given the faithfulness score, which is a 0-1 score indicating how faithful the `actual output` is to the retrieval context (higher the better), CONCISELY summarize the contradictions to justify the score. 

** 
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <faithfulness_score> because <your_reason>."
}}

If there are no contradictions, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
Your reason MUST use information in `contradiction` in your reason.
Be sure in your reason, as if you know what the actual output is from the contradictions.
**

Unten ist eine Liste von Widersprüchen. Es handelt sich um eine Liste von Texten, die erklären, warum die "tatsächliche Ausgabe" (actual output) nicht mit den Informationen im "Abfragekontext" (retrieval context) übereinstimmt. Widersprüche treten in der "tatsächlichen Ausgabe" (actual output) auf, NICHT im "Abrufkontext" (retrieval context).

Angesichts der Genauigkeitsbewertung - einem Wert zwischen 0 und 1, der angibt, wie genau die "tatsächliche Ausgabe" dem Abfragekontext entspricht (je höher, desto besser) - fasse die Widersprüche KURZ zusammen, um die Bewertung zu begründen.

**
WICHTIG: Bitte stelle sicher, dass die Antwort ausschließlich im JSON-Format erfolgt, wobei der Schlüssel "reason" die Begründung enthält!!!
Beispiel-JSON:
{{
    "reason": "Die Bewertung beträgt <faithfulness_score>, weil <deine_Begründung>."
}}

Falls es keine Widersprüche gibt, schreibe etwas Positives in einem motivierenden Ton (aber nicht übertrieben, sonst wird es nervig).
Deine Begründung MUSS Informationen aus "contradiction" verwenden.
Sei dir sicher in deiner Begründung, als ob du die tatsächliche Ausgabe aus den Widersprüchen kennen würdest.
**

Treue Score (Faithfulness Score):
{score}

Widersprüche (Contradictions):
{contradictions}

JSON:
"""