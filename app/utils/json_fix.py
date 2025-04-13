import json
import re


def extract_json_from_text(text):
    """
    Attempts to extract a valid JSON object from text that might contain other content.
    """
    # Try to find JSON-like content between curly braces
    json_pattern = re.compile(r'\{.*\}', re.DOTALL)
    matches = json_pattern.findall(text)

    if not matches:
        return None

    # Try each match to see if any is valid JSON
    for match in matches:
        try:
            # Attempt to parse the match as JSON
            data = json.loads(match)
            return data
        except json.JSONDecodeError:
            # Try to fix common JSON formatting issues
            try:
                # Replace single quotes with double quotes
                fixed_json = match.replace("'", "\"")
                # Replace unquoted JSON keys with quoted keys
                fixed_json = re.sub(r'(\s*?)(\w+?)\s*?:', r'\1"\2":', fixed_json)
                data = json.loads(fixed_json)
                return data
            except:
                # If still not working, continue to the next match
                continue

    return None


def sanitize_llm_response(text):
    """
    Sanitizes an LLM response to extract valid JSON.
    """
    # First try to parse the text directly as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from the text
        extracted_json = extract_json_from_text(text)
        if extracted_json:
            return extracted_json

        # If all else fails, throw an error
        raise ValueError("Unable to extract valid JSON from the response")