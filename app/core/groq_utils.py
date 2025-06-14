import re
import json
import base64
from typing import Dict, Any

def parse_groq_response(response: str) -> Dict[str, Any]:
    """Robust JSON extraction from LLM response"""
    try:
        # Remove markdown code blocks and whitespace
        clean = re.sub(r'``````', '', response).strip()
        
        # Find first valid JSON object
        json_match = re.search(r'{(.*?)}', clean, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
            
        # Fallback to parsing entire response
        return json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {str(e)}")
        print(f"Problematic response: {response[:200]}...")
    except Exception as e:
        print(f"Groq Response Error: {str(e)}")
    return {}
