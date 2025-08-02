# api/query_parser.py

import os
import requests
import json
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOTE: The API key for Gemini will be automatically handled by the Canvas environment.
# When running this locally, you would need to set an environment variable:
# export GEMINI_API_KEY="your_api_key_here"

async def llm_parse_query(question: str) -> Dict[str, Any]:
    """
    Uses the Gemini API to parse a natural language question into a structured format.
    
    Args:
        question (str): The natural language question from the user.
        
    Returns:
        Dict[str, Any]: A dictionary containing the parsed query, intent, and entities.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyARwMZPD1eiqP-hWeWgPF8TtBfBIY6Rlmc")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    prompt = f"""
    You are a query parsing assistant. Your task is to extract key entities and intent from a user's question.
    
    User question: {question}
    
    Extract the following information and present it as a JSON object:
    - "query": The original question.
    - "intent": The purpose of the question (e.g., "check coverage", "find definition", "check waiting period").
    - "entities": A list of key entities or keywords mentioned in the question (e.g., ["maternity", "premium payment", "grace period"]).
    """
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING"},
                    "intent": {"type": "STRING"},
                    "entities": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    }
                }
            }
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        
        if result.get("candidates"):
            json_text = result["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(json_text)
        else:
            logger.error(f"LLM API response did not contain candidates: {result}")
            return {"error": "Failed to get a response from the LLM."}
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API call failed: {e}")
        return {"error": "LLM API call failed."}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {"error": "Failed to parse LLM response."}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return {"error": "An unexpected error occurred."}

if __name__ == "__main__":
    # --- Local Test Case ---
    # To run this test, simply execute: python -m api.query_parser
    import asyncio

    async def run_test():
        test_question = "What were the main reasons behind Schaeffler's Q1 2021 sales growth and margin performance across divisions?"
        logger.info(f"--- Starting local test for query parsing ---")
        logger.info(f"Test Question: '{test_question}'")
        
        parsed_query = await llm_parse_query(test_question)
        
        if "error" in parsed_query:
            logger.error(f"Test failed with error: {parsed_query['error']}")
        else:
            logger.info("Test Succeeded! Parsed Query:")
            logger.info(json.dumps(parsed_query, indent=2))
        
        logger.info("--- Local test finished ---")
        
    asyncio.run(run_test())

