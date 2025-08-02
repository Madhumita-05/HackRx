# api/answer_generator.py

import os
import json
import logging
import requests
import asyncio # Add asyncio for use with exponential backoff
from typing import List

# Set up logging for this module
logger = logging.getLogger(__name__)

async def llm_synthesize_answer(question: str, context_chunks: List[str]) -> str:
    """
    Uses a Large Language Model (LLM) to synthesize a final answer
    from the retrieved context chunks and the user's question.
    
    Args:
        question (str): The natural language question from the user.
        context_chunks (List[str]): A list of text chunks retrieved from the document.
        
    Returns:
        str: The final, synthesized answer.
    """
    # NOTE: The API key for Gemini will be automatically handled by the Canvas environment.
    # The prompt below is designed to perform a similar function as GPT-4 would for this task.
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyARwMZPD1eiqP-hWeWgPF8TtBfBIY6Rlmc")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    # We will use the non-streaming API
    headers = {
        "Content-Type": "application/json",
    }
    
    # Construct the prompt for the LLM
    # We combine the user's question with the retrieved context.
    context_str = "\n\n".join(context_chunks)
    prompt = f"""
    You are an expert at extracting information from a document to answer a user's question.
    Use ONLY the following context to answer the question. If the answer is not in the context,
    state that you cannot find the answer. Do not use any external knowledge.
    
    After providing the answer, provide a brief rationale explaining which part of the document
    you used, citing the specific sentences or clauses from the provided context.
    
    --- Context ---
    {context_str}
    
    --- Question ---
    {question}
    
    --- Answer ---
    """

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    try:
        # Use exponential backoff to handle rate limits
        for i in range(3): # Retry up to 3 times
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code == 429:
                delay = 2 ** i
                logger.warning(f"Rate limit hit. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                continue
            response.raise_for_status()
            break
        else:
            raise requests.exceptions.RequestException("Max retries exceeded for LLM API call.")

        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            final_answer = result["candidates"][0]["content"]["parts"][0]["text"]
            return final_answer
        else:
            logger.error(f"LLM API response did not contain candidates: {result}")
            return "An error occurred while generating the answer."
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API call failed: {e}")
        return "Failed to connect to the LLM API."
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred."
