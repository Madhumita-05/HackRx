# api/answer_generator.py

import os
import json
import logging
import asyncio
from typing import List, Dict
import httpx

logger = logging.getLogger(__name__)

async def llm_synthesize_answer(question: str, context_chunks: List[str]) -> Dict[str, str]:
    """
    Uses a Large Language Model (LLM) to synthesize a final answer
    from the retrieved context chunks and the user's question.
    
    Args:
        question (str): The natural language question from the user.
        context_chunks (List[str]): A list of text chunks retrieved from the document.
        
    Returns:
        Dict[str, str]: A dictionary with 'answer' and 'rationale' keys.
    """
    api_key = os.environ.get("GEMINI_API_KEY", None)
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        return {"answer": "Internal error: LLM API key not configured.", "rationale": ""}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json",
    }
    
    context_str = "\n\n".join(context_chunks)
    prompt = f"""
You are an expert extracting relevant information strictly from the provided document context.
Use ONLY the following context to answer the question. If the answer is not in the context,
say you cannot find the answer.

After the answer, provide a brief rationale citing the exact text segments you used.

--- Context ---
{context_str}

--- Question ---
{question}

--- Answer and Rationale ---
"""

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }

    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(3):
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 429:
                delay = 2 ** i
                logger.warning(f"Rate limit hit. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                continue
            response.raise_for_status()
            break
        else:
            logger.error("Max retries exceeded for LLM API call.")
            return {"answer": "Failed to generate answer due to rate limiting.", "rationale": ""}

        result = response.json()
        candidates = result.get("candidates", [])
        if candidates:
            # Grab first candidate text
            full_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            # Ideally, your prompt formats answer and rationale separated clearly. Parse them:
            # Simple split example: split at "Rationale:" or "---"
            if "rationale" in full_text.lower():
                parts = full_text.split("\n\n")
                # Assume first para is answer, last para with key "rationale" is rationale
                answer = parts[0].strip()
                rationale = "\n".join(parts[1:]).strip()
            else:
                answer = full_text.strip()
                rationale = ""
            return {"answer": answer, "rationale": rationale}
        else:
            logger.error(f"LLM response did not contain candidates: {result}")
            return {"answer": "An error occurred generating the answer.", "rationale": ""}
