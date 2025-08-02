# api/vector_store.py

import os
import requests
import json
import logging
import time
from typing import List, Dict, Any, Tuple
from fastapi import HTTPException
import asyncio
from pinecone import Pinecone, ServerlessSpec

# Set up logging for this module
logger = logging.getLogger(__name__)

# NOTE: The API keys for Gemini and Pinecone will be automatically handled.
# A placeholder is provided for local testing, but you should use environment variables.
# The `AIzaSy...` key below is an example. The actual key will be injected at runtime.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyARwMZPD1eiqP-hWeWgPF8TtBfBIY6Rlmc")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_5yevzJ_CSrC3APqKWWfs56gt5voArBiiehGZoVVX5pyPj7pC2LHkwgv4c4xVdHKHcpir6")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "hackrx-rag-index")
# This matches the embedding model's dimensionality
EMBEDDING_DIMENSIONS = 768

async def _generate_embedding_with_backoff(text: str, max_retries: int = 5) -> List[float]:
    """
    Generates a vector embedding with exponential backoff for retries.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
    
    payload = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": text}]
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()
            
            if result.get("embedding"):
                return result["embedding"]["values"]
            else:
                logger.error(f"Embedding API response did not contain an embedding: {result}")
                raise HTTPException(status_code=500, detail="Failed to get an embedding from the LLM.")
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                logger.warning(f"Embedding API call failed. Retrying in {sleep_time} seconds... Error: {e}")
                time.sleep(sleep_time)
            else:
                logger.error(f"Embedding API call failed after {max_retries} attempts: {e}")
                raise HTTPException(status_code=500, detail=f"LLM Embedding API call failed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during embedding generation: {e}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred.")


async def create_pinecone_index_and_upsert(chunks: List[str]):
    """
    Initializes a Pinecone index and upserts the embedded chunks.
    
    Args:
        chunks (List[str]): A list of text chunks.
        
    Raises:
        HTTPException: If Pinecone fails to initialize or upsert.
    """
    try:
        # Initialize Pinecone
        if not PINECONE_API_KEY:
            raise Exception("You haven't specified an API key. Please set the PINECONE_API_KEY environment variable.")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index already exists by trying to describe it
        try:
            pc.describe_index(PINECONE_INDEX_NAME)
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")
        except Exception:
            # If the index does not exist, create it.
            logger.info(f"Creating new Pinecone index: '{PINECONE_INDEX_NAME}'...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSIONS,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            # Wait for the index to be ready
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
            logger.info(f"Index '{PINECONE_INDEX_NAME}' created successfully.")

        # Connect to the index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Generate embeddings and prepare for upsert
        upsert_data = []
        for i, chunk in enumerate(chunks):
            embedding = await _generate_embedding_with_backoff(chunk)
            # Pinecone expects (id, vector, metadata)
            upsert_data.append((str(i), embedding, {"text": chunk}))
            
        logger.info(f"Upserting {len(upsert_data)} vectors to Pinecone...")
        index.upsert(vectors=upsert_data)
        logger.info(f"Successfully upserted {len(upsert_data)} vectors.")
        
    except Exception as e:
        logger.error(f"Failed to interact with Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Pinecone error: {e}")


async def semantic_search_pinecone(
    query: str,
    top_k: int = 3
) -> List[str]:
    """
    Performs a semantic search on a Pinecone index.
    
    Args:
        query (str): The user's natural language query.
        top_k (int): The number of top-matching chunks to retrieve.
        
    Returns:
        List[str]: A list of the most relevant text chunks.
        
    Raises:
        HTTPException: If Pinecone fails to query.
    """
    try:
        # Initialize Pinecone and connect to the index
        if not PINECONE_API_KEY:
            raise Exception("You haven't specified an API key. Please set the PINECONE_API_KEY environment variable.")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        # Generate the query embedding
        query_embedding = await _generate_embedding_with_backoff(query)

        # Query the Pinecone index
        search_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        relevant_chunks = [match['metadata']['text'] for match in search_results['matches']]
        return relevant_chunks
    
    except Exception as e:
        logger.error(f"Failed to query Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Pinecone query error: {e}")

if __name__ == "__main__":
    # --- Local Test Case ---
    # To run this test, simply execute: python -m api.vector_store
    
    async def run_test():
        logger.info("--- Starting local test for Pinecone vector store functionality ---")
        
        # A mock list of text chunks
        mock_chunks = [
            "A grace period of thirty days is provided for premium payment.",
            "Pre-existing diseases have a waiting period of 36 months.",
            "The policy covers maternity expenses after 24 months.",
            "This is an unrelated sentence about a different topic."
        ]
        
        # Create the Pinecone index and upsert the chunks
        await create_pinecone_index_and_upsert(mock_chunks)
        
        # Perform the semantic search
        test_question = "What is the waiting period for pre-existing diseases?"
        top_results = await semantic_search_pinecone(test_question, top_k=2)
        
        logger.info(f"\nTest Query: '{test_question}'")
        logger.info("Top 2 relevant chunks retrieved from Pinecone:")
        for i, chunk in enumerate(top_results):
            logger.info(f"Chunk {i+1}: {chunk}")
        
        logger.info("--- Local test finished ---")
    
    # This is needed to run an async function
    asyncio.run(run_test())

