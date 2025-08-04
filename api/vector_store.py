# api/vector_store.py

import os
import logging
import time
import asyncio
import hashlib
from typing import List
from fastapi import HTTPException
import httpx
from pinecone import Pinecone, ServerlessSpec
from functools import lru_cache

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "hackrx-rag-index")
EMBEDDING_DIMENSIONS = 768

@lru_cache(maxsize=1024)
def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

async def _generate_embedding_with_backoff(text: str, max_retries: int = 5) -> List[float]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": text}]
        }
    }
    async with httpx.AsyncClient(timeout=20) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                embedding = result.get("embedding", {}).get("values")
                if embedding:
                    return embedding
                else:
                    logger.error(f"Embedding API response missing 'embedding': {result}")
                    raise HTTPException(status_code=500, detail="Failed to get embedding from LLM.")
            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    logger.warning(f"Embedding API error, retrying in {sleep_time}s: {e}")
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Embedding API failed after retries: {e}")
                    raise HTTPException(status_code=500, detail=f"LLM Embedding API call failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during embedding generation: {e}")
                raise HTTPException(status_code=500, detail="Unexpected embedding error.")

async def create_pinecone_index_and_upsert(chunks: List[str]):
    if not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY not set in environment.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    try:
        try:
            pc.describe_index(PINECONE_INDEX_NAME)
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' exists.")
        except Exception:
            logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSIONS,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
            logger.info(f"Index '{PINECONE_INDEX_NAME}' created.")

        index = pc.Index(PINECONE_INDEX_NAME)

        upsert_data = []
        for i, chunk in enumerate(chunks):
            cache_id = _cache_key(chunk)
            embedding = await _generate_embedding_with_backoff(chunk)
            upsert_data.append((cache_id, embedding, {"text": chunk, "chunk_id": i}))
        logger.info(f"Upserting {len(upsert_data)} vectors to Pinecone...")
        index.upsert(vectors=upsert_data)
        logger.info("Vector upsert successful.")
    except Exception as e:
        logger.error(f"Pinecone interaction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pinecone error: {e}")

async def semantic_search_pinecone(query: str, top_k: int = 3) -> List[str]:
    if not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY not set.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    query_embedding = await _generate_embedding_with_backoff(query)
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    relevant_chunks = [match['metadata']['text'] for match in search_results['matches']]
    return relevant_chunks
