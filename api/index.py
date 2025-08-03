# main.py

import os
import requests
import json
import logging
import asyncio
import time
import io
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from docx import Document
from PyPDF2 import PdfReader

# --- Configuration and Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOTE: The API keys for Gemini and Pinecone will be automatically handled by the Canvas environment.
# Placeholders are provided for local testing.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "hackrx-rag-index")
EMBEDDING_DIMENSIONS = 768

# --- API Initialization ---
app = FastAPI()

# --- Pydantic Model for Request Body ---
class QuestionRequest(BaseModel):
    """
    Defines the expected structure of the JSON payload for the /ask endpoint.
    """
    document_url: str
    question: str

# --- User-Provided Functions (Integrated and slightly refactored) ---

def download_document_from_url(url: str):
    """
    Downloads a document from a given URL and returns its content and file extension.
    """
    try:
        logger.info(f"Attempting to download document from URL: {url}")
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        logger.info(f"Successfully downloaded document from URL: {url}")
        
        file_extension = os.path.splitext(url.lower().split('?')[0])[1]
        
        return response.content, file_extension
    except requests.exceptions.Timeout:
        logger.error(f"Download request timed out for URL: {url}")
        raise HTTPException(status_code=408, detail="Document download request timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download document from URL: {url}. Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to download the document.")

def _extract_text_from_pdf(content: bytes) -> str:
    """Helper function to extract text from a PDF file."""
    try:
        pdf_reader = PdfReader(io.BytesIO(content))
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF. Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF file.")

def _extract_text_from_docx(content: bytes) -> str:
    """Helper function to extract text from a DOCX file."""
    try:
        document = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in document.paragraphs])
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX. Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract text from DOCX file.")

def process_document(url: str) -> str:
    """
    Main function to download a document from a URL and extract its text.
    It handles different file types based on the file extension.
    """
    document_content, file_extension = download_document_from_url(url)
    
    if file_extension == '.pdf':
        return _extract_text_from_pdf(document_content)
    elif file_extension == '.docx':
        return _extract_text_from_docx(document_content)
    else:
        logger.error(f"Unsupported file type: {file_extension}")
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}. Only .pdf and .docx are supported.")

async def _generate_embedding_with_backoff(text: str, max_retries: int = 5) -> List[float]:
    """
    Generates a vector embedding with exponential backoff.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"model": {"name": "models/text-embedding-004"}, "content": {"parts": [{"text": text}]}}

    for i in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code == 429:
                delay = 2 ** i
                logger.warning(f"Rate limit hit. Retrying embedding generation in {delay} seconds...")
                time.sleep(delay)
                continue
            response.raise_for_status()
            embedding = response.json()["embedding"]["values"]
            return embedding
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding API call failed on retry {i+1}: {e}")
            if i == max_retries - 1:
                raise
    return []

async def create_pinecone_index_and_upsert(chunks: List[str]):
    """
    Creates a Pinecone index if it doesn't exist and upserts vector embeddings for the chunks.
    """
    try:
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_INDEX_NAME not in pinecone.list_indexes().names:
            logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
            pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSIONS,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )
            while not pinecone.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
        
        index = pinecone.Index(PINECONE_INDEX_NAME)
        
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            embedding = await _generate_embedding_with_backoff(chunk)
            vectors_to_upsert.append((f"chunk-{i}", embedding, {"text": chunk}))
            
        index.upsert(vectors=vectors_to_upsert)
        logger.info(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone.")

    except Exception as e:
        logger.error(f"Failed to create/upsert to Pinecone index: {e}")
        raise HTTPException(status_code=500, detail=f"Pinecone operation error: {e}")

async def semantic_search_pinecone(query: str, top_k: int = 3) -> List[str]:
    """
    Performs a semantic search on the Pinecone index to find relevant chunks.
    """
    try:
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        index = pinecone.Index(PINECONE_INDEX_NAME)
        
        query_embedding = await _generate_embedding_with_backoff(query)
        
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        relevant_chunks = [match['metadata']['text'] for match in response['matches']]
        return relevant_chunks
    except Exception as e:
        logger.error(f"Failed to query Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Pinecone query error: {e}")

async def llm_synthesize_answer(question: str, context_chunks: List[str]) -> str:
    """
    Uses an LLM to synthesize a final answer from the retrieved context chunks.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    context_text = "\n\n".join(context_chunks)
    
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the user's question.
    If the answer is not in the context, state that you cannot find the answer and do not
    make up any information.
    
    Context:
    {context_text}
    
    Question: {question}
    
    Answer:
    """
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
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

# --- The Main RAG Pipeline Endpoint ---
@app.post("/ask")
async def ask_document(request: QuestionRequest):
    """
    A unified endpoint that takes a document URL and a question,
    then processes the document and generates an answer using a RAG pipeline.
    """
    logger.info(f"Received request for URL: {request.document_url} with question: '{request.question}'")
    
    try:
        # Step 1: Process the document and get raw text
        raw_text = process_document(request.document_url)
        
        # Step 2: Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(raw_text)
        
        # Step 3: Create Pinecone index and upsert embeddings
        # NOTE: This is a synchronous blocking call in this example. For a real-world
        # application, you might want to perform this step offline or on a schedule.
        # For simplicity, we are doing it here.
        await create_pinecone_index_and_upsert(chunks)
        
        # Step 4: Perform semantic search to get relevant chunks
        relevant_chunks = await semantic_search_pinecone(request.question, top_k=3)
        
        # Step 5: Synthesize the final answer with the LLM
        final_answer = await llm_synthesize_answer(request.question, relevant_chunks)
        
        return {"answer": final_answer}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during the RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- Requirements.txt file content ---
# To run this code, you'll need the following libraries.
# You can install them with `pip install -r requirements.txt`.
#
# FastAPI
# uvicorn[standard]
# python-dotenv
# requests
# docx
# pypdf2
# pinecone-client
# langchain
#
# Note: The `langchain` dependency is needed for the `RecursiveCharacterTextSplitter`.
# The versions of these libraries should be compatible. A good starting point is:
# fastapi==0.111.0
# uvicorn==0.29.0
# python-dotenv==1.0.1
# requests==2.31.0
# python-docx==1.1.0
# pypdf2==3.0.1
# pinecone-client==4.1.0
# langchain==0.2.11
