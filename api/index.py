# index.py

import os
import asyncio
import uuid
import logging
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import the core logic from the uploaded files
from .document_processor import process_document
from .query_parser import llm_parse_query  # <-- Added the missing import
from .vector_store import create_pinecone_index_and_upsert, semantic_search_pinecone, delete_pinecone_index
from .answer_generator import llm_synthesize_answer

# FastAPI application setup
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for Request and Response ---
class RunRequest(BaseModel):
    """
    Defines the structure of the incoming request body.
    """
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    """
    Defines the structure of the outgoing response body.
    """
    answers: List[str]

# --- Main API Endpoint ---
@app.post("/hackrx/run", response_model=RunResponse)
async def run_rag_pipeline(request: RunRequest):
    """
    This endpoint orchestrates the RAG pipeline.
    It takes a document URL and a list of questions, processes the document,
    performs a semantic search, and synthesizes answers for each question.
    """
    logger.info("Received a new request to run the RAG pipeline.")
    
    document_url = request.documents
    questions = request.questions
    
    # --- Step 1: Document Processing and Chunking ---
    try:
        logger.info(f"Downloading and processing document from URL: {document_url}")
        extracted_text = process_document(document_url)
        
        # Use a RecursiveCharacterTextSplitter to create text chunks
        # This matches the configuration from test_processor.py
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(extracted_text)
        logger.info(f"Document processed and split into {len(chunks)} chunks.")
        
    except HTTPException as e:
        logger.error(f"Document processing failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during document processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    # --- Step 2: Vectorization and Upsert to Pinecone ---
    index_name = f"rag-index-{uuid.uuid4()}"
    try:
        logger.info(f"Creating Pinecone index '{index_name}' and upserting chunks.")
        await create_pinecone_index_and_upsert(chunks, index_name)
        logger.info("Pinecone index created and chunks upserted successfully.")
    except HTTPException as e:
        logger.error(f"Pinecone operation failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred with Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to interact with Pinecone: {e}")

    # --- Step 3: Iterate through questions, perform search, and generate answers ---
    answers = []
    
    # Run the question processing in parallel for efficiency
    tasks = []
    for question in questions:
        async def process_question(q):
            try:
                # NOTE: As per your test_processor.py, we can call llm_parse_query
                # for diagnostic purposes, but its output is not used in the RAG
                # pipeline steps that follow.
                # parsed_query = await llm_parse_query(q)
                # logger.info(f"Parsed query for '{q}': {parsed_query}")

                # Perform a semantic search on Pinecone to get the most relevant chunks
                relevant_chunks = await semantic_search_pinecone(q, index_name, top_k=3)
                
                # Synthesize the final answer using the LLM and the retrieved context
                final_answer = await llm_synthesize_answer(q, relevant_chunks)
                return final_answer
            except Exception as e:
                logger.error(f"Failed to process question '{q}': {e}")
                return f"Failed to generate an answer for the question: '{q}'. An error occurred."

        tasks.append(process_question(question))
    
    answers = await asyncio.gather(*tasks)

    logger.info("All questions processed and answers generated.")
    
    # --- Step 4: Clean up the Pinecone index ---
    try:
        await delete_pinecone_index(index_name)
        logger.info(f"Pinecone index '{index_name}' deleted successfully.")
    except Exception as e:
        logger.warning(f"Failed to clean up Pinecone index '{index_name}': {e}")

    return RunResponse(answers=answers)
