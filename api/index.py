# api/index.py

import logging
import os
import asyncio
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from api.document_processor import process_document
from api.query_parser import llm_parse_query
from api.vector_store import create_pinecone_index_and_upsert, semantic_search_pinecone
from api.answer_generator import llm_synthesize_answer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Request body model
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

# Response body model
class RunResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(payload: RunRequest):
    logger.info("--- Starting Remote RAG Integration Test with Pinecone and LLM ---")

    try:
        logger.info(f"Downloading and processing document from URL: {payload.documents}")
        extracted_text = process_document(payload.documents)
        logger.info("Document text extracted successfully!")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(extracted_text)

        logger.info(f"Document split into {len(chunks)} chunks.")

    except HTTPException as e:
        logger.error(f"Error during document processing: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during document processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        await create_pinecone_index_and_upsert(chunks)
        answers = []

        for question in payload.questions:
            logger.info(f"\nParsing the question using LLM: '{question}'")
            parsed_query = await llm_parse_query(question)

            if "error" in parsed_query:
                logger.error(f"Query parsing failed: {parsed_query['error']}")
                answers.append("Query parsing failed.")
                continue

            relevant_chunks = await semantic_search_pinecone(question, top_k=3)

            logger.info("Generating answer...")
            final_answer = await llm_synthesize_answer(question, relevant_chunks)
            answers.append(final_answer.strip())

    except Exception as e:
        logger.error(f"Error during semantic search or answer generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("--- Remote RAG Test Finished ---")
    return RunResponse(answers=answers)
