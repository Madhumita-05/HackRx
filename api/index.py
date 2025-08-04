# api/index.py

import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict
from api.document_processor import process_document
from api.query_parser import llm_parse_query
from api.vector_store import create_pinecone_index_and_upsert, semantic_search_pinecone
from api.answer_generator import llm_synthesize_answer
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerWithRationale(BaseModel):
    answer: str
    rationale: str

class RunResponse(BaseModel):
    answers: List[AnswerWithRationale]

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(payload: RunRequest, request: Request):
    auth_header = request.headers.get("Authorization")
    required_token = os.environ.get("AUTH_TOKEN")
    if not required_token or auth_header != required_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        extracted_text = process_document(payload.documents)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
        chunks = splitter.split_text(extracted_text)
        await create_pinecone_index_and_upsert(chunks)
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    async def process_one_question(question: str):
        parsed = await llm_parse_query(question)
        if "error" in parsed:
            return {"answer": "Failed to parse question.", "rationale": ""}
        relevant_chunks = await semantic_search_pinecone(question, top_k=3)
        result = await llm_synthesize_answer(question, relevant_chunks)
        return result

    tasks = [process_one_question(q) for q in payload.questions]
    answers = await asyncio.gather(*tasks)

    return RunResponse(answers=answers)
