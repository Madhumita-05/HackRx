# api/index.py

# --- 1. Imports ---
import os
import logging
from typing import List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the new pipeline components
from .document_processor import process_document
from .text_splitter import split_text_into_chunks
from .vector_store import create_pinecone_index_and_upsert, semantic_search_pinecone
from .query_parser import llm_parse_query
# Import the new answer generation function
from .answer_generator import llm_synthesize_answer

# Set up logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Pydantic Models for API Request and Response ---
class HackathonRequest(BaseModel):
    """
    Data model for the incoming request payload.
    - 'documents' is the URL to the document blob.
    - 'questions' is a list of natural language queries.
    """
    documents: str = Field(..., description="URL to the document blob (e.g., PDF, DOCX)")
    questions: List[str] = Field(..., description="List of natural language questions to ask")

class HackathonResponse(BaseModel):
    """
    Data model for the outgoing JSON response.
    - 'answers' is a list of strings, where each string is the
      answer corresponding to a question in the request.
    """
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")

# --- 3. FastAPI Application Setup ---
app = FastAPI(
    title="HackRX LLM-Powered Query-Retrieval API",
    description="API for HackRX submission, with a complete processing pipeline.",
    version="1.0.0",
)

# --- 4. API Endpoint Definition ---
@app.post("/hackrx/run", tags=["HackRx API"], response_model=HackathonResponse)
async def run_submission(request_data: HackathonRequest, request: Request):
    """
    Processes a list of questions against a specified document URL using a
    Retrieval-Augmented Generation (RAG) pipeline.
    """
    logger.info(f"Received request for documents: {request_data.documents}")
    logger.info(f"Received {len(request_data.questions)} questions.")

    # --- 4.1. Authentication Check ---
    auth_header = request.headers.get("Authorization")
    required_token = os.environ.get("AUTH_TOKEN")
    
    if not required_token or auth_header != required_token:
        logger.warning(f"Invalid Authorization token received: {auth_header}")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Bearer token")
    logger.info("Authorization token is valid.")

    # --- 4.2. Document Processing & Vector Store Creation ---
    try:
        # Step 1: Process the document and split it into chunks
        raw_text = process_document(request_data.documents)
        chunks = split_text_into_chunks(raw_text)
        logger.info(f"Document processed and split into {len(chunks)} chunks.")

        # Step 2: Create a Pinecone index and upsert the chunk embeddings
        # NOTE: This is a one-time setup step. In a production environment,
        # you would run this process asynchronously or as a separate job.
        await create_pinecone_index_and_upsert(chunks)
        
    except HTTPException as e:
        logger.error(f"Failed to process document or create vector store: {e.detail}")
        raise e
    
    # --- 4.3. Query Processing and Answer Generation ---
    answers = []
    for i, question in enumerate(request_data.questions):
        logger.info(f"Processing question {i+1}: '{question}'")

        try:
            # Step 3: Parse the user's question using an LLM
            parsed_query = await llm_parse_query(question)
            logger.info(f"Parsed question: {parsed_query}")
            
            # Step 4: Perform a semantic search to retrieve relevant chunks
            # We use the original question for the search to ensure all context is used.
            relevant_chunks = await semantic_search_pinecone(question)
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks from Pinecone.")

            # Step 5: Synthesize a final answer from the retrieved chunks and the original question
            answer = await llm_synthesize_answer(question, relevant_chunks)
            
            answers.append(answer)
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            answers.append(f"An error occurred while processing the question: {e}")

    # --- 4.4. Final Response Construction ---
    response_payload = HackathonResponse(answers=answers)
    logger.info("Successfully processed all questions. Returning response.")

    return JSONResponse(content=response_payload.dict())

# --- 5. Main Entry Point ---
# This part is for local development.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
