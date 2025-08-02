# api/index.py

# --- 1. Imports ---
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

# Import the new modules
from document_processor import process_document
from query_parser import llm_parse_query
from vector_store import create_pinecone_index_and_upsert, semantic_search_pinecone
from answer_generator import llm_synthesize_answer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Pydantic Models for API Request and Response ---
class HackathonRequest(BaseModel):
    """
    Data model for the incoming request payload.
    """
    documents: str = Field(..., description="URL to the document blob (e.g., PDF, DOCX)")
    questions: List[str] = Field(..., description="List of natural language questions to ask")

class HackathonResponse(BaseModel):
    """
    Data model for the outgoing JSON response.
    """
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")

# --- 3. FastAPI Application Setup ---
app = FastAPI(
    title="HackRX LLM-Powered Query-Retrieval API",
    description="API for HackRX submission, providing a mock implementation for testing.",
    version="1.0.0",
)

# --- 4. API Endpoint Definition ---
@app.post("/hackrx/run", tags=["HackRx API"], response_model=HackathonResponse)
async def run_submission(request_data: HackathonRequest, request: Request):
    """
    Processes a list of questions against a specified document URL.
    This endpoint simulates the full LLM-powered query-retrieval pipeline.
    """
    logger.info(f"Received request for documents: {request_data.documents}")
    logger.info(f"Received {len(request_data.questions)} questions.")

    # --- 4.1. Authentication Check ---
    auth_header = request.headers.get("Authorization")
    required_token = os.environ.get("AUTH_TOKEN")

    if not required_token or auth_header != required_token:
        logger.warning(f"Invalid Authorization token received: {auth_header}")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid Bearer token"
        )
    logger.info("Authorization token is valid.")

    # --- 4.2. Document Processing and Chunking ---
    try:
        logger.info(f"Downloading and processing document from URL: {request_data.documents}")
        extracted_text = process_document(request_data.documents)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(extracted_text)
        logger.info(f"Document split into {len(chunks)} chunks.")

    except HTTPException as e:
        logger.error(f"Failed during document processing: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during document processing: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    # --- 4.3. Embedding and Vector Store ---
    try:
        # Create or connect to the Pinecone index and upsert the chunks
        await create_pinecone_index_and_upsert(chunks)
        logger.info("Chunks successfully embedded and upserted to Pinecone.")
    except HTTPException as e:
        logger.error(f"Failed during vector store operations: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during vector store operations: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    # --- 4.4. Process Questions and Generate Answers ---
    answers = []
    for i, question in enumerate(request_data.questions):
        logger.info(f"Processing question {i+1}: '{question}'")
        try:
            # 1. Parse the query (optional but good for a full RAG pipeline)
            # parsed_query = await llm_parse_query(question)
            # logger.info(f"Parsed query: {parsed_query.get('intent', 'N/A')}")

            # 2. Perform semantic search to get relevant chunks
            relevant_chunks = await semantic_search_pinecone(question, top_k=3)
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")

            # 3. Generate the final answer using the LLM and the retrieved context
            final_answer = await llm_synthesize_answer(question, relevant_chunks)
            answers.append(final_answer)

        except HTTPException as e:
            logger.error(f"Failed to process question '{question}': {e.detail}")
            answers.append(f"Error processing question: {e.detail}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while answering question '{question}': {e}")
            answers.append("An unexpected error occurred while generating the answer.")

    # --- 4.5. Final Response Construction ---
    response_payload = HackathonResponse(answers=answers)
    logger.info("Successfully processed all questions. Returning response.")

    return JSONResponse(content=response_payload.dict())

# --- 5. Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)