import os
import requests
import json
import logging
import asyncio
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import tiktoken
from fastapi.middleware.cors import CORSMiddleware
from requests.exceptions import RequestException

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API KEY & CONFIGURATION ---
# The environment variables will be set in the Vercel dashboard.
# DO NOT hardcode your production API keys here.
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Check if environment variables are set
if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, GEMINI_API_KEY, AUTH_TOKEN, OPENAI_API_KEY]):
    raise ValueError("One or more required environment variables are not set.")

# --- INITIALIZE API CLIENTS ---
pc = Pinecone(api_key=PINECONE_API_KEY)
# Initialize OpenAI for embeddings
client = OpenAI(api_key=OPENAI_API_KEY)

# --- FASTAPI SETUP ---
app = FastAPI(title="HackRx RAG API")

# Add CORS middleware to allow requests from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- DOCUMENT PROCESSING FUNCTIONS ---

def download_document_from_url(url: str):
    """
    Downloads a document from a given URL and returns its content and file extension.
    """
    try:
        logger.info(f"Attempting to download document from URL: {url}")
        # Use a timeout to prevent the application from hanging
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        logger.info(f"Successfully downloaded document from URL: {url}")

        file_extension = os.path.splitext(url.lower().split('?')[0])[1]
        
        # Read content from the stream
        document_content = response.content
        
        return document_content, file_extension
    except requests.exceptions.Timeout:
        logger.error(f"Download request timed out for URL: {url}")
        raise HTTPException(status_code=408, detail=f"Download request timed out for URL: {url}")
    except RequestException as e:
        logger.error(f"Download request failed for URL: {url}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download document. Error: {e}")

def _extract_text_from_pdf(content: bytes) -> str:
    """
    Helper function to extract text from a PDF file.
    """
    import io
    from PyPDF2 import PdfReader
    try:
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF file.")

def _extract_text_from_docx(content: bytes) -> str:
    """
    Helper function to extract text from a DOCX file.
    """
    import io
    from docx import Document
    try:
        document = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in document.paragraphs])
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text from DOCX file.")

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
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file_extension}")

# --- VECTOR STORE FUNCTIONS (PINEOCNE) ---
async def create_pinecone_index_and_upsert(chunks: List[str]):
    """
    Creates a Pinecone index if it doesn't exist and upserts document chunks.
    """
    index_name = "hackrx-rag-index"
    if index_name not in pc.list_indexes().names:
        logger.info(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536, # OpenAI 'text-embedding-3-small' dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_ENVIRONMENT
            )
        )

    index = pc.Index(index_name)

    logger.info("Generating embeddings and upserting into Pinecone...")
    batch_size = 100 # Adjust batch size based on API limits and performance
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # Generate embeddings for the batch
        embeddings_response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        embeddings = [item.embedding for item in embeddings_response.data]

        # Prepare data for upsert
        upsert_data = [
            (str(i+j), embeddings[j], {"text": chunk})
            for j, chunk in enumerate(batch)
        ]
        
        # Upsert the batch
        index.upsert(vectors=upsert_data, namespace="policy")
        logger.info(f"Upserted batch {i//batch_size + 1}")
    logger.info("Upserting finished.")


async def semantic_search_pinecone(query_text: str, top_k: int = 3) -> List[str]:
    """
    Performs a semantic search on Pinecone to retrieve the most relevant chunks.
    """
    index_name = "hackrx-rag-index"
    if index_name not in pc.list_indexes().names:
        logger.error(f"Pinecone index '{index_name}' does not exist.")
        raise HTTPException(status_code=500, detail="Pinecone index not found. Please run the test pipeline first to create it.")
    
    index = pc.Index(index_name)
    
    # Generate an embedding for the query
    query_embedding_response = client.embeddings.create(
        input=query_text,
        model="text-embedding-3-small"
    )
    query_embedding = query_embedding_response.data[0].embedding
    
    # Query the Pinecone index
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="policy"
    )
    
    relevant_chunks = [match.metadata['text'] for match in search_results.matches]
    return relevant_chunks

# --- LLM FUNCTIONS ---
async def llm_parse_query(question: str) -> Dict[str, Any]:
    """
    Uses the Gemini API to parse a natural language question into a structured format.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""
    You are a query parsing assistant. Your task is to extract key entities and intent from a user's question.

    User question: {question}

    Extract the following information and present it as a JSON object:
    - "intent": The user's goal (e.g., "find_coverage", "check_waiting_period", "get_definition").
    - "entities": A list of key entities mentioned in the question (e.g., "pre-existing diseases", "maternity expenses", "grace period").

    Example for "What is the waiting period for pre-existing diseases?":
    {{
        "intent": "check_waiting_period",
        "entities": ["pre-existing diseases"]
    }}

    Return only the JSON object.
    """
    
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "intent": {"type": "STRING"},
                    "entities": {"type": "ARRAY", "items": {"type": "STRING"}}
                }
            }
        }
    }
    
    try:
        response = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if result.get("candidates") and result["candidates"][0].get("content"):
            # The JSON object is returned as a string in the 'text' field
            json_text = result["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(json_text)
        else:
            logger.error(f"LLM API response did not contain candidates: {result}")
            return {"error": "Failed to get a response from the LLM."}
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API call failed: {e}")
        return {"error": "LLM API call failed."}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {"error": "Failed to parse LLM response."}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return {"error": "An unexpected error occurred."}


async def llm_synthesize_answer(question: str, retrieved_context: List[str]) -> str:
    """
    Uses the Gemini API to generate a final answer based on the question and retrieved context.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
    
    context_text = "\n---\n".join(retrieved_context)
    prompt = f"""
    You are an expert policy assistant. Use the following context to answer the user's question.
    Do not use any outside knowledge. If the answer is not in the context, state that you cannot find the answer.
    The context is a snippet from a policy document.

    Context:
    {context_text}

    Question:
    {question}

    Answer:
    """
    
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }
    
    try:
        response = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if result.get("candidates") and result["candidates"][0].get("content"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            logger.error(f"LLM API response did not contain candidates: {result}")
            return "Failed to generate an answer from the LLM."
    except Exception as e:
        logger.error(f"Failed to synthesize answer: {e}")
        return f"An error occurred while generating the answer: {e}"


# --- Pydantic Models for API Request & Response ---

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# --- API ENDPOINT ---

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submission(request_data: RunRequest, request: Request):
    """
    Main endpoint to process a document, extract text, embed chunks, search for
    relevant information based on a list of questions, and synthesize answers.
    """
    # Authorization check
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    logger.info("--- Starting RAG pipeline for new request ---")
    
    try:
        # Step 1: Document Processing and Chunking
        document_text = process_document(request_data.documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(document_text)
        logger.info(f"Document processed and split into {len(chunks)} chunks.")
        
        # Step 2: Create Pinecone index and upsert chunks
        await create_pinecone_index_and_upsert(chunks)
        
        # Step 3: Process each question in parallel
        answers = []
        for question in request_data.questions:
            # Semantic search to find relevant chunks for the current question
            relevant_chunks = await semantic_search_pinecone(question, top_k=3)
            
            # Synthesize the final answer using the LLM and retrieved context
            final_answer = await llm_synthesize_answer(question, relevant_chunks)
            answers.append(final_answer)
            logger.info(f"Answer generated for question: '{question}'")
            
        logger.info("--- RAG pipeline finished successfully ---")
        return RunResponse(answers=answers)
    
    except HTTPException as e:
        logger.error(f"API Error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

