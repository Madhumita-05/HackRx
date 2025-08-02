# api/test_processor.py

import logging
import os
import asyncio
import json
from document_processor import process_document
from query_parser import llm_parse_query
from vector_store import create_pinecone_index_and_upsert, semantic_search_pinecone
from answer_generator import llm_synthesize_answer # Import the answer generation function
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import HTTPException

# Configure logging to see the output from the document_processor
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A sample document URL to test with
TEST_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
RAW_OUTPUT_FILE_PATH = "extracted_raw_text.txt"
CHUNKS_OUTPUT_FILE_PATH = "extracted_chunks.txt"
TEST_QUESTION = "What were the main reasons behind Schaeffler's Q1 2021 sales growth and margin performance across divisions?"


async def run_local_test():
    """
    Runs a local test of the combined document processing, query parsing, embedding, and
    answer generation pipeline.
    """
    logger.info("--- Starting Local RAG Integration Test with Pinecone and LLM ---")
    
    # --- Step 1: Document Processing and Chunking ---
    try:
        logger.info(f"Downloading and processing document from URL: {TEST_DOCUMENT_URL}")
        extracted_text = process_document(TEST_DOCUMENT_URL)
        logger.info("Document text extracted successfully!")
        
        with open(RAW_OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        logger.info(f"Full raw text saved to '{RAW_OUTPUT_FILE_PATH}'.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(extracted_text)
        
        with open(CHUNKS_OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(chunk)
                f.write("\n\n")

        logger.info(f"Document split into {len(chunks)} chunks.")

    except HTTPException as e:
        logger.error(f"Test Failed during document processing! An HTTPException occurred: {e.detail}")
        return
    except Exception as e:
        logger.error(f"Test Failed during document processing! An unexpected error occurred: {e}")
        return
        
    logger.info("--- Document processing and chunking finished. ---")
    
    # --- Step 2: LLM Query Parsing ---
    try:
        logger.info(f"\nParsing the test question using LLM: '{TEST_QUESTION}'")
        parsed_query = await llm_parse_query(TEST_QUESTION)
        
        if "error" in parsed_query:
            logger.error(f"Test failed during query parsing: {parsed_query['error']}")
            return
        else:
            logger.info("Query parsed successfully! Result:")
            logger.info(json.dumps(parsed_query, indent=2))
            
    except Exception as e:
        logger.error(f"Test Failed during query parsing: {e}")
        return
    
    logger.info("--- Query parsing finished. ---")
    
    # --- Step 3: Embedding Search (Semantic Retrieval with Pinecone) ---
    try:
        # Create or connect to the Pinecone index and upsert the chunks
        await create_pinecone_index_and_upsert(chunks)
        
        # Perform the semantic search on Pinecone to get the most relevant chunks
        relevant_chunks = await semantic_search_pinecone(TEST_QUESTION, top_k=3)
        
        logger.info("\nSemantic search completed successfully! Retrieved relevant chunks from Pinecone:")
        for i, chunk in enumerate(relevant_chunks):
            logger.info(f"Relevant Chunk {i+1}:\n{chunk}\n{'='*50}")
            
    except HTTPException as e:
        logger.error(f"Test Failed during embedding search: {e.detail}")
        return
    except Exception as e:
        logger.error(f"Test Failed during embedding search: {e}")
        return
    
    logger.info("--- Embedding search finished. ---")
    
    # --- Step 4: Final Answer Generation ---
    try:
        logger.info("\nGenerating the final answer using the LLM and retrieved context...")
        final_answer = await llm_synthesize_answer(TEST_QUESTION, relevant_chunks)
        
        logger.info("\n--- Final Answer Generated ---")
        logger.info(final_answer)
        logger.info("--- End of Final Answer ---")
        
    except Exception as e:
        logger.error(f"Test Failed during answer generation: {e}")
        return
    
    logger.info("\n--- Local RAG Integration Test Finished Successfully ---")

# This is needed to run the async test function
if __name__ == "__main__":
    asyncio.run(run_local_test())
