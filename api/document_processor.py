# api/document_processor.py

import requests
import logging
from fastapi import HTTPException
import os
import io
# Libraries for document processing
from docx import Document
from PyPDF2 import PdfReader

# Set up logging for this module
logger = logging.getLogger(__name__)

def download_document_from_url(url: str):
    """
    Downloads a document from a given URL and returns its content and file extension.
    
    Args:
        url (str): The URL of the document to download.

    Returns:
        tuple[bytes, str]: A tuple containing the document content and its file extension.
        
    Raises:
        HTTPException: If the document cannot be downloaded.
    """
    try:
        logger.info(f"Attempting to download document from URL: {url}")
        # Use a timeout to prevent the application from hanging on a slow request
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        logger.info(f"Successfully downloaded document from URL: {url}")
        
        # Get the file extension from the URL
        file_extension = os.path.splitext(url.lower().split('?')[0])[1]
        
        return response.content, file_extension
    except requests.exceptions.Timeout:
        logger.error(f"Download request timed out for URL: {url}")
        raise HTTPException(status_code=408, detail=f"Request to download document timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download document from URL: {url}. Error: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download document from URL: {e}")

def _extract_text_from_pdf(content: bytes) -> str:
    """
    Helper function to extract text from a PDF file.
    """
    try:
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF. Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF file.")

def _extract_text_from_docx(content: bytes) -> str:
    """
    Helper function to extract text from a DOCX file.
    """
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
        logger.warning(f"Unsupported document type: {file_extension}")
        raise HTTPException(status_code=415, detail=f"Unsupported document type: {file_extension}")
