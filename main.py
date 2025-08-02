# --- 1. Imports ---
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn
import logging

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Pydantic Models for API Request and Response ---
# These models ensure that the incoming and outgoing data
# strictly adhere to the required format.

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
# Create a FastAPI instance
app = FastAPI(
    title="HackRX LLM-Powered Query-Retrieval API",
    description="API for HackRX submission, providing a mock implementation for testing.",
    version="1.0.0",
)

# Define the base URL for local development
BASE_URL = "http://localhost:8000/api/v1"

# --- 4. Mock Data and Logic ---
# This is the "fake" logic requested by the user.
# In a real implementation, this would be replaced by
# LLM calls, vector database searches, and document processing.

# This dictionary holds the mock answers for the sample questions.
# The keys are the questions, and the values are the expected answers.
# This ensures that the output perfectly matches the example provided
# in the problem statement.
MOCK_ANSWERS_DB = {
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?":
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "What is the waiting period for pre-existing diseases (PED) to be covered?":
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    "Does this policy cover maternity expenses, and what are the conditions?":
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    "What is the waiting period for cataract surgery?":
        "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "Are the medical expenses for an organ donor covered under this policy?":
        "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
    "What is the No Claim Discount (NCD) offered in this policy?":
        "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
    "Is there a benefit for preventive health check-ups?":
        "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
    "How does the policy define a 'Hospital'?":
        "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
    "What is the extent of coverage for AYUSH treatments?":
        "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
    "Are there any sub-limits on room rent and ICU charges for Plan A?":
        "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN).",
}

# --- 5. API Endpoint Definition ---
@app.post("/hackrx/run", tags=["HackRx API"], response_model=HackathonResponse)
async def run_submission(request_data: HackathonRequest, request: Request):
    """
    Processes a list of questions against a specified document URL.
    This endpoint simulates the full LLM-powered query-retrieval pipeline.
    """
    logger.info(f"Received request for documents: {request_data.documents}")
    logger.info(f"Received {len(request_data.questions)} questions.")

    # --- 5.1. Authentication Check (Mocked) ---
    # The problem statement specifies a Bearer token. This is a simple
    # mock check to show where a real auth check would go.
    auth_header = request.headers.get("Authorization")
    required_token = "Bearer 2b85f37b231edd800c4ade1d01fb1745ecc3bf24a453c10ebe81ade8ad8a1928"
    if auth_header != required_token:
        logger.warning(f"Invalid Authorization token received: {auth_header}")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid Bearer token"
        )
    logger.info("Authorization token is valid.")

    # --- 5.2. Placeholder Workflow ---
    # Simulate the pipeline steps mentioned in the problem statement.
    # We will use the mock data to return the correct answers.

    answers = []
    for i, question in enumerate(request_data.questions):
        logger.info(f"Processing question {i+1}: '{question}'")

        # LLM Parser (Mocked): Simulate parsing the question.
        # In a real app, this would extract keywords or intent.
        
        # Embedding Search & Clause Matching (Mocked):
        # We're just looking up the question in our mock database.
        answer = MOCK_ANSWERS_DB.get(question, "Answer not found in the mock data.")

        # Logic Evaluation & JSON Output (Mocked):
        # The mock answer is already in the final format.
        answers.append(answer)

    # --- 5.3. Final Response Construction ---
    # Create the final JSON response based on the generated answers.
    response_payload = HackathonResponse(answers=answers)
    logger.info("Successfully processed all questions. Returning response.")

    return JSONResponse(content=response_payload.dict())

# --- 6. Main Entry Point ---
# This part is needed to run the application using `uvicorn`.
# It's not part of the API endpoint itself.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn main:app --reload --host 0.0.0.0 --port 8000