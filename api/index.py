# api/index.py

# --- 1. Imports ---
import os  # Import the 'os' module to access environment variables
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

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

# --- 4. Mock Data and Logic ---
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

    # --- 5.1. Authentication Check (Updated to use environment variable) ---
    auth_header = request.headers.get("Authorization")
    # Fetch the token from the environment variable
    required_token = os.environ.get("AUTH_TOKEN")
    
    if not required_token or auth_header != required_token:
        logger.warning(f"Invalid Authorization token received: {auth_header}")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid Bearer token"
        )
    logger.info("Authorization token is valid.")

    # --- 5.2. Placeholder Workflow ---
    answers = []
    for i, question in enumerate(request_data.questions):
        logger.info(f"Processing question {i+1}: '{question}'")
        answer = MOCK_ANSWERS_DB.get(question, "Answer not found in the mock data.")
        answers.append(answer)

    # --- 5.3. Final Response Construction ---
    response_payload = HackathonResponse(answers=answers)
    logger.info("Successfully processed all questions. Returning response.")

    return JSONResponse(content=response_payload.dict())

# --- 6. Main Entry Point ---
# This part is for local development and is ignored by Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
