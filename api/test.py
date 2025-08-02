# main.py

import logging
import asyncio
from test_processor import run_local_test

# Configure logging to see all output from the test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting local test...")
    # Run the asynchronous test function
    asyncio.run(run_local_test())
    logger.info("Local test finished.")
