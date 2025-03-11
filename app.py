import logging
from pas2 import create_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create and launch the interface
logger.info("Starting PAS2 Hallucination Detector")
interface = create_interface()
logger.info("Launching Gradio interface...")

# This is the entry point for Hugging Face Spaces
app = interface 