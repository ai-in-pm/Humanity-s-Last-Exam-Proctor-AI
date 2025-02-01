from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EMERGENCEAI_API_KEY = os.getenv("EMERGENCEAI_API_KEY")

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
QUESTIONS_DIR = DATA_DIR / "questions"

# Evaluation settings
MIN_CONFIDENCE_THRESHOLD = 0.5
MAX_QUESTIONS_PER_SESSION = 3000
CONFIDENCE_CALIBRATION_BINS = 10

# Question types
QUESTION_TYPES = {
    "multiple_choice": "multiple_choice",
    "short_answer": "short_answer",
    "image_based": "image_based"
}

# Subject areas
SUBJECT_AREAS = [
    "mathematics",
    "science",
    "humanities",
    "logic",
    "ethics",
    "computer_science",
    "general_knowledge"
]

# Evaluation metrics
METRICS = {
    "accuracy": "accuracy",
    "confidence_calibration": "confidence_calibration",
    "reasoning_depth": "reasoning_depth",
    "hallucination_rate": "hallucination_rate"
}
