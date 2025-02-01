from pydantic import BaseModel
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    IMAGE_BASED = "image_based"

class SubjectArea(str, Enum):
    MATHEMATICS = "mathematics"
    SCIENCE = "science"
    HUMANITIES = "humanities"
    LOGIC = "logic"
    ETHICS = "ethics"
    COMPUTER_SCIENCE = "computer_science"
    GENERAL_KNOWLEDGE = "general_knowledge"

class Question(BaseModel):
    id: str
    type: QuestionType
    subject_area: SubjectArea
    difficulty: float
    content: str
    correct_answer: str
    options: Optional[List[str]] = None  # For multiple choice questions
    image_url: Optional[str] = None  # For image-based questions
    explanation: Optional[str] = None
    metadata: Optional[Dict] = None

class Response(BaseModel):
    question_id: str
    answer: str
    confidence: float
    reasoning: str
    timestamp: datetime

class Metrics(BaseModel):
    accuracy: float
    confidence_calibration_error: float
    average_confidence: float
    hallucination_rate: float
    subject_area_performance: Dict[str, float]
    difficulty_performance: Dict[str, float]
    question_type_performance: Dict[str, float]

class EvaluationResult(BaseModel):
    session_id: str
    llm_name: str
    total_questions: int
    metrics: Metrics
    start_time: datetime
    end_time: datetime
    recommendations: Optional[List[str]] = None
