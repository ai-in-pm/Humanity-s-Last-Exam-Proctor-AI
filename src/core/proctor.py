from typing import Dict, List, Optional, Tuple
import random
from datetime import datetime
import json
from pathlib import Path

from src.models.evaluation import Question, Response, EvaluationResult, Metrics
from src.utils.metrics import calculate_metrics, detect_hallucination
from src.core.question_bank import QuestionBank

class HLEProctor:
    def __init__(self):
        """Initialize the HLE Proctor with a question bank."""
        self.question_bank = QuestionBank()
        self.current_session: Optional[str] = None
        self.llm_name: Optional[str] = None
        self.max_questions: int = 3000
        self.questions_asked: List[str] = []
        self.responses: Dict[str, Response] = {}
        self.current_difficulty: float = 1.0
        self.session_start_time: Optional[datetime] = None

    def start_session(self, llm_name: str, max_questions: int = 3000) -> str:
        """Start a new evaluation session."""
        self.current_session = f"{llm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.llm_name = llm_name
        self.max_questions = max_questions
        self.questions_asked = []
        self.responses = {}
        self.current_difficulty = 1.0
        self.session_start_time = datetime.now()
        return self.current_session

    def get_next_question(self) -> Optional[Question]:
        """Get the next question based on current performance."""
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")
        
        if len(self.questions_asked) >= self.max_questions:
            return None
        
        # Adjust difficulty based on recent performance
        if len(self.responses) > 0:
            recent_responses = list(self.responses.values())[-5:]
            correct_count = sum(1 for r in recent_responses if self._is_correct(r))
            if correct_count >= 4:  # If doing well, increase difficulty
                self.current_difficulty = min(3.0, self.current_difficulty + 0.2)
            elif correct_count <= 1:  # If struggling, decrease difficulty
                self.current_difficulty = max(1.0, self.current_difficulty - 0.2)
        
        question = self.question_bank.get_question(self.current_difficulty)
        if question and question.id not in self.questions_asked:
            self.questions_asked.append(question.id)
            return question
        return None

    def submit_response(self, question_id: str, answer: str, confidence: float, reasoning: str) -> Tuple[bool, Dict]:
        """Submit and evaluate a response."""
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")
        
        if question_id not in self.questions_asked:
            raise ValueError("Invalid question ID.")
        
        question = self.question_bank.get_question_by_id(question_id)
        if not question:
            raise ValueError("Question not found.")
        
        # Record response
        response = Response(
            question_id=question_id,
            answer=answer,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
        self.responses[question_id] = response
        
        # Evaluate response
        is_correct = self._is_correct(response)
        has_hallucination = detect_hallucination(reasoning)
        
        return is_correct, {
            "is_correct": is_correct,
            "has_hallucination": has_hallucination,
            "correct_answer": question.correct_answer,
            "explanation": question.explanation
        }

    def _is_correct(self, response: Response) -> bool:
        """Check if a response is correct."""
        question = self.question_bank.get_question_by_id(response.question_id)
        if not question:
            return False
        
        if question.type == "multiple_choice":
            return response.answer.lower() == question.correct_answer.lower()
        else:
            # For short answer questions, implement more sophisticated matching
            # This is a simple implementation and should be enhanced
            return response.answer.lower().strip() == question.correct_answer.lower().strip()

    def generate_report(self) -> EvaluationResult:
        """Generate a comprehensive evaluation report."""
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")
        
        metrics = calculate_metrics(self.responses, self.question_bank)
        
        return EvaluationResult(
            session_id=self.current_session,
            llm_name=self.llm_name,
            total_questions=len(self.questions_asked),
            metrics=metrics,
            start_time=self.session_start_time,
            end_time=datetime.now(),
            recommendations=self._generate_recommendations(metrics)
        )

    def _generate_recommendations(self, metrics: Metrics) -> List[str]:
        """Generate recommendations based on performance metrics."""
        recommendations = []
        
        if metrics.accuracy < 0.6:
            recommendations.append("Consider reviewing fundamental concepts across subjects.")
        
        if metrics.confidence_calibration_error > 0.2:
            recommendations.append("Work on improving confidence calibration.")
        
        if metrics.hallucination_rate > 0.1:
            recommendations.append("Focus on providing more precise and factual responses.")
        
        # Add subject-specific recommendations
        for subject, performance in metrics.subject_area_performance.items():
            if performance < 0.5:
                recommendations.append(f"Additional practice recommended in {subject}.")
        
        return recommendations

    def export_results(self, filepath: str):
        """Export evaluation results to a file."""
        if not self.current_session:
            raise ValueError("No active session. Call start_session first.")
        
        report = self.generate_report()
        
        # Ensure the directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Export as JSON
        with open(filepath, 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)
