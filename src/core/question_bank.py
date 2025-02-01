import json
import random
from pathlib import Path
from typing import Optional, List, Dict
from ..models.evaluation import Question, QuestionType, SubjectArea

class QuestionBank:
    def __init__(self, questions_dir: Optional[Path] = None):
        self.questions_dir = questions_dir or Path(__file__).parent.parent.parent / "data" / "questions"
        self.questions: Dict[str, Question] = {}
        self.load_questions()

    def load_questions(self):
        """Load questions from the questions directory."""
        if not self.questions_dir.exists():
            self._create_sample_questions()
        
        for question_file in self.questions_dir.glob("*.json"):
            with open(question_file, 'r') as f:
                data = json.load(f)
                # Handle both formats: list of questions or {"questions": [...]}
                questions_data = data.get("questions", data) if isinstance(data, dict) else data
                for q_data in questions_data:
                    # Convert string difficulty to float if needed
                    if isinstance(q_data.get("difficulty"), str):
                        if q_data["difficulty"].lower() == "easy":
                            q_data["difficulty"] = 1.0
                        elif q_data["difficulty"].lower() == "medium":
                            q_data["difficulty"] = 2.0
                        elif q_data["difficulty"].lower() == "hard":
                            q_data["difficulty"] = 3.0
                    
                    # Convert string enums to proper enum values
                    if isinstance(q_data.get("type"), str):
                        q_data["type"] = q_data["type"].lower()
                    
                    if isinstance(q_data.get("subject_area"), str):
                        q_data["subject_area"] = q_data["subject_area"].lower()
                    
                    try:
                        question = Question(**q_data)
                        self.questions[question.id] = question
                    except Exception as e:
                        print(f"Error loading question {q_data.get('id')}: {str(e)}")

    def _create_sample_questions(self):
        """Create sample questions if none exist."""
        self.questions_dir.mkdir(parents=True, exist_ok=True)
        sample_questions = [
            {
                "id": "math_001",
                "type": QuestionType.MULTIPLE_CHOICE,
                "subject_area": SubjectArea.MATHEMATICS,
                "difficulty": 1.0,
                "content": "What is the square root of 144?",
                "correct_answer": "12",
                "options": ["10", "11", "12", "13"],
                "explanation": "The square root of 144 is 12 because 12 × 12 = 144"
            },
            {
                "id": "cs_001",
                "type": QuestionType.SHORT_ANSWER,
                "subject_area": SubjectArea.COMPUTER_SCIENCE,
                "difficulty": 2.0,
                "content": "Explain the concept of recursion in programming.",
                "correct_answer": "Recursion is a programming concept where a function calls itself to solve a problem by breaking it down into smaller subproblems.",
                "explanation": "Recursion is fundamental to many algorithms and problem-solving approaches in computer science."
            }
        ]
        
        with open(self.questions_dir / "sample_questions.json", 'w') as f:
            json.dump(sample_questions, f, indent=2)
    
    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Get a question by its ID."""
        return self.questions.get(question_id)
    
    def get_questions_by_difficulty(self, difficulty: float, limit: Optional[int] = None) -> List[Question]:
        """Get questions with a specific difficulty level."""
        questions = [q for q in self.questions.values() if q.difficulty == difficulty]
        if limit:
            questions = random.sample(questions, min(limit, len(questions)))
        return questions
    
    def get_questions_by_subject(self, subject: SubjectArea, limit: Optional[int] = None) -> List[Question]:
        """Get questions for a specific subject area."""
        questions = [q for q in self.questions.values() if q.subject_area == subject]
        if limit:
            questions = random.sample(questions, min(limit, len(questions)))
        return questions
    
    def get_random_question(self) -> Optional[Question]:
        """Get a random question from the bank."""
        if not self.questions:
            return None
        return random.choice(list(self.questions.values()))
