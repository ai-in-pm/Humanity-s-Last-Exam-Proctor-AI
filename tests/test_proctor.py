import pytest
from datetime import datetime
from src.core.proctor import HLEProctor
from src.models.evaluation import Question, QuestionType, SubjectArea

@pytest.fixture
def proctor():
    return HLEProctor()

@pytest.fixture
def sample_question():
    return Question(
        id="test_001",
        type=QuestionType.MULTIPLE_CHOICE,
        subject_area=SubjectArea.MATHEMATICS,
        difficulty=1.0,
        content="What is 2 + 2?",
        correct_answer="4",
        options=["3", "4", "5", "6"]
    )

def test_start_session(proctor):
    session_id = proctor.start_session("test_llm")
    assert session_id is not None
    assert proctor.current_session is not None
    assert proctor.current_session["llm_name"] == "test_llm"

def test_get_next_question(proctor):
    proctor.start_session("test_llm")
    question = proctor.get_next_question()
    assert question is not None
    assert isinstance(question, Question)

def test_submit_response(proctor, sample_question):
    proctor.start_session("test_llm")
    proctor.question_bank.add_question(sample_question)
    
    is_correct, details = proctor.submit_response(
        question_id=sample_question.id,
        answer="4",
        confidence=0.9,
        reasoning="Basic arithmetic"
    )
    
    assert is_correct is True
    assert "current_difficulty" in details

def test_generate_report(proctor, sample_question):
    proctor.start_session("test_llm")
    proctor.question_bank.add_question(sample_question)
    
    proctor.submit_response(
        question_id=sample_question.id,
        answer="4",
        confidence=0.9,
        reasoning="Basic arithmetic"
    )
    
    report = proctor.generate_report()
    assert report.llm_name == "test_llm"
    assert report.total_questions == 1
    assert hasattr(report, "metrics")

def test_difficulty_adjustment(proctor, sample_question):
    proctor.start_session("test_llm")
    initial_difficulty = proctor.current_session["current_difficulty"]
    
    # Test correct answer increases difficulty
    proctor.submit_response(
        question_id=sample_question.id,
        answer="4",
        confidence=0.9,
        reasoning="Basic arithmetic"
    )
    assert proctor.current_session["current_difficulty"] > initial_difficulty
    
    # Test incorrect answer decreases difficulty
    initial_difficulty = proctor.current_session["current_difficulty"]
    proctor.submit_response(
        question_id=sample_question.id,
        answer="3",
        confidence=0.9,
        reasoning="Wrong answer"
    )
    assert proctor.current_session["current_difficulty"] < initial_difficulty
