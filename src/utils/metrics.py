from typing import Dict, List
import numpy as np
from sklearn.metrics import brier_score_loss
from ..models.evaluation import Metrics, Response
from ..core.question_bank import QuestionBank

def calculate_metrics(responses: Dict[str, Response], question_bank: QuestionBank) -> Metrics:
    """Calculate evaluation metrics from responses."""
    if not responses:
        raise ValueError("No responses to calculate metrics from")

    # Get questions for responses
    questions = {qid: question_bank.get_question_by_id(qid) for qid in responses.keys()}
    
    # Calculate basic metrics
    total_responses = len(responses)
    correct_responses = sum(1 for qid, resp in responses.items() 
                          if resp.answer.lower().strip() == questions[qid].correct_answer.lower().strip())
    accuracy = correct_responses / total_responses

    # Calculate confidence metrics
    confidences = [r.confidence for r in responses.values()]
    actual_outcomes = [1 if r.answer.lower().strip() == questions[r.question_id].correct_answer.lower().strip() 
                      else 0 for r in responses.values()]
    
    # Confidence calibration error (using Brier score)
    calibration_error = brier_score_loss(actual_outcomes, confidences)

    # Average confidence
    avg_confidence = np.mean(confidences)

    # Hallucination rate
    hallucinations = sum(1 for r in responses.values() if detect_hallucination(r.reasoning))
    hallucination_rate = hallucinations / total_responses

    # Group performance by subject area
    subject_performance = {}
    for subject in set(q.subject_area for q in questions.values()):
        subject_responses = {qid: resp for qid, resp in responses.items() 
                           if questions[qid].subject_area == subject}
        if subject_responses:
            correct = sum(1 for qid, resp in subject_responses.items() 
                        if resp.answer.lower().strip() == questions[qid].correct_answer.lower().strip())
            subject_performance[subject] = correct / len(subject_responses)

    # Group performance by difficulty
    difficulty_ranges = {
        "easy": (1.0, 1.5),
        "medium": (1.5, 2.0),
        "hard": (2.0, 2.5),
        "expert": (2.5, 3.0)
    }
    
    difficulty_performance = {}
    for diff_name, (min_diff, max_diff) in difficulty_ranges.items():
        diff_responses = {qid: resp for qid, resp in responses.items() 
                        if min_diff <= questions[qid].difficulty < max_diff}
        if diff_responses:
            correct = sum(1 for qid, resp in diff_responses.items() 
                        if resp.answer.lower().strip() == questions[qid].correct_answer.lower().strip())
            difficulty_performance[diff_name] = correct / len(diff_responses)

    # Group performance by question type
    type_performance = {}
    for q_type in set(q.type for q in questions.values()):
        type_responses = {qid: resp for qid, resp in responses.items() 
                        if questions[qid].type == q_type}
        if type_responses:
            correct = sum(1 for qid, resp in type_responses.items() 
                        if resp.answer.lower().strip() == questions[qid].correct_answer.lower().strip())
            type_performance[q_type] = correct / len(type_responses)

    return Metrics(
        accuracy=accuracy,
        confidence_calibration_error=calibration_error,
        average_confidence=avg_confidence,
        hallucination_rate=hallucination_rate,
        subject_area_performance=subject_performance,
        difficulty_performance=difficulty_performance,
        question_type_performance=type_performance
    )

def detect_hallucination(reasoning: str) -> bool:
    """
    Detect if a response shows signs of hallucination.
    
    Currently implements a simple heuristic:
    1. Very long reasoning that might indicate making up information
    2. Contains phrases that might indicate uncertainty or making up information
    """
    # Check for suspiciously long reasoning
    if len(reasoning) > 1000:  # Arbitrary threshold
        return True
    
    # Check for phrases that might indicate making up information
    uncertainty_phrases = [
        "i think", "probably", "might be", "could be", "maybe",
        "i believe", "i guess", "possibly", "perhaps"
    ]
    
    reasoning_lower = reasoning.lower()
    return any(phrase in reasoning_lower for phrase in uncertainty_phrases)

def calculate_confidence_calibration(predicted_probs: List[float], 
                                   actual_outcomes: List[int],
                                   n_bins: int = 10) -> Dict:
    """Calculate confidence calibration metrics."""
    
    # Create confidence bins
    bins = np.linspace(0, 1, n_bins + 1)
    binned = np.digitize(predicted_probs, bins) - 1
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_idx in range(n_bins):
        mask = binned == bin_idx
        if np.any(mask):
            bin_acc = np.mean(np.array(actual_outcomes)[mask])
            bin_conf = np.mean(np.array(predicted_probs)[mask])
            bin_count = np.sum(mask)
            
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_count)
    
    return {
        "accuracies": bin_accuracies,
        "confidences": bin_confidences,
        "counts": bin_counts
    }
