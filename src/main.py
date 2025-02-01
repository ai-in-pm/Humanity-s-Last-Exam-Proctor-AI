from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict
import os

from src.core.proctor import HLEProctor
from src.models.evaluation import Question, Response

app = FastAPI(title="HLE Proctor AI")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "image")
app.mount("/images", StaticFiles(directory=static_dir), name="images")

# Initialize proctor
proctor = HLEProctor()

class SessionRequest(BaseModel):
    llm_name: str
    max_questions: Optional[int] = 3000

class AnswerSubmission(BaseModel):
    question_id: str
    answer: str
    confidence: float
    reasoning: str

@app.post("/session/start")
async def start_session(request: SessionRequest):
    """Start a new evaluation session."""
    try:
        session_id = proctor.start_session(request.llm_name, request.max_questions)
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/question/next")
async def get_next_question():
    """Get the next question in the current session."""
    try:
        question = proctor.get_next_question()
        if question is None:
            raise HTTPException(status_code=404, detail="No more questions available")
        return question
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answer/submit")
async def submit_answer(submission: AnswerSubmission):
    """Submit an answer for evaluation."""
    try:
        is_correct, details = proctor.submit_response(
            submission.question_id,
            submission.answer,
            submission.confidence,
            submission.reasoning
        )
        return {
            "is_correct": is_correct,
            **details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/generate")
async def generate_report():
    """Generate evaluation report for the current session."""
    try:
        report = proctor.generate_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report/export")
async def export_report(filepath: str):
    """Export evaluation results to a file."""
    try:
        proctor.export_results(filepath)
        return {"message": f"Results exported to {filepath}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=9999, reload=True)
