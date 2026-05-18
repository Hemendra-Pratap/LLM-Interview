from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# =========================
# 🔷 CONFIG
# =========================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ FIXED MODEL (important)
model = genai.GenerativeModel("gemini-flash-latest")

app = FastAPI()

# =========================
# 🔷 CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔷 SAFE LLM CALL
# =========================
def safe_generate(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("LLM ERROR:", e)
        return None


# =========================
# 🔷 LLM: QUESTIONS
# =========================
def generate_questions_llm(role, skills, question_count=5, difficulty='medium'):
    question_count = max(1, min(question_count, 10))
    difficulty_text = difficulty.capitalize()

    prompt = f"""
Generate exactly {question_count} technical interview questions for a {role} with skills in {', '.join(skills)}.

Question level: {difficulty_text}

Rules:
- Clear and concise
- No explanations
- Numbered list
"""

    raw_text = safe_generate(prompt)
    if not raw_text:
        return []

    questions = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if "." in line:
            parts = line.split(".", 1)
            if parts[0].isdigit():
                line = parts[1].strip()

        questions.append(line)

    return questions[:question_count]


# =========================
# 🔷 LLM: IDEAL ANSWERS
# =========================
def generate_ideal_answers_llm(questions):
    prompt = "Generate ideal answers:\n"

    for i, q in enumerate(questions):
        prompt += f"\n{i+1}. {q}"

    raw_text = safe_generate(prompt)

    if not raw_text:
        return ["Ideal answer unavailable."] * len(questions)

    answers = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if "." in line:
            parts = line.split(".", 1)
            if parts[0].isdigit():
                line = parts[1].strip()

        answers.append(line)

    # ✅ fix mismatch
    while len(answers) < len(questions):
        answers.append("Ideal answer unavailable.")

    return answers[:len(questions)]


# =========================
# 🔥 NEW: BATCH FEEDBACK (OPTIMIZED)
# =========================
def generate_feedback_batch(questions, ideal_answers, candidate_answers):
    prompt = "You are an expert technical interviewer.\n\n"

    for i in range(len(questions)):
        prompt += f"""
{i+1}. Question: {questions[i]}
Ideal Answer: {ideal_answers[i]}
Candidate Answer: {candidate_answers[i]}
"""

    prompt += """
Instructions:
- Give feedback for each answer
- Mention what is correct, missing, and how to improve
- Keep 2-3 lines per answer
- No scores
- Return numbered list only
"""

    raw = safe_generate(prompt)

    if not raw:
        return ["Feedback unavailable."] * len(questions)

    feedbacks = []

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        if "." in line:
            parts = line.split(".", 1)
            if parts[0].isdigit():
                feedbacks.append(parts[1].strip())

    # ✅ fix mismatch
    while len(feedbacks) < len(questions):
        feedbacks.append("Feedback unavailable.")

    return feedbacks[:len(questions)]


# =========================
# 🔷 MODELS
# =========================
class EvaluationRequest(BaseModel):
    ideal_answers: List[str]
    candidate_answers: List[str]


class QuestionRequest(BaseModel):
    role: str
    skills: List[str]
    question_count: int = 5
    difficulty: str = 'medium'


# =========================
# 🔷 API: GENERATE
# =========================
@app.post("/generate-questions")
def generate_questions(request: QuestionRequest):
    try:
        questions = generate_questions_llm(
            request.role,
            request.skills,
            question_count=request.question_count,
            difficulty=request.difficulty
        )

        if not questions:
            raise ValueError("LLM failed")

        ideal_answers = generate_ideal_answers_llm(questions)

        return {
            "questions": questions,
            "ideal_answers": ideal_answers
        }

    except Exception as e:
        print("GENERATE ERROR:", e)

        return {
            "questions": ["Fallback question"],
            "ideal_answers": ["Fallback answer"]
        }


@app.post("/evaluate")
def evaluate(request: EvaluationRequest):

    if len(request.ideal_answers) != len(request.candidate_answers):
        return {"error": "Length mismatch"}

    questions = [
        f"Question {i+1}"
        for i in range(len(request.ideal_answers))
    ]

    # 🔷 Batch feedback
    feedbacks = generate_feedback_batch(
        questions,
        request.ideal_answers,
        request.candidate_answers
    )

    results = []

    for i in range(len(request.ideal_answers)):

        candidate = request.candidate_answers[i]
        ideal = request.ideal_answers[i]

        score_prompt = f"""
You are an expert technical interviewer.

Evaluate the candidate answer compared to the ideal answer.

Ideal Answer:
{ideal}

Candidate Answer:
{candidate}

Rules:
- Score from 0 to 100
- Consider:
  correctness
  completeness
  technical depth
  clarity
- Return ONLY a single integer
- Example output: 85
"""

        score = safe_generate(score_prompt)

        try:
            score = int(score.strip())
        except:
            score = 50

        # 🔷 Category logic
        if score >= 75:
            category = "Strong"
        elif score >= 45:
            category = "Average"
        else:
            category = "Weak"

        results.append({
            "score": score,
            "category": category,
            "feedback": feedbacks[i]
        })

    return results
