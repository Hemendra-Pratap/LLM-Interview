from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from ml_module import evaluate_all_answers

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
def generate_questions_llm(role, skills):
    prompt = f"""
Generate exactly 5 technical interview questions for a {role} with skills in {', '.join(skills)}.

Rules:
- Clear and concise
- Beginner to intermediate
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

    return questions[:5]


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


# =========================
# 🔷 API: GENERATE
# =========================
@app.post("/generate-questions")
def generate_questions(request: QuestionRequest):
    try:
        questions = generate_questions_llm(request.role, request.skills)

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


# =========================
# 🔷 API: EVALUATE (OPTIMIZED)
# =========================
@app.post("/evaluate")
def evaluate(request: EvaluationRequest):

    if len(request.ideal_answers) != len(request.candidate_answers):
        return {"error": "Length mismatch"}

    # 🔷 ML scoring
    base_results = evaluate_all_answers(
        request.ideal_answers,
        request.candidate_answers
    )

    # 🔥 SINGLE LLM CALL (optimized)
    questions = [f"Question {i+1}" for i in range(len(base_results))]

    feedbacks = generate_feedback_batch(
        questions,
        request.ideal_answers,
        request.candidate_answers
    )

    final_results = []

    for i in range(len(base_results)):
        result = base_results[i]
        result["feedback"] = feedbacks[i]
        final_results.append(result)

    return final_results