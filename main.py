from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import json
import re

load_dotenv()

# =========================
# 🔷 CONFIG
# =========================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Stable lightweight model
model = genai.GenerativeModel("gemini-flash-latest")

app = FastAPI()

# =========================
# 🔷 SERVE FRONTEND
# =========================
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


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
# 🔷 SAFE GEMINI CALL
# =========================
def safe_generate(prompt):

    try:

        response = model.generate_content(prompt)

        print("\n================ TOKEN USAGE ================\n")

        try:
            print(response.usage_metadata)
        except:
            print("Token metadata unavailable")

        print("\n=============================================\n")

        return response.text.strip()

    except Exception as e:

        print("LLM ERROR:", e)
        return None


# =========================
# 🔷 REQUEST MODELS
# =========================
class QuestionRequest(BaseModel):
    role: str
    skills: List[str]
    question_count: int = 3
    difficulty: str = "easy"


class EvaluationRequest(BaseModel):
    questions: List[str]
    ideal_answers: List[str]
    candidate_answers: List[str]


# =========================
# 🔷 GENERATE QUESTIONS
# 🔥 ONLY 1 API CALL
# =========================
@app.post("/generate-questions")
def generate_questions(request: QuestionRequest):

    prompt = f"""
Generate exactly {request.question_count} SHORT technical interview questions and concise ideal answers.

Role:
{request.role}

Skills:
{', '.join(request.skills)}

Difficulty:
{request.difficulty}

Return ONLY valid JSON.

Format:

[
  {{
    "question": "question here",
    "ideal_answer": "ideal answer here"
  }}
]

Rules:
- Beginner to intermediate level
- Questions must be short
- Ideal answers must be ONE concise sentence
- Keep ideal answers under 25 words
- No markdown
- No explanations
"""

    raw = safe_generate(prompt)

    if not raw:
        return {
            "error": "LLM generation failed"
        }

    try:

        # remove markdown wrappers
        raw = re.sub(r"```json", "", raw)
        raw = re.sub(r"```", "", raw)

        data = json.loads(raw)

        questions = []
        ideal_answers = []

        for item in data:

            questions.append(item["question"])
            ideal_answers.append(item["ideal_answer"])

        return {
            "questions": questions,
            "ideal_answers": ideal_answers
        }

    except Exception as e:

        print("JSON PARSE ERROR:", e)
        print(raw)

        return {
            "error": "Failed to parse Gemini response"
        }


# =========================
# 🔷 EVALUATE ANSWERS
# 🔥 ONLY 1 API CALL
# =========================
@app.post("/evaluate")
def evaluate(request: EvaluationRequest):

    if len(request.questions) != len(request.candidate_answers):

        return {
            "error": "Length mismatch"
        }

    qa_block = ""

    for i in range(len(request.questions)):

        qa_block += f"""

Question {i+1}:
{request.questions[i]}

Ideal Answer:
{request.ideal_answers[i]}

Candidate Answer:
{request.candidate_answers[i]}
"""

    prompt = f"""
Evaluate all candidate answers.

Return ONLY valid JSON.

Format:

[
  {{
    "score": 85,
    "category": "Strong",
    "feedback": "Good answer but missing detail."
  }}
]

Rules:
- Score from 0-100
- Category must be:
  Strong
  Average
  Weak
- Feedback must be ONE short sentence
- Keep feedback under 15 words
- Avoid long explanations

Questions and Answers:

{qa_block}
"""

    raw = safe_generate(prompt)

    if not raw:

        return {
            "error": "LLM evaluation failed"
        }

    try:

        raw = re.sub(r"```json", "", raw)
        raw = re.sub(r"```", "", raw)

        data = json.loads(raw)

        return data

    except Exception as e:

        print("EVALUATION PARSE ERROR:", e)
        print(raw)

        return {
            "error": "Failed to parse evaluation response"
        }
