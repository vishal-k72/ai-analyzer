import os
import io
import json
from urllib.parse import quote_plus

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pypdf import PdfReader

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY missing. .env file me daalo.")

client = Groq(api_key=GROQ_API_KEY)
app = FastAPI(title="AI Resume Analyzer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def ask_llm(system: str, user: str, json_mode: bool = False) -> str:
    try:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.5,
            **kwargs,
        )
        return res.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq error: {e}")


def read_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        raise HTTPException(status_code=400, detail="PDF read nahi ho paya.")
    text = text.strip()
    if len(text) < 50:
        raise HTTPException(
            status_code=400,
            detail="PDF me text nahi mila (scanned image PDF ho sakta hai).",
        )
    return text[:12000]


@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.post("/api/analyze")
async def analyze(resume: UploadFile = File(...), role: str = Form("")):
    if not (resume.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sirf PDF upload karo.")
    text = read_pdf(await resume.read())
    target = f"Target role: {role}" if role.strip() else "No specific target role given."
    system = "You are an expert resume reviewer and career coach. Reply in clear Markdown."
    user = f"""{target}

Resume text:
\"\"\"
{text}
\"\"\"

Give:
## Overall Score (out of 100) with one-line reason
## Strengths
## Weaknesses / Gaps
## Missing Keywords & Skills (for the role)
## ATS Tips
## Improvement Suggestions (specific, actionable bullets)
## Suggested Rewrite of Summary (3-4 lines)"""
    return {"result": ask_llm(system, user)}


@app.post("/api/ask")
async def ask(role: str = Form(""), question: str = Form(...)):
    system = (
        "You are a senior hiring manager and career mentor. "
        "Answer questions about jobs, roles, skills, salary, interviews and careers. "
        "Be practical and concise. Reply in Markdown."
    )
    user = f"Job/Role: {role or 'general'}\n\nQuestion: {question}"
    return {"result": ask_llm(system, user)}


@app.post("/api/roadmap")
async def roadmap(role: str = Form(...), level: str = Form("Beginner"), months: str = Form("6")):
    system = "You are a career roadmap expert. Reply in clean Markdown."
    user = f"""Create a step-by-step learning roadmap to become a {role}.
Current level: {level}. Time available: {months} months.

Format:
## Overview
## Month-by-Month Plan (skills, topics, 1 mini project each)
## Tools & Technologies
## Free Resources
## Final Portfolio Projects
## Interview Prep Checklist"""
    return {"result": ask_llm(system, user)}


@app.post("/api/youtube")
async def youtube(topic: str = Form(...)):
    system = "You output ONLY valid JSON."
    user = f"""Suggest 6 useful YouTube video ideas for interview preparation on: {topic}.
Return JSON exactly like:
{{"videos":[{{"title":"short video title","query":"best youtube search query","why":"one line why useful"}}]}}"""
    raw = ask_llm(system, user, json_mode=True)
    try:
        videos = json.loads(raw).get("videos", [])
    except Exception:
        raise HTTPException(status_code=500, detail="AI response parse nahi hua, dobara try karo.")
    for v in videos:
        v["url"] = "https://www.youtube.com/results?search_query=" + quote_plus(v.get("query", topic))
    return {"videos": videos}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
