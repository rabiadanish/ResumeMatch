from google import genai
from google.genai import types
import os
from typing import List

# -----------------------------
# Generate Interview Questions
# -----------------------------
# 1) ============ GEMINI CONFIGURATION ============
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set. Please configure it in your .env or environment variables.")
else:
    client = genai.Client(api_key=GEMINI_API_KEY)

model = "gemini-2.5-flash-lite-preview-06-17"
tools = [types.Tool(googleSearch=types.GoogleSearch())]

generate_content_config = types.GenerateContentConfig(
    temperature=0.7,
    thinking_config=types.ThinkingConfig(thinking_budget=0),
    tools=tools,
    response_mime_type="text/plain",
)

def build_llm_prompt(job_text: str) -> str:
    return f"""
You are an HR expert. Based on the following job description, generate exactly 5 interview questions
tailored to assess the candidate's skills, experience, and cultural fit. Do not include answers.
Keep each question on a single line and avoid numbering or bullet characters.

Job Description:
{job_text}
"""
# 2) ================Streaming Helper================ 

def query_gemini_stream(job_text: str) -> str:
    """
    Calls Gemini in streaming mode and returns the concatenated text output.
    """
    contents = [
        types.Content(
            role="user",
            parts=[{"text": build_llm_prompt(job_text)}]
        )
    ]
    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        # each chunk may have .text
        if getattr(chunk, "text", None):
            full_response += chunk.text
    return full_response.strip()

# 3) ============== Question Generator ====================

def generate_interview_questions(job_text: str) -> List[str]:
    raw = query_gemini_stream(job_text)
    # split into lines, strip any leading bullets/hyphens/spaces
    lines = []
    for line in raw.splitlines():
        clean = line.strip().lstrip("-• ").rstrip()
        if clean:
            lines.append(clean)
        if len(lines) >= 5:
            break
    return lines