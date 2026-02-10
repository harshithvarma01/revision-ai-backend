from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
from pypdf import PdfReader


app = FastAPI()

# Load small free model
generator = pipeline("text-generation", model="distilgpt2")

class InputText(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "AI Revision Agent Running"}


@app.post("/analyze")
def analyze(data: InputText):

    summary_prompt = f"Give a short exam revision summary:\n{data.text}\nSummary:"
    question_prompt = f"Create 5 short exam questions:\n{data.text}\nQuestions:"

    summary = generator(
    summary_prompt,
    max_length=80,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True,
    top_k=50,
    top_p=0.95
)[0]["generated_text"]


    questions = generator(
    question_prompt,
    max_length=100,
    num_return_sequences=1,
    temperature=0.8,
    do_sample=True,
    top_k=50,
    top_p=0.95
)[0]["generated_text"]


    return {
        "summary": summary,
        "questions": questions
    }

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    content = await file.read()

    text = ""

    # If PDF
    if file.filename.endswith(".pdf"):
        with open("temp.pdf", "wb") as f:
            f.write(content)

        reader = PdfReader("temp.pdf")
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

    else:
        text = content.decode("utf-8")

    summary_prompt = f"Give a short exam revision summary:\n{text}\nSummary:"
    question_prompt = f"Create 5 short exam questions:\n{text}\nQuestions:"

    summary = generator(
        summary_prompt,
        max_length=80,
        num_return_sequences=1
    )[0]["generated_text"]

    questions = generator(
        question_prompt,
        max_length=100,
        num_return_sequences=1
    )[0]["generated_text"]

    return {
        "summary": summary,
        "questions": questions
    }
