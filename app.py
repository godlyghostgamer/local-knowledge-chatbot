import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from transformers import pipeline

from sentence_transformers import SentenceTransformer, util

# Create FastAPI app
app = FastAPI()

# Load HTML templates
templates = Jinja2Templates(directory="templates")

# Load DistilBERT QA model
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

# Load semantic similarity model
semantic_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Load dataset
data = pd.read_csv("dataset/faq.csv")


# Home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# Chatbot route
@app.get("/ask")
def ask_question(question: str):

    # Convert user question into embedding
    question_embedding = semantic_model.encode(
        question,
        convert_to_tensor=True
    )

    best_score = -1
    best_context = ""

    # Find most similar dataset question
    for index, row in data.iterrows():

        dataset_question = row["Question"]

        dataset_embedding = semantic_model.encode(
            dataset_question,
            convert_to_tensor=True
        )

        similarity = util.cos_sim(
            question_embedding,
            dataset_embedding
        )

        score = similarity.item()

        # Keep best match
        if score > best_score:
            best_score = score
            best_context = row["Context"]

    # Generate answer using DistilBERT
    result = qa_pipeline(
        question=question,
        context=best_context
    )

    return {
        "question": question,
        "context": best_context,
        "answer": result["answer"],
        "similarity_score": round(best_score, 2)
    }