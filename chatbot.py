import pandas as pd
from transformers import pipeline

# Load DistilBERT question-answering model
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

# Load dataset
data = pd.read_csv("dataset/faq.csv")

# Ask user question
question = input("Ask a question: ")

# Find best matching context
best_context = ""

for index, row in data.iterrows():
    if any(word.lower() in row["Question"].lower() for word in question.split()):
        best_context = row["Context"]
        break

# If no context found
if best_context == "":
    print("\nSorry, I could not find relevant information.")

else:
    # Run model
    result = qa_pipeline(
        question=question,
        context=best_context
    )

    # Print answer
    print("\nContext:", best_context)
    print("Answer:", result["answer"])