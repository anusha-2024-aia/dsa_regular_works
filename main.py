
from transformers import pipeline

# Text Generation
generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

prompt = "Artificial Intelligence is changing the world because"

generated = generator(
    prompt,
    max_new_tokens=30,
    num_return_sequences=1
)

print("Generated Text:")
print(generated[0]["generated_text"])

# Text Classification
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

text = "I really enjoyed this movie."

classification = classifier(text)

print("\nClassification Result:")
print("Label:", classification[0]["label"])
print("Confidence:", round(classification[0]["score"], 4))



