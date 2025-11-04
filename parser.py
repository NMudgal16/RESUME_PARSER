from transformers import pipeline
from utils import preprocess_text, extract_email

# Load fine-tuned NER pipeline
ner_pipeline = pipeline("ner", model="./fine_tuned_model", tokenizer="./fine_tuned_model")

def parse_resume(text):
    text = preprocess_text(text)
    entities = ner_pipeline(text)
    
    # Group entities (e.g., combine B-PER I-PER into full name)
    parsed = {"name": None, "organizations": [], "skills": []}
    current_entity = None
    for ent in entities:
        if ent["entity"].startswith("B-"):
            if current_entity:
                parsed[current_entity["type"]].append(current_entity["text"])
            current_entity = {"type": ent["entity"][2:].lower(), "text": ent["word"]}
        elif ent["entity"].startswith("I-") and current_entity:
            current_entity["text"] += " " + ent["word"]
    if current_entity:
        parsed[current_entity["type"]].append(current_entity["text"])
    
    parsed["email"] = extract_email(text)
    return parsed