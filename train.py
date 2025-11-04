import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments

# Load preprocessed dataset
df = pd.read_csv("C:/ Users/niharika mudgal/OneDrive/Desktop/AI_RESUME_PARSER/data/ready_data.csv")
df['labels'] = df['labels'].apply(eval)  # Convert string lists to actual lists

# Debug and clean: Filter rows where text and labels lengths roughly match
df = df[df.apply(lambda row: len(row['text'].split()) == len(row['labels']), axis=1)]
print(f"Filtered dataset shape: {df.shape}")  # Check if rows were removed

dataset = Dataset.from_pandas(df)

# Define labels (based on preprocessing)
label_list = ["O", "B-SKILL", "I-SKILL", "B-POSITION", "I-POSITION", "B-EDU", "I-EDU"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize and align labels (with bounds check to fix IndexError)
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx < len(label):  # Bounds check to prevent IndexError
                if word_idx != previous_word_idx:
                    label_ids.append(label2id.get(label[word_idx], -100))
                else:
                    label_ids.append(label2id.get(label[word_idx], -100) if label[word_idx].startswith("I-") else -100)
            else:
                label_ids.append(-100)  # Fallback for out-of-range
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Split dataset
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = tokenized_dataset["train"]
val_test = tokenized_dataset["test"].train_test_split(test_size=0.5)
val_dataset = val_test["train"]
test_dataset = val_test["test"]

# Load model
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_list), id2label=id2label, label2id=label2id)

# Training arguments (updated for newer transformers versions)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Changed from 'evaluation_strategy'
    save_strategy="epoch",
    num_train_epochs=3,  # Adjust for time
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and save
trainer.train()
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Model trained and saved!")

