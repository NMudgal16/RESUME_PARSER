import pandas as pd

# Load the dataset (relative path from src/)
df = pd.read_csv("C:/Users/niharika mudgal/OneDrive\Desktop/AI_RESUME_PARSER/data/resume_data.csv")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head())

# Function to create NER data from structured columns
def create_ner_data(row):
    text_parts = []
    labels = []
    
    # Add career objective (tag as O)
    if pd.notna(row['career_objective']):
        words = row['career_objective'].split()
        text_parts.extend(words)
        labels.extend(["O"] * len(words))
    
    # Add skills (tag as B-SKILL)
    if pd.notna(row['skills']):
        skill_list = str(row['skills']).split(', ')
        for skill in skill_list:
            words = skill.split()
            text_parts.extend(words)
            labels.extend(["B-SKILL"] + ["I-SKILL"] * (len(words) - 1))
    
    # Add positions (tag as B-POSITION)
    if pd.notna(row['positions']):
        pos_words = str(row['positions']).split()
        text_parts.extend(pos_words)
        labels.extend(["B-POSITION"] + ["I-POSITION"] * (len(pos_words) - 1))
    
    # Add educational institution (tag as B-EDU)
    if pd.notna(row['educational_institution_name']):
        edu_words = str(row['educational_institution_name']).split()
        text_parts.extend(edu_words)
        labels.extend(["B-EDU"] + ["I-EDU"] * (len(edu_words) - 1))
    
    # You can add more columns here (e.g., 'languages' as B-LANG)
    
    return " ".join(text_parts), labels

# Apply to create new columns
df['text'] = df.apply(lambda row: create_ner_data(row)[0], axis=1)
df['labels'] = df.apply(lambda row: create_ner_data(row)[1], axis=1)

# Save the NER-ready dataset
df[['text', 'labels']].to_csv("C:/Users/niharika mudgal/OneDrive/Desktop/AI_RESUME_PARSER/data/ready_data.csv", index=False)
print("NER dataset created and saved!")
