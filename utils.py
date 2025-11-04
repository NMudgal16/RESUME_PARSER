import PyPDF2
import re
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text
def preprocess_text(text):
    """Clean and normalize text."""
    text = text.lower()  # Lowercase for consistency
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()
def extract_email(text):
    """Extract email using regex."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None
def extract_skills(text, skill_list=['python', 'machine learning', 'nlp', 'sql']):
    """Simple keyword-based skill extraction."""
    found_skills = [skill for skill in skill_list if skill in text]
    return found_skills