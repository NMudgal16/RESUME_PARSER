from utils import extract_text_from_pdf
from parser import parse_resume

def main():
    pdf_path = input("Enter path to resume PDF: ")  # e.g., ../data/sample_resume.pdf
    text = extract_text_from_pdf(pdf_path)
    parsed = parse_resume(text)
    
    print("Parsed Resume Data:")
    for key, value in parsed.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()