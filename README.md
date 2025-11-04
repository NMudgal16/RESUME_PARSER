# RESUME_PARSER
# AI Resume Parser

An end-to-end machine learning project for intelligent resume parsing using Named Entity Recognition (NER). This application extracts key information like skills, positions, education, and contact details from PDF resumes, making it useful for HR tools, job matching, and recruitment automation.

## Features
- **Data Preprocessing**: Converts structured Kaggle dataset into NER-ready format with BIO tagging.
- **Model Training**: Fine-tunes a DistilBERT model for entity recognition (skills, positions, education).
- **Real-Time Parsing**: Processes PDF uploads and extracts entities using the trained model.
- **Web Interface**: Deployable Streamlit app for easy user interaction.
- **Evaluation**: Includes training metrics and error handling for robustness.

## Tech Stack
- **Languages**: Python
- **Libraries**: Hugging Face Transformers, PyTorch, Pandas, Streamlit, PyPDF2, SpaCy
- **Tools**: GitHub for version control, Streamlit Cloud for deployment

## Installation and Setup
1. **Clone the Repository**:
2. **Set Up Environment**:
3. **Prepare Data**:
- Place your Kaggle dataset CSV as `data/resume_data.csv`.
- Run preprocessing: `python src/preprocess.py`.

4. **Train the Model**:
- Run `python src/train.py` (install `accelerate` if needed).

5. **Run the App**:
- `streamlit run src/app.py`
- Upload a PDF resume to test parsing.

## Usage
- Upload a PDF resume via the Streamlit interface.
- View extracted entities in JSON format (e.g., skills, positions).
- Deploy on Streamlit Cloud for a live demo.

## Project Structure
AI_Resume_Parser/ ├── data/ # Dataset files ├── src/ # Source code │ ├── preprocess.py # Data preprocessing │ ├── train.py # Model training │ ├── parser.py # Parsing logic │ ├── utils.py # Helper functions │ └── app.py # Streamlit app ├── requirements.txt # Dependencies └── README.md # This file

## Results
- Trained on 9,544 samples with BIO tagging.
- Achieves entity extraction for custom labels (O, B-SKILL, etc.).
- Suitable for further fine-tuning or integration into larger systems.


## License
MIT License - Free to use and modify.


