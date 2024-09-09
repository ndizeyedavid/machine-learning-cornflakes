import fitz  # PyMuPDF
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def generate_response(text, question):
    """Generate a response to a question based on the text."""
    # Load a pre-trained question-answering model
    nlp = pipeline("question-answering")
    result = nlp(question=question, context=text)
    return result['answer']

def main(pdf_path, question):
    """Main function to extract text and generate a response."""
    text = extract_text_from_pdf(pdf_path)
    answer = generate_response(text, question)
    print("Question:", question)
    print("Answer:", answer)

if __name__ == "__main__":
    # Path to your PDF file
    pdf_path = "test.pdf"
    # Question to ask
    question = "What is Hypothyroidism?"
    main(pdf_path, question)
