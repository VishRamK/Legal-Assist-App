import os
from flask import request  # For Flask; use appropriate imports for FastAPI
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader  # Example for reading PDF files
import docx  # Library for handling .docx files
from config import Config

# Ensure the upload folder exists
if not os.path.exists(Config.UPLOAD_FOLDER):
    os.makedirs(Config.UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the uploaded file is allowed based on its extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def upload_document(file):
    """Handle document upload and processing."""
    if not file or file.filename == '':
        raise ValueError("No file part or no file selected.")
    
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(file_path)  # Save the file to the specified upload folder

        # Process the file (extract text)
        content = extract_text_from_file(file_path, filename)
        return content
    else:
        raise ValueError("File type not allowed. Please upload a PDF or DOCX file.")

def extract_text_from_file(file_path, filename):
    """Extract text from the uploaded file based on its type."""
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type for text extraction.")

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ''  # Handle case where text might be None
    return text

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    text = ""
    doc = docx.Document(file_path)
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text
