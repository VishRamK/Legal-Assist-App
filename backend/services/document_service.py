from api.document import upload_document  # Import the upload function from api
from flask import request

class DocumentService:
    def upload_and_process_document(self, file):
        """Handle document upload and processing through the API."""
        if file is None or file.filename == '':
            raise ValueError("No file part or no file selected.")
        return upload_document(file)  # Call the existing upload function
