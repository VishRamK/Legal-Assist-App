import os

class Config:
    # Flask or FastAPI secret key for session management and CSRF protection
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_default_secret_key')

    # Database configuration (if you're using a database)
    DATABASE_URI = os.environ.get('DATABASE_URI', 'sqlite:///your_database.db')

    # API keys for external services (like LLM API)
    LLM_API_KEY = os.environ.get('LLM_API_KEY', 'your_llm_api_key')

    # Other application settings
    DEBUG = os.environ.get('DEBUG', 'False') == 'True'  # Convert string to boolean

    # File upload settings (if applicable)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Limit upload size to 16 MB

    # Other settings can be added here as needed
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Directory for uploaded files
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}  # Allowed file extensions for uploads

# To use the Config class in your app
# from config import Config
# app.config.from_object(Config)  # For Flask
# or for FastAPI, use environment variables directly
