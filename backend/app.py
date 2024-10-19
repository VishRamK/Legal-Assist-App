from flask import Flask, request, jsonify, render_template
from config import Config
from models import init_models
from marshmallow import ValidationError
from models.request_models import DocumentRequestSchema, StrategyRequestSchema
from services import init_services
from services.document_service import DocumentService
from services.llm_service import LLMService
from api.document import upload_document
from api.strategy import generate_legal_strategy
from api.questioning import create_question
from api.judge import judge_deliberation

app = Flask(__name__)
app.config.from_object(Config)
init_models()
init_services()
document_service = DocumentService()
llm_service = LLMService(api_key='YOUR_API_KEY')  # Replace with your actual API key

# Define your API endpoints

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    """Route to handle document uploads."""
    if 'document' not in request.files:
        return jsonify({"error": "No document part in the request"}), 400
    
    file = request.files['document']
    try:
        content = document_service.upload_and_process_document(file)  # Use the service method
        return jsonify({"message": "Document uploaded successfully", "content": content}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/generate-strategy', methods=['POST'])
def generate_strategy():
    """Route to generate a legal strategy."""
    case_details = request.json.get('case_details')
    if not case_details:
        return jsonify({"error": "Case details are required"}), 400
    
    strategy = llm_service.generate_strategy(case_details)
    return jsonify({"strategy": strategy}), 200

@app.route('/ask-question', methods=['POST'])
def ask_question():
    """Route to ask a question."""
    context = request.json.get('context')
    if not context:
        return jsonify({"error": "Context is required"}), 400
    
    question = llm_service.ask_question(context)
    return jsonify({"question": question}), 200

@app.route('/judge-deliberation', methods=['POST'])
def judge_deliberation():
    """Route for judge deliberation."""
    case_summary = request.json.get('case_summary')
    if not case_summary:
        return jsonify({"error": "Case summary is required"}), 400
    
    deliberation = llm_service.judge_deliberation(case_summary)
    return jsonify({"deliberation": deliberation}), 200

# Error handling (optional)
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not Found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error"}), 500

# Helper functions (you can also define these in separate modules)
def extract_text_from_pdf(file):
    # Implement PDF extraction logic (e.g., using PyPDF2 or similar)
    pass

def calculate_win_probability():
    # Implement your logic to calculate win probability
    return 75  # Placeholder value

if __name__ == '__main__':
    app.run(debug=True)
