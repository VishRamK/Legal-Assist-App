import openai
import PyPDF2
import os

# OpenAI API setup (replace with your API key)
openai.api_key = 'sk-proj-OWK5pyZahlATWnG4TX25T3BlbkFJnKXDAARC550AF1KqdkRz'

def get_case_brief_facts(transcript):
    prompt = f"""
    Extract a very concise case brief from the following case brief . In one paragraph,the case brief should include:
    
    1. Facts of the case
    2. Issues
    
    3. Reasoning
   
    This will be read by the judge as the proceedings start.keep the numbers like timeline and cost in the output.
    Transcript:
    {transcript}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    case_brief = response['choices'][0]['message']['content']
    return case_brief

def process_file_for_case_brief(file_path, file_type):
    transcript_text = ""
    
    if file_type == "pdf":
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                transcript_text += page.extract_text()
    elif file_type == "txt":
        with open(file_path, "r") as file:
            transcript_text = file.read()
    else:
        raise ValueError("Unsupported file type. Please use pdf, docx, or txt files.")
    
    # Get case brief from the transcript using OpenAI API
    case_brief = get_case_brief_facts(transcript_text)
    
    # Print or return the case brief
    print("Case Brief Extracted:")
    print(case_brief)

# Example usage
file_path_pdf = "backend/api/Documents/Case1_brief.pdf"

process_file_for_case_brief(file_path_pdf, "pdf")

