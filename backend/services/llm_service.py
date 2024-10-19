# services/llm_service.py

import openai  # Assuming you're using OpenAI's API

class LLMService:
    def __init__(self, api_key):
        """Initialize the LLM service with the API key."""
        openai.api_key = api_key  # Set your API key here

    def generate_strategy(self, case_details):
        """Generate a legal strategy based on case details."""
        prompt = f"Based on the following case details, generate a winning legal strategy:\n{case_details}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Choose the appropriate model
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']

    def ask_question(self, context):
        """Generate a question based on the context provided by the prosecutor or lawyer."""
        prompt = f"Generate a question based on the following context:\n{context}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']

    def judge_deliberation(self, case_summary):
        """Provide a judge's deliberation based on a case summary."""
        prompt = f"As a judge, provide a deliberation on the following case summary, including the probability of winning:\n{case_summary}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content()']
