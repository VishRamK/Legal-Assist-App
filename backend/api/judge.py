from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch
import librosa
import numpy as np
import speech_recognition as sr
import openai
import os
from dotenv import load_dotenv
import assemblyai as aai

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLY_AI_KEY")
transcriber = aai.Transcriber()

#from ..services.llm_service import LLMService

class Judge:
    def __init__(self):
        self.total_responses_weight = 0   # Total weight of all responses
        self.num_responses = 0            # Number of responses
        self.response_weights = []        # Each response's weight
        self.responses = []               # Responses so far
        self.criticism = {}               # Feedback for bad responses
        self.merits = {}                  # Feedback for good responses
        self.current_response = None
        self.current_audio = None
        self.current_response_weight = None
        self.good_bound = 80              # Minimum weight for a good response
        self.bad_bound = 50               # Maximum weight for a bad response
        #self.llm = LLMService()           # Openai LLM

    def new_response(self, audio_file_path):
        self.current_audio = audio_file_path
        self.current_response = self.audio_to_text(audio_file_path)
        self.responses.append(self.current_response)
        self.num_responses += 1


    def evaluate_case(self, responses, evidence):
        """
        Evaluate the case based on performances and audio responses.
        
        performances: List of tuples representing good and bad performances [(performance_type, weight)]
        responses: List of audio responses [(response_text, response_audio)]
        evidence: Available evidence to compare against responses
        """
        return self.calculate_win_probability()
    

    def evaluate_response(self, evidence):
        """Evaluate audio responses based on multiple factors and update response weights."""
        if self.current_response:
            tone_confidence_weight = self.analyze_audio_tone()
            legal_strength_weight = self.analyze_legal_strength()
            correctness_weight = self.analyze_correctness(evidence)

            confidence_factor = 0.2
            strength_factor = 0.3
            correctness_factor = 0.5
            
            # Combine all factors into a total response weight
            total_response_weight = confidence_factor*tone_confidence_weight + strength_factor*legal_strength_weight + correctness_factor*correctness_weight
            self.total_responses_weight += total_response_weight
            self.current_response_weight = total_response_weight
            self.response_weights.append(total_response_weight)


    def audio_to_text(self, filename: str) -> str:
        """Convert audio file to text using SpeechRecognition."""
        return transcriber.transcribe(filename).text


    def analyze_audio_tone(self):
        # Load audio file
        y, sr = librosa.load(self.current_audio)

        # Extract pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Calculate average pitch
        average_pitch = min(np.mean(pitches[pitches > 0]), 1000)
        
        # Extract energy (root mean square energy)
        energy = librosa.feature.rms(y=y)
        average_energy = min(np.mean(energy), 0.1)

        # Extract duration of pauses (silences)
        silence_intervals = librosa.effects.split(y, top_db=20)  # Find non-silent intervals
        silence_durations = []
        previous_end = 0

        for start, end in silence_intervals:
            silence_duration = (start - previous_end) / sr
            if silence_duration > 0:  # Add only significant silences
                silence_durations.append(min(silence_duration, 2.5))
            previous_end = end
        
        average_silence_duration = np.mean(silence_durations) if silence_durations else 0
        optimal_pitch = 500
        optimal_energy = 0.05
        # Evaluate confidence based on features
        confidence_score = 100
        confidence_score -= 0.04*abs(optimal_pitch - average_pitch)
        confidence_score -= 800*abs(optimal_energy - average_energy)
        silence = max(0, average_silence_duration - 0.5)   # Example threshold for long pauses
        confidence_score -= 20*silence

        # Ensure confidence score is between 0 and 1
        confidence_score = max(0, min(confidence_score, 1))

        return confidence_score

    def analyze_legal_strength(self):
        """Use LegalBERT to evaluate the legal strength of a response."""
        print(self.current_response)
        if self.current_response == "Unintelligible":
            return 0

        tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")

        # Tokenize the response
        inputs = tokenizer(self.current_response, return_tensors="pt")
        outputs = model(**inputs)
        
        # Get logits (output before softmax)
        logits = outputs.logits[0]
        
        # Option 1: Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # Use the probability of the predicted class (highest probability)
        legal_strength_score = torch.max(probabilities).item() * 100  # Scale to 0-100

        return legal_strength_score


    def get_sentence_embedding(self, sentence, model, tokenizer):
        """Get the embedding of a sentence using BERT."""
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the mean of the output embeddings as the sentence embedding
        return torch.mean(outputs.last_hidden_state, dim=1)


    def analyze_correctness(self, evidence):
        """Evaluate correctness based on overall meaning using BERT embeddings."""
        if self.current_response == "Unintelligible":
            return 0
        
        # Load pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        # Get the sentence embeddings for the response and the evidence
        evidence_text = " ".join(evidence)  # Combine evidence into one string
        response_embedding = self.get_sentence_embedding(self.current_response, model, tokenizer)
        evidence_embedding = self.get_sentence_embedding(evidence_text, model, tokenizer)
        
        # Compute cosine similarity between the response and evidence embeddings
        cosine_sim = torch.nn.functional.cosine_similarity(response_embedding, evidence_embedding)
        
        # Convert similarity score to a percentage (0 to 100)
        correctness_score = cosine_sim.item() * 100
        
        return correctness_score
    

    def calculate_win_probability(self):
        """Calculate the win probability based on performances and responses."""

        # If there are no performances or responses, default to 50% probability
        if self.total_responses_weight == 0 and self.num_responses == 0:
            return 50

        win_probability = 100 * self.total_responses_weight / self.num_responses
        return min(max(win_probability, 0), 100)  # Ensure probability is between 0 and 100
    

# Helper function to interact with the judge class
def judge_deliberation(responses, evidence):
    """
    Function to handle the judge's deliberation.
    Performances is a list of tuples: (performance_type, weight)
    Responses is a list of tuples: (response_text, response_audio)
    Evidence is a list of evidence items (e.g., documents, facts)
    """
    judge = Judge()
    return judge.evaluate_case(responses, evidence)


responses = [
                ("Next page, please. Next page, please. Yes, I am familiar with this document.", "1.mp3"),
                ("I was the assistant director during that time.", "2.mp3"),
                ("I intended to inform the public and parents and students in the school about what our pedagogy entailed.", "3.mp3"),
                ("The sky is black.", "4.mp3"),
                ("I have not taken money from the department.", "5.mp3")    
            ]
evidence = ["Took a grant several years ago for research.", "Worked as a research assistant before changing industry.", "Leaked sensitive documents."]
print(judge_deliberation(responses, evidence))


    # def analyze_correctness(self, response_text, evidence, current_question):
    #     """Evaluate correctness based on evidence, consistency with previous responses, and relevance to the current question."""
        
    #     # Correctness based on evidence
    #     correctness_score_evidence = self.evaluate_evidence_consistency(response_text, evidence)
        
    #     # Consistency with previous responses
    #     consistency_score = self.evaluate_response_consistency(response_text)
        
    #     # Relevance to current question
    #     relevance_score = self.evaluate_relevance(response_text, current_question)
        
    #     # Combine the three scores, with appropriate weighting
    #     overall_correctness_score = (
    #         (correctness_score_evidence * 0.4) + 
    #         (consistency_score * 0.4) + 
    #         (relevance_score * 0.2)
    #     )
        
    #     # Store the current response for future consistency checks
    #     self.previous_responses.append(response_text)
        
    #     return overall_correctness_score

    # def normalize(claim):
    #     """Normalize the claim by converting to lowercase and stripping whitespace."""
    #     return claim.strip().lower()

    # def extract_features(claim):
    #     """Extract basic features such as keywords or phrases."""
    #     # This function can be extended to extract more sophisticated features.
    #     keywords = ['always', 'never', 'must', 'should', 'can', 'may', 'might', 'not']
    #     extracted_keywords = [word for word in keywords if word in claim]
    #     return extracted_keywords

    # def get_embedding(claim):
    #     """Get embedding for the claim from the LLM."""
    #     response = openai.Embedding.create(
    #         input=claim,
    #         model="text-embedding-ada-002"  # Use the appropriate embedding model
    #     )
    #     return response['data'][0]['embedding']

    # def query_llm(claim_a, claim_b):
    #     """Query the LLM to check for contradictions."""
    #     prompt = f"Do the following claims contradict each other? Claim A: '{claim_a}' Claim B: '{claim_b}'"
    #     response = openai.ChatCompletion.create(
    #         model="gpt-4",  # Use the appropriate model
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #     return response.choices[0].message['content']

    # def interpret_llm_response(response):
    #     """Interpret the LLM response to determine a contradiction score."""
    #     if "contradict" in response.lower():
    #         return 0.9  # High contradiction
    #     elif "compatible" in response.lower():
    #         return 0.1  # Low contradiction
    #     else:
    #         return 0.5  # Ambiguous case

    # def check_contradiction(claim_a, claim_b):
    #     """Main function to check if two claims contradict each other."""
    #     # Normalize claims
    #     claim_a_normalized = normalize(claim_a)
    #     claim_b_normalized = normalize(claim_b)

    #     # Get embeddings (optional, can be used for additional checks)
    #     embedding_a = get_embedding(claim_a_normalized)
    #     embedding_b = get_embedding(claim_b_normalized)

    #     # Query LLM for contradiction analysis
    #     llm_response = query_llm(claim_a_normalized, claim_b_normalized)

    #     # Interpret the LLM response
    #     contradiction_score = interpret_llm_response(llm_response)

    #     return {
    #         "contradicts": contradiction_score >= 0.8,
    #         "compatible": contradiction_score <= 0.2,
    #         "score": contradiction_score,
    #         "explanation": llm_response
    #     }
    

    # def evaluate_relevance(self, response_text, current_question):
    #     """Evaluate the relevance of the response to the current question."""
    #     # Check semantic similarity to the current question
    #     question_doc = self.nlp(current_question)
    #     response_doc = self.nlp(response_text)

    #     # Calculate semantic similarity
    #     similarity = question_doc.similarity(response_doc)

    #     # Convert to a score (0 to 100)
    #     relevance_score = similarity * 100
        
    #     return relevance_score
    
    # def generate_personalized_feedback(self, response_text, tone_confidence, legal_strength, correctness, context):
    #     """Generate personalized feedback using an LLM based on the trial point context."""
    #     # Example prompts tailored to different stages of the trial (opening statement, cross-examination, etc.)
    #     prompt = f"""
    #     You are a legal expert evaluating a mock trial performance. The specific context is '{context}'.
    #     The tone and confidence score is {tone_confidence}, the legal strength score is {legal_strength}, 
    #     and the correctness based on evidence is {correctness}.
    #     The performance was: "{response_text}".
        
    #     Provide detailed, constructive feedback specific to this context (e.g., cross-examination, opening statement), 
    #     highlighting strengths and areas for improvement.
    #     """
        
    #     # Placeholder for calling the LLM (e.g., OpenAI's GPT-4)
    #     response = self.call_llm(prompt)
        
    #     return response
    

    # def call_llm(self, prompt):
    #     """Call the LLM (e.g., GPT-4) to generate feedback based on the prompt."""
    #     # Replace with actual LLM API call
    #     # Example using OpenAI's ChatCompletion API:
    #     response = ChatCompletion.create(
    #         model="gpt-4",
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #     return response['choices'][0]['message']['content']
