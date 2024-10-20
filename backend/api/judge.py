from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel
import torch
import librosa
import numpy as np
from datasets import load_dataset
import os
from dotenv import load_dotenv
import assemblyai as aai

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLY_AI_KEY")
transcriber = aai.Transcriber()

# Path to your fine-tuned model
finetuned_model_path = './finetuned_legal_bert'

model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")

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

    def new_response(self, question, audio_file_path):
        self.current_audio = audio_file_path
        self.responses.append(self.current_response)
        self.current_response = (question, self.audio_to_text(audio_file_path))
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
        if self.current_response[1] == "Unintelligible":
            return 0

        tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")

        # Tokenize the response
        inputs = tokenizer(self.current_response[1], return_tensors="pt")
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
        response_embedding = self.get_sentence_embedding(self.current_response[1], model, tokenizer)
        question_embedding = self.get_sentence_embedding(self.current_response[0], model, tokenizer)
        evidence_embedding = self.get_sentence_embedding(evidence_text, model, tokenizer)
        
        # Compute cosine similarity between the response and evidence embeddings
        cosine_sim_e = torch.nn.functional.cosine_similarity(response_embedding, evidence_embedding)
        cosine_sim_q = torch.nn.functional.cosine_similarity(response_embedding, question_embedding)
        
        # Convert similarity score to a percentage (0 to 100)
        correctness_score = np.mean([cosine_sim_e.item(), cosine_sim_q.item()]) * 100
        
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