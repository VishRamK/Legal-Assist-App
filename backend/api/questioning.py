import random

class Questioning:
    def __init__(self):
        self.questions = self.load_questions()  # Load initial questions from a predefined source

    def load_questions(self):
        """Load a set of predefined questions for the mock trial."""
        # In a real application, this could load from a database or file
        return [
            "Can you explain your whereabouts on the night of the incident?",
            "What evidence do you have to support your claims?",
            "Why should we believe your version of events?",
            "How do you respond to the allegations made against you?",
            "What is your relationship with the witness?",
            "Is there anyone who can corroborate your story?",
        ]

    def generate_question(self, previous_responses):
        """Generate a follow-up question based on previous responses."""
        # Example logic to select a follow-up question
        if previous_responses:
            last_response = previous_responses[-1]
            if self.is_weak_response(last_response):
                return self.reinforce_with_follow_up(last_response)
        
        # Return a random question if no specific follow-up is needed
        return random.choice(self.questions)

    def is_weak_response(self, response):
        """Determine if the response is weak and may require further questioning."""
        # Define criteria for a weak response (customize as needed)
        return "I don't know" in response or len(response.split()) < 5

    def reinforce_with_follow_up(self, last_response):
        """Generate a more pointed follow-up question based on the last weak response."""
        # Customize follow-up questions based on the last response
        follow_up_questions = [
            "Could you provide more details about that?",
            "Why do you think that is the case?",
            "What evidence do you have to counter that point?",
            "Can you clarify what you meant by that?",
        ]
        return random.choice(follow_up_questions)

# Helper function to interact with the questioning class
def create_question(previous_responses):
    """Function to handle the questioning process."""
    questioning = Questioning()
    return questioning.generate_question(previous_responses)
