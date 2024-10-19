class Judge:
    def __init__(self):
        self.evidence_strength = 0  # Initialize evidence strength
        self.argument_strength = 0  # Initialize argument strength
        self.total_questions = 0     # Total number of questions asked
        self.answered_questions = 0   # Number of questions successfully answered

    def evaluate_case(self, evidence, responses):
        """Evaluate the case based on evidence and responses."""
        self.evaluate_evidence(evidence)
        self.evaluate_responses(responses)
        return self.calculate_win_probability()

    def evaluate_evidence(self, evidence):
        """Evaluate the strength of the evidence."""
        # Simple logic for evidence evaluation (customize as needed)
        if evidence:
            self.evidence_strength = len(evidence) * 10  # Arbitrary scoring based on evidence length
        else:
            self.evidence_strength = 0

    def evaluate_responses(self, responses):
        """Evaluate the responses to questions."""
        self.total_questions = len(responses)
        for response in responses:
            if self.is_strong_response(response):
                self.answered_questions += 1

    def is_strong_response(self, response):
        """Determine if the response is strong."""
        # Define logic to assess the strength of the response (customize as needed)
        return "I don't know" not in response  # Example check

    def calculate_win_probability(self):
        """Calculate the win probability based on current evaluation."""
        if self.total_questions == 0:
            return 50  # Default to 50% if no questions were asked

        # Simple win probability formula
        response_strength = (self.answered_questions / self.total_questions) * 100
        probability = (self.evidence_strength + response_strength) / 2
        return min(max(probability, 0), 100)  # Ensure probability is between 0 and 100

# Helper function to interact with the judge class
def judge_deliberation(evidence, responses):
    """Function to handle the judge's deliberation."""
    judge = Judge()
    return judge.evaluate_case(evidence, responses)
