class StrategyGenerator:
    def __init__(self):
        self.base_strategies = self.load_strategies()  # Load initial strategies

    def load_strategies(self):
        """Load a set of predefined strategies for the mock trial."""
        # In a real application, this could load from a database or file
        return {
            "defensive": [
                "Stick to your alibi and provide as much detail as possible.",
                "Focus on discrediting the opposing evidence.",
                "Keep emotions in check and remain calm during questioning."
            ],
            "offensive": [
                "Highlight any weaknesses in the prosecutor's case.",
                "Use character witnesses to bolster your credibility.",
                "Present alternative explanations for the evidence presented."
            ],
            "neutral": [
                "Acknowledge the evidence but provide context.",
                "Emphasize that the burden of proof is on the prosecution.",
                "Stay factual and avoid speculation."
            ]
        }

    def generate_strategy(self, evidence, previous_responses):
        """Generate a strategy based on evidence and previous responses."""
        strategy_type = self.analyze_case(evidence, previous_responses)
        return self.base_strategies.get(strategy_type, [])

    def analyze_case(self, evidence, previous_responses):
        """Analyze the case to determine the best type of strategy."""
        # Simplistic analysis logic (customize as needed)
        if evidence and len(previous_responses) < 3:
            return "defensive"
        elif evidence and self.is_strength_in_responses(previous_responses):
            return "offensive"
        else:
            return "neutral"

    def is_strength_in_responses(self, previous_responses):
        """Determine if there's strength in previous responses."""
        # Define logic to assess the strength of previous responses (customize as needed)
        return any("strong" in response for response in previous_responses)

# Helper function to interact with the strategy generator
def generate_legal_strategy(evidence, previous_responses):
    """Function to handle strategy generation."""
    strategy_gen = StrategyGenerator()
    return strategy_gen.generate_strategy(evidence, previous_responses)
