# models/__init__.py

from ..api.strategy import StrategyGenerator  # Import Strategy model (if applicable)
from ..api.questioning import Questioning  # Import Question model (if applicable)
from ..api.judge import Judge  # Import Judge model (if applicable)

__all__ = [
    "StrategyGenerator",
    "Questioning",
    "Judge"
]

# Optional: Initialize any database connections or configurations if needed
def init_models():
    """Initialize the models if needed (e.g., database connection)."""
    pass  # Add your initialization logic here
