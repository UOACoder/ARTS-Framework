from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional


class BaseModel(ABC):
    """
    Abstract base class for all LLM wrappers in the ARTS framework.
    Enforces a consistent interface for generating traces and reasoning.
    """

    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs

    @abstractmethod
    def call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Execute a chat completion call.

        Args:
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "..."}])

        Returns:
            The raw string response from the model.
        """
        pass

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Optional hook for model-specific response parsing.
        Base implementation returns None, relying on the generic parser.
        """
        # Child classes can override this if they have special output formats
        return None
