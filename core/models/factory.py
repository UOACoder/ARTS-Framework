import os
from .base import BaseModel


class ModelFactory:
    """
    Factory class to instantiate the appropriate model adapter based on the model name.
    """

    @staticmethod
    def create(model_name: str) -> BaseModel:
        model_name_lower = model_name.lower()

        if "claude" in model_name_lower:
            from .anthropic import AnthropicModel

            return AnthropicModel(model_name, _get_key("ANTHROPIC_API_KEY"))

        elif "gemini" in model_name_lower:
            from .google import GeminiModel

            return GeminiModel(model_name, _get_key("GOOGLE_API_KEY"))

        elif "deepseek" in model_name_lower:
            from .openai_like import OpenAILikeModel

            return OpenAILikeModel(
                model_name=model_name,
                api_key=_get_key("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )

        else:
            # Default to OpenAI
            from .openai_like import OpenAILikeModel

            return OpenAILikeModel(model_name, _get_key("OPENAI_API_KEY"))


def _get_key(env_var: str) -> str:
    key = os.getenv(env_var)
    if not key:
        raise ValueError(f"Environment variable '{env_var}' is missing.")
    return key
