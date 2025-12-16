import time
import google.generativeai as genai
from .base import BaseModel


class GeminiModel(BaseModel):
    """
    Adapter for Google Gemini models via GenAI SDK.
    """

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def call(self, messages, temperature=0.0):
        MAX_RETRIES = 6
        REQUEST_DELAY = 1

        # Convert OpenAI-style messages to Gemini format
        gemini_messages = []
        conversation_start_idx = 0

        # Handle System Prompt: Inject via explicit user/model turns
        if messages and messages[0]["role"] == "system":
            gemini_messages.extend(
                [
                    {"role": "user", "parts": [messages[0]["content"]]},
                    {"role": "model", "parts": ["Understood."]},
                ]
            )
            conversation_start_idx = 1

        for msg in messages[conversation_start_idx:]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        config = genai.types.GenerationConfig(
            max_output_tokens=16384, temperature=temperature
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = self.model.generate_content(
                    gemini_messages, generation_config=config
                )
                return response.text
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                time.sleep(REQUEST_DELAY)

        return ""
