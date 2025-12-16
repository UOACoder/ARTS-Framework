from openai import OpenAI
import time
from typing import List, Dict, Optional, Any
from .base import BaseModel


class OpenAILikeModel(BaseModel):
    """
    Adapter for OpenAI-compatible APIs (GPT-4, DeepSeek, vLLM, etc.).
    """

    def __init__(self, model_name: str, api_key: str, base_url: str = None):
        super().__init__(model_name, api_key)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:

        MAX_RETRIES = 6
        REQUEST_DELAY = 1
        MAX_TOKENS = 16384

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=MAX_TOKENS,
                    response_format=response_format,
                )
                return response.choices[0].message.content

            except Exception as e:
                # Simple exponential backoff or retry logic could be added here
                if attempt == MAX_RETRIES - 1:
                    raise e
                time.sleep(REQUEST_DELAY)

        return ""
