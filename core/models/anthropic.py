import time
import anthropic
from typing import List, Dict, Optional, Any
from .base import BaseModel


class AnthropicModel(BaseModel):
    """
    Adapter for Anthropic Claude models with JSON pre-fill support.
    """

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.client = anthropic.Anthropic(api_key=api_key)

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:

        MAX_RETRIES = 6
        REQUEST_DELAY = 1
        MAX_TOKENS = 16384

        # Extract system prompt as top-level parameter
        system_prompt = (
            messages[0]["content"]
            if messages and messages[0]["role"] == "system"
            else ""
        )
        active_messages = messages[1:] if system_prompt else messages

        # JSON Pre-fill: Force Assistant to start with "{" to enforce validity
        should_prefill = (
            response_format and response_format.get("type") == "json_object"
        )
        messages_to_send = [msg.copy() for msg in active_messages]

        if should_prefill:
            messages_to_send.append({"role": "assistant", "content": "{"})

        for attempt in range(MAX_RETRIES):
            try:
                with self.client.messages.stream(
                    model=self.model_name,
                    system=system_prompt,
                    messages=messages_to_send,
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                ) as stream:
                    for _ in stream.text_stream:
                        pass
                    final_message = stream.get_final_message()

                content = final_message.content[0].text

                # Reconstruct valid JSON if pre-filled
                return "{" + content if should_prefill else content

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                time.sleep(REQUEST_DELAY)

        return ""
