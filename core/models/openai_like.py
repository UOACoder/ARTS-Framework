from openai import OpenAI
import time
from typing import List, Dict, Optional, Any
from .base import BaseModel

class OpenAILikeModel(BaseModel):
    """
    OpenAI 协议兼容模型适配器。
    支持: OpenAI (GPT-4), DeepSeek, Moonshot (Kimi), Qwen, vLLM, Ollama 等。
    """
    
    def __init__(self, model_name: str, api_key: str, base_url: str = None):
        super().__init__(model_name, api_key)
        # 如果传入 base_url (如 DeepSeek)，则使用该地址；否则默认连 OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def call(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.0,
             response_format: Optional[Dict[str, Any]] = None) -> str:
        """
        执行 API 调用，包含来自 engine.py 的重试逻辑和参数配置。
        """
        # 复用 engine.py 中的配置常量
        MAX_RETRIES = 6
        REQUEST_DELAY = 1
        MAX_TOKENS = 16384  # 你原代码中写死的数值

        for attempt in range(MAX_RETRIES):
            try:
                # 严格对应原 OpenAIEngine.call 中的参数传递
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    # [关键] 严格保留你 engine.py 中的参数名 max_completion_tokens
                    # 注意：部分非 OpenAI 模型可能需要改回 max_tokens，但在你的环境中我保持原样
                    max_completion_tokens=MAX_TOKENS, 
                    response_format=response_format
                )
                
                # 返回 content
                return response.choices[0].message.content

            except Exception as e:
                print(f"[DEBUG] API调用尝试 {attempt + 1}/{MAX_RETRIES} 失败: {e}")
                
                # 最后一次尝试如果失败，抛出异常供上层处理
                if attempt == MAX_RETRIES - 1:
                    raise e
                
                # 延时重试
                time.sleep(REQUEST_DELAY)
        
        return ""