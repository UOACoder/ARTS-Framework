import os
from .base import BaseModel

class ModelFactory:
    """
    模型工厂：负责根据模型名称字符串，实例化对应的模型适配器。
    并自动从环境变量中读取 API Key。
    """
    
    @staticmethod
    def create(model_name: str) -> BaseModel:
        model_name_lower = model_name.lower()

        # =========================================================
        # 1. Anthropic (Claude 系列)
        # =========================================================
        if "claude" in model_name_lower:
            # 延迟导入：防止未安装 anthropic 库时报错
            from .anthropic import AnthropicModel
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Environment variable 'ANTHROPIC_API_KEY' is not set.")
                
            return AnthropicModel(model_name=model_name, api_key=api_key)

        # =========================================================
        # 2. Google (Gemini 系列)
        # =========================================================
        elif "gemini" in model_name_lower:
            # 延迟导入：防止未安装 google-generativeai 库时报错
            from .google import GeminiModel
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Environment variable 'GOOGLE_API_KEY' is not set.")
                
            return GeminiModel(model_name=model_name, api_key=api_key)

        # =========================================================
        # 3. DeepSeek (OpenAI 协议兼容)
        # =========================================================
        elif "deepseek" in model_name_lower:
            from .openai_like import OpenAILikeModel
            
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("Environment variable 'DEEPSEEK_API_KEY' is not set.")
            
            return OpenAILikeModel(
                model_name=model_name, 
                api_key=api_key, 
                base_url="https://api.deepseek.com" # DeepSeek 官方端点
            )

        # =========================================================
        # 4. Default: OpenAI (GPT-4, o1, etc.)
        # =========================================================
        else:
            # 默认为 OpenAI 官方模型
            from .openai_like import OpenAILikeModel
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Environment variable 'OPENAI_API_KEY' is not set.")
                
            return OpenAILikeModel(model_name=model_name, api_key=api_key)