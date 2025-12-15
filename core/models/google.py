import time
import google.generativeai as genai
from .base import BaseModel

class GeminiModel(BaseModel):
    """
    Google Gemini 模型适配器。
    支持 Gemini 1.5 Pro, Flash 等模型。
    """
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        # 初始化 Google GenAI SDK
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def call(self, messages, temperature=0.0):
        """
        执行 Gemini API 调用。
        逻辑源自 engine.py 中的 GeminiEngine。
        """
        # 复用 engine.py 中的配置常量
        MAX_RETRIES = 6
        REQUEST_DELAY = 1
        MAX_TOKENS = 16384

        # 1. 转换消息格式 (OpenAI -> Gemini)
        gemini_messages = []
        
        # 提取 System Prompt (如果存在)
        if messages and messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
            # engine.py 中的 Hack: 通过 user/model 对话对来注入 system prompt
            gemini_messages.extend([
                {'role': 'user', 'parts': [system_prompt]},
                {'role': 'model', 'parts': ["OK, I will strictly follow these directives."]}
            ])
            conversation_messages = messages[1:]
        else:
            conversation_messages = messages

        # 转换剩余对话
        for msg in conversation_messages:
            # Gemini 使用 'user' 和 'model' 角色
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_messages.append({'role': role, 'parts': [msg['content']]})

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=MAX_TOKENS,
            temperature=temperature
        )

        # 2. 执行调用 (带重试机制)
        for attempt in range(MAX_RETRIES):
            try:
                # 直接使用 self.model 实例调用
                response = self.model.generate_content(
                    gemini_messages,
                    generation_config=generation_config
                )
                return response.text

            except Exception as e:
                print(f"[DEBUG] Gemini API调用尝试 {attempt + 1}/{MAX_RETRIES} 失败: {e}")
                
                # 最后一次尝试如果失败，抛出异常
                if attempt == MAX_RETRIES - 1:
                    raise e
                
                # 延时重试
                time.sleep(REQUEST_DELAY)
        
        return ""