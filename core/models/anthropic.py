import time
import anthropic
from typing import List, Dict, Optional, Any
from .base import BaseModel

class AnthropicModel(BaseModel):
    """
    Anthropic Claude 模型适配器。
    支持 Claude 3.5 Sonnet, Opus, Haiku 等。
    """
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.client = anthropic.Anthropic(api_key=api_key)

    def call(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.0, 
             response_format: Optional[Dict[str, Any]] = None) -> str:
        """
        执行 Claude API 调用。
        包含核心的 JSON Pre-fill (预填) 逻辑，源自原 engine.py。
        """
        # 复用配置常量
        MAX_RETRIES = 6
        REQUEST_DELAY = 1
        MAX_TOKENS = 16384

        # 1. 提取 System Prompt (Anthropic API 要求将其作为顶层参数)
        system_prompt = ""
        active_messages = []
        
        # 检查第一条消息是否是 system
        if messages and messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
            active_messages = messages[1:]
        else:
            active_messages = messages

        # 2. 处理 JSON Pre-fill 逻辑 (原 AnthropicEngine 的核心特性)
        # 如果要求 JSON 格式，我们在消息末尾强制由 Assistant 说出第一个大括号 "{"
        # 这会强迫 Claude 直接进入 JSON 生成模式，跳过 "Here is..." 等废话
        should_prefill = (response_format and response_format.get("type") == "json_object")
        
        # 深拷贝消息列表，避免污染上层数据
        messages_to_send = [msg.copy() for msg in active_messages]
        
        if should_prefill:
            messages_to_send.append({"role": "assistant", "content": "{"})

        # 3. 执行调用 (带重试机制)
        for attempt in range(MAX_RETRIES):
            try:
                # 使用 stream 模式 (保持原 engine.py 的实现方式)
                with self.client.messages.stream(
                    model=self.model_name,
                    system=system_prompt,
                    messages=messages_to_send,
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                ) as stream:
                    # 消耗流以获取完整响应
                    for _ in stream.text_stream:
                        pass
                    final_message = stream.get_final_message()
                
                raw_content = final_message.content[0].text
                
                # 4. 补全逻辑
                # 因为我们帮它说了 "{"，API 返回的内容里就不包含这个 "{" 了
                # 我们需要把它补回去，否则 JSON 解析会失败
                if should_prefill:
                    raw_content = "{" + raw_content
                    
                return raw_content

            except Exception as e:
                print(f"[DEBUG] Claude API调用尝试 {attempt + 1}/{MAX_RETRIES} 失败: {e}")
                
                # 最后一次尝试如果失败，抛出异常
                if attempt == MAX_RETRIES - 1:
                    raise e
                
                # 延时重试
                time.sleep(REQUEST_DELAY)
        
        return ""