# evaluation/models/base.py
from abc import ABC, abstractmethod
from typing import List, Dict

# [修正] 从同级目录 metrics 导入，或者从包路径导入
from evaluation.metrics import Evaluator 

class BaseModel(ABC):
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def call(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """核心接口：返回纯文本"""
        pass
    
    def parse_response(self, raw_response: str) -> Dict:
        """
        通用解析逻辑，子类可以覆盖它。
        """
        return Evaluator.parse_response(raw_response)