from .base import BaseModel
from .factory import ModelFactory

# 仅暴露基类和工厂，强制使用工厂模式创建实例
__all__ = [
    "BaseModel",
    "ModelFactory"
]