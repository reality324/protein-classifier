"""编码方式模块 - 插件式架构，支持轻松扩展新编码

默认注册:
- onehot: One-Hot 编码 (20维)
- ctd: CTD 编码 (147维)
- esm2: ESM2 预训练嵌入 (480维)

添加新编码:
1. 在本目录创建新文件, 如 my_encoding.py
2. 继承 ProteinEncoder 基类
3. 使用 @register_encoder("my_encoding") 装饰器
4. 自动被本模块加载
"""
from .base import (
    ProteinEncoder,
    EncoderRegistry,
    register_encoder,
)

# 自动注册所有内置编码器
EncoderRegistry.load_builtin_encodings()

# 方便直接导入
__all__ = [
    "ProteinEncoder",
    "EncoderRegistry",
    "register_encoder",
    "OneHotEncoder",
    "CTDEncoder",
    "ESM2Encoder",
]

# 为了向后兼容，也导出具体类
from .onehot import OneHotEncoder
from .ctd import CTDEncoder
from .esm2 import ESM2Encoder
