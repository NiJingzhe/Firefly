"""
屏幕截图工具
提供屏幕截图功能，支持全屏或指定区域截图
"""

import os
import tempfile
from typing import Optional, Tuple
from SimpleLLMFunc.tool import tool
from SimpleLLMFunc.llm_decorator.multimodal_types import ImgPath

try:
    import pyautogui
    from PIL import ImageGrab
    SCREENSHOT_AVAILABLE = True
except ImportError:
    SCREENSHOT_AVAILABLE = False


@tool(name="capture_screen", description="截取屏幕截图")
def capture_screen() -> tuple[str, ImgPath]:
    """
    截取屏幕截图并返回图片路径
        
    Returns:
        截图文件的路径
        
    Raises:
        RuntimeError: 当截图功能不可用时抛出
    """
    if not SCREENSHOT_AVAILABLE:
        raise RuntimeError("屏幕截图功能不可用。请安装依赖：pip install pyautogui pillow")
    
    try:
        screenshot = ImageGrab.grab()
    
        # 使用临时文件
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix='.png', 
            prefix='screenshot_'
        )
        file_path = temp_file.name
        temp_file.close()
        
        # 保存截图
        screenshot.save(file_path, 'PNG')
        
        return "这是截图的结果： ", ImgPath(file_path)
        
    except Exception as e:
        raise RuntimeError(f"截图失败: {str(e)}") from e


