"""
屏幕截图工具
提供屏幕截图功能，支持全屏或指定区域截图
"""

import os
import tempfile
import json
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from SimpleLLMFunc.tool import tool
from SimpleLLMFunc.llm_decorator.multimodal_types import ImgPath
from paddleocr import PaddleOCR
from config.config import get_config
from PIL import Image, ImageDraw, ImageFont
from .common import print_tool_output

config = get_config()   

try:
    import pyautogui
    from PIL import ImageGrab
    SCREENSHOT_AVAILABLE = True
except ImportError:
    SCREENSHOT_AVAILABLE = False


ocr = PaddleOCR(
    text_detection_model_dir=config.PPOCR_DET_MODEL_DIR,
    text_recognition_model_dir=config.PPOCR_REC_MODEL_DIR,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
) # 更换 PP-OCRv5_server 模型



@tool(name="capture_screen", description="截取屏幕截图")
def capture_screen() -> str:
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
        
        # 保持比例的缩放到1080p
        width, height = screenshot.size
        target_width = 1920
        target_height = int((target_width / width) * height)    
        
        screenshot = screenshot.resize((target_width, target_height), Image.Resampling.LANCZOS)    
     
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

        print_tool_output(title="截图保存成功", content=f"📷 截图已保存到临时文件: {file_path}, 开始OCR识别")

        # 使用predict_iter生成器方式实现边识别边处理
        result_iter = ocr.predict_iter(file_path)
        
        # 处理OCR结果
        processed_result = _process_ocr_result_iter(result_iter, file_path)

        print_tool_output(title="OCR识别完成", content="处理结果如下：")
        print_tool_output(title="OCR识别结果", content=processed_result["text_result"])

        return processed_result["text_result"]
        
    except Exception as e:
        raise RuntimeError(f"截图失败: {str(e)}") from e


def _calculate_bbox_area(bbox: np.ndarray) -> float:
    """计算边界框面积"""
    # bbox格式: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # 使用向量叉积计算多边形面积
    x = bbox[:, 0]
    y = bbox[:, 1]
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))


def _get_bbox_top_left(bbox: np.ndarray) -> Tuple[int, int]:
    """获取边界框左上角坐标"""
    x_min = int(np.min(bbox[:, 0]))
    y_min = int(np.min(bbox[:, 1]))
    return x_min, y_min


def _process_ocr_result(ocr_result_list, original_image_path: str) -> Dict[str, Any]:
    """
    处理OCR结果：
    1. 找到文本和bounding box的对应关系
    2. 按bbox面积从大到小排序
    3. 选取前50%的结果
    4. 绘制标注图像和生成JSON结果
    """
    # OCR结果是一个列表，遍历所有结果
    if not ocr_result_list:
        return {
            "text_result": "OCR未识别到任何文本",
            "image_path": original_image_path
        }
    
    # 收集所有OCR结果中的文本和边界框
    all_texts = []
    all_polys = []
    all_scores = []
    
    for ocr_result in ocr_result_list:
        # 获取OCR结果的JSON数据
        ocr_json = ocr_result.json
        
        # 从 'res' 字段中提取实际结果
        res_data = ocr_json.get('res', {})
        
        # 提取文本、边界框和置信度
        rec_texts = res_data.get('rec_texts', [])
        rec_polys = res_data.get('rec_polys', [])
        rec_scores = res_data.get('rec_scores', [])
        
        # 合并到总列表中
        all_texts.extend(rec_texts)
        all_polys.extend(rec_polys)
        all_scores.extend(rec_scores)
    
    if not all_texts or not all_polys:
        return {
            "text_result": "OCR未识别到任何文本",
            "image_path": original_image_path
        }
    
    # 创建文本-边界框对应关系列表
    text_bbox_pairs = []
    for i, (text, bbox_list, score) in enumerate(zip(all_texts, all_polys, all_scores)):
        # 将列表转换为numpy数组
        bbox = np.array(bbox_list, dtype=np.int32)
        area = _calculate_bbox_area(bbox)
        top_left = _get_bbox_top_left(bbox)
        
        text_bbox_pairs.append({
            'index': i,
            'text': text,
            'bbox': bbox,
            'area': area,
            'top_left': top_left,
            'score': score
        })
    
    # 合并相近的边界框
    print_tool_output(title="边界框合并", content=f"合并前有 {len(text_bbox_pairs)} 个文本框")
    merged_pairs = _merge_nearby_bboxes(text_bbox_pairs, distance_threshold=80.0)
    print_tool_output(title="边界框合并完成", content=f"合并后有 {len(merged_pairs)} 个文本框")
    
    # 按位置排序：先按Y坐标（从上到下），再按X坐标（从左到右）
    merged_pairs.sort(key=lambda x: (x['top_left'][1], x['top_left'][0]))
    
    # 选取前50%的结果（按面积）
    merged_pairs_by_area = sorted(merged_pairs, key=lambda x: x['area'], reverse=True)
    top_50_percent = merged_pairs_by_area[:max(1, len(merged_pairs_by_area) // 3)]
    
    # 将选中的结果重新按位置排序
    top_50_percent.sort(key=lambda x: (x['top_left'][1], x['top_left'][0]))
    
    # 生成标注图像
    annotated_image_path = _create_annotated_image(original_image_path, top_50_percent)
    
    # 生成JSON结果
    json_result = []
    for i, item in enumerate(top_50_percent, 1):
        json_result.append({
            'id': i,
            'text': item['text'],
            'top_left_x': item['top_left'][0],
            'top_left_y': item['top_left'][1],
            'area': item['area'],
            'confidence': item['score']
        })
    
    # 构造文本结果
    text_result = f"OCR识别结果（前50%，共{len(top_50_percent)}个）:\n"
    text_result += json.dumps(json_result, ensure_ascii=False, indent=2)
    
    return {
        "text_result": text_result,
        "image_path": annotated_image_path
    }


def _process_ocr_result_iter(ocr_result_iter, original_image_path: str) -> Dict[str, Any]:
    """
    处理OCR迭代器结果：
    1. 边识别边处理，逐步收集文本和bounding box
    2. 按bbox面积从大到小排序
    3. 选取前50%的结果
    4. 绘制标注图像和生成JSON结果
    """
    # 收集所有OCR结果中的文本和边界框
    all_texts = []
    all_polys = []
    all_scores = []
    
    print_tool_output(title="OCR识别", content="开始逐步处理OCR结果...")
    
    # 使用迭代器逐步处理结果
    try:
        for i, ocr_result in enumerate(ocr_result_iter):
            # 获取OCR结果的JSON数据
            ocr_json = ocr_result.json
            
            # 从 'res' 字段中提取实际结果
            res_data = ocr_json.get('res', {})
            
            # 提取文本、边界框和置信度
            rec_texts = res_data.get('rec_texts', [])
            rec_polys = res_data.get('rec_polys', [])
            rec_scores = res_data.get('rec_scores', [])
            
            # 合并到总列表中
            all_texts.extend(rec_texts)
            all_polys.extend(rec_polys)
            all_scores.extend(rec_scores)
            
            # 显示处理进度
            if rec_texts:
                print_tool_output(title=f"处理批次{i+1}", content=f"识别到 {len(rec_texts)} 个文本")
    
    except Exception as e:
        print_tool_output(title="OCR处理警告", content=f"迭代器处理出现异常: {str(e)}")
        # 如果迭代器出现问题，继续处理已收集的结果
    
    if not all_texts or not all_polys:
        return {
            "text_result": "OCR未识别到任何文本",
            "image_path": original_image_path
        }
    
    print_tool_output(title="OCR识别完成", content=f"总共识别到 {len(all_texts)} 个文本")
    
    # 创建文本-边界框对应关系列表
    text_bbox_pairs = []
    for i, (text, bbox_list, score) in enumerate(zip(all_texts, all_polys, all_scores)):
        # 将列表转换为numpy数组
        bbox = np.array(bbox_list, dtype=np.int32)
        area = _calculate_bbox_area(bbox)
        top_left = _get_bbox_top_left(bbox)
        
        text_bbox_pairs.append({
            'index': i,
            'text': text,
            'bbox': bbox,
            'area': area,
            'top_left': top_left,
            'score': score
        })
    
    # 合并相近的边界框
    print_tool_output(title="边界框合并", content=f"合并前有 {len(text_bbox_pairs)} 个文本框")
    merged_pairs = _merge_nearby_bboxes(text_bbox_pairs, distance_threshold=80.0)
    print_tool_output(title="边界框合并完成", content=f"合并后有 {len(merged_pairs)} 个文本框")
    
    # 按位置排序：先按Y坐标（从上到下），再按X坐标（从左到右）
    merged_pairs.sort(key=lambda x: (x['top_left'][1], x['top_left'][0]))
    
    # 选取前50%的结果（按面积）
    merged_pairs_by_area = sorted(merged_pairs, key=lambda x: x['area'], reverse=True)
    top_50_percent = merged_pairs_by_area[:max(1, len(merged_pairs_by_area) // 4)]
    
    # 将选中的结果重新按位置排序
    top_50_percent.sort(key=lambda x: (x['top_left'][1], x['top_left'][0]))
    
    # 生成标注图像
    annotated_image_path = _create_annotated_image(original_image_path, top_50_percent)
    
    # 生成JSON结果
    json_result = []
    for i, item in enumerate(top_50_percent, 1):
        json_result.append({
            'id': i,
            'text': item['text'],
            'top_left_x': item['top_left'][0],
            'top_left_y': item['top_left'][1],
            'area': item['area'],
            'confidence': item['score']
        })
    
    # 构造文本结果
    text_result = f"OCR识别结果（前50%，共{len(top_50_percent)}个）:\n"
    text_result += json.dumps(json_result, ensure_ascii=False, indent=2)
    
    return {
        "text_result": text_result,
        "image_path": annotated_image_path
    }


def _create_annotated_image(original_image_path: str, text_bbox_pairs: List[Dict]) -> str:
    """创建带标注的图像"""
    # 打开原始图像
    image = Image.open(original_image_path)
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体，如果失败则使用默认字体
    try:
        # 在macOS上尝试加载系统字体
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 24)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # 绘制边界框和标号
    for i, item in enumerate(text_bbox_pairs, 1):
        bbox = item['bbox']
        
        # 绘制边界框
        # 将bbox转换为PIL可用的格式
        bbox_points = [(int(point[0]), int(point[1])) for point in bbox]
        draw.polygon(bbox_points, outline='red', width=2)
        
        # 在左上角绘制标号
        top_left = item['top_left']
        
        # 绘制标号背景
        text_bbox = draw.textbbox((top_left[0], top_left[1] - 30), str(i), font=font)
        draw.rectangle(text_bbox, fill='red')
        
        # 绘制标号文字
        draw.text((top_left[0], top_left[1] - 30), str(i), fill='white', font=font)
    
    # 保存标注图像
    annotated_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix='_annotated.png',
        prefix='screenshot_'
    )
    annotated_path = annotated_file.name
    annotated_file.close()
    
    image.save(annotated_path, 'PNG')
    return annotated_path


def _calculate_bbox_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """计算两个边界框的最小距离"""
    # 获取两个边界框的中心点
    center1 = np.mean(bbox1, axis=0)
    center2 = np.mean(bbox2, axis=0)
    
    # 计算边界框的最小外接矩形
    x1_min, y1_min = np.min(bbox1, axis=0)
    x1_max, y1_max = np.max(bbox1, axis=0)
    x2_min, y2_min = np.min(bbox2, axis=0)
    x2_max, y2_max = np.max(bbox2, axis=0)
    
    # 计算矩形间的距离
    dx = max(0, max(x1_min - x2_max, x2_min - x1_max))
    dy = max(0, max(y1_min - y2_max, y2_min - y1_max))
    
    return np.sqrt(dx*dx + dy*dy)


def _merge_bboxes(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """合并两个边界框，返回包含两者的最小边界框"""
    all_points = np.vstack([bbox1, bbox2])
    
    # 找到最小外接矩形的四个角点
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    
    # 返回矩形的四个角点
    return np.array([
        [x_min, y_min],
        [x_max, y_min], 
        [x_max, y_max],
        [x_min, y_max]
    ], dtype=np.int32)


def _is_vertical_merge(bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
    """判断两个边界框是否应该进行垂直方向的合并（上下排列）"""
    # 获取边界框的边界和中心点
    y1_min, y1_max = np.min(bbox1[:, 1]), np.max(bbox1[:, 1])
    y2_min, y2_max = np.min(bbox2[:, 1]), np.max(bbox2[:, 1])
    x1_min, x1_max = np.min(bbox1[:, 0]), np.max(bbox1[:, 0])
    x2_min, x2_max = np.min(bbox2[:, 0]), np.max(bbox2[:, 0])
    
    # 计算中心点
    center1_y = (y1_min + y1_max) / 2
    center2_y = (y2_min + y2_max) / 2
    
    # 计算高度
    height1 = y1_max - y1_min
    height2 = y2_max - y2_min
    
    # 计算垂直距离
    vertical_distance = abs(center1_y - center2_y)
    
    # 计算水平重叠
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    width1 = x1_max - x1_min
    width2 = x2_max - x2_min
    
    if width1 == 0 or width2 == 0:
        return False
    
    # 计算水平重叠比例（相对于任一个bbox的宽度）
    overlap_ratio1 = x_overlap / width1
    overlap_ratio2 = x_overlap / width2
    max_overlap_ratio = max(overlap_ratio1, overlap_ratio2)
    
    # 垂直合并条件：
    # 1. 垂直距离小于任一bbox高度的1.1倍
    # 2. 水平重叠超过任一bbox宽度的70%
    height_threshold = max(height1, height2) * 1.1
    
    return (vertical_distance < height_threshold and 
            max_overlap_ratio > 0.3)


def _merge_nearby_bboxes(text_bbox_pairs: List[Dict], distance_threshold: float = 50.0) -> List[Dict]:
    """合并相近的边界框（仅垂直方向）"""
    if len(text_bbox_pairs) <= 1:
        return text_bbox_pairs
    
    merged_pairs = []
    used_indices = set()
    
    for i, pair1 in enumerate(text_bbox_pairs):
        if i in used_indices:
            continue
            
        # 初始化合并组
        merge_group = [pair1]
        merge_indices = {i}
        
        # 查找需要合并的边界框（仅考虑垂直方向）
        for j, pair2 in enumerate(text_bbox_pairs):
            if j in used_indices or j <= i:
                continue
                
            # 首先检查是否符合垂直合并条件
            is_vertical = _is_vertical_merge(pair1['bbox'], pair2['bbox'])
            if not is_vertical:
                continue
                
            # 计算与组中任一边界框的最小距离
            min_distance = float('inf')
            for group_pair in merge_group:
                dist = _calculate_bbox_distance(group_pair['bbox'], pair2['bbox'])
                min_distance = min(min_distance, dist)
            
            # 如果距离小于阈值且是垂直排列，加入合并组
            if min_distance <= distance_threshold:
                merge_group.append(pair2)
                merge_indices.add(j)
        
        # 标记已使用的索引
        used_indices.update(merge_indices)
        
        if len(merge_group) == 1:
            # 没有需要合并的，直接添加
            merged_pairs.append(pair1)
        else:
            # 合并多个边界框（按Y坐标排序）
            merge_group_sorted = sorted(merge_group, key=lambda x: np.mean(x['bbox'][:, 1]))
            
            merged_bbox = merge_group_sorted[0]['bbox']
            merged_texts = [merge_group_sorted[0]['text']]
            merged_scores = [merge_group_sorted[0]['score']]
            
            for group_pair in merge_group_sorted[1:]:
                merged_bbox = _merge_bboxes(merged_bbox, group_pair['bbox'])
                merged_texts.append(group_pair['text'])
                merged_scores.append(group_pair['score'])
            
            # 创建合并后的项
            merged_area = _calculate_bbox_area(merged_bbox)
            merged_top_left = _get_bbox_top_left(merged_bbox)
            
            # 垂直合并用换行符连接
            merged_text = '\n'.join(merged_texts)
            merged_score = np.mean(merged_scores)  # 平均置信度
            
            merged_pairs.append({
                'index': len(merged_pairs),
                'text': merged_text,
                'bbox': merged_bbox,
                'area': merged_area,
                'top_left': merged_top_left,
                'score': merged_score
            })
    
    return merged_pairs


if __name__ == "__main__":
    """简单测试截图和OCR功能"""
    print("开始测试屏幕截图和OCR功能...")
    print("请确保屏幕上有一些文字内容，测试将在3秒后开始截图")
    
    import time
    
    # 给用户准备时间
    for i in range(3, 0, -1):
        print(f"倒计时: {i}秒")
        time.sleep(1)
    
    try:
        print("正在截图并进行OCR识别...")
        text_result, image_path = capture_screen()
        
        print("\n" + "="*50)
        print("OCR识别结果:")
        print("="*50)
        print(text_result)
        print("\n" + "="*50)
        print(f"标注图像已保存到: {image_path}")
        print("="*50)
        
        # 尝试打开图像查看（仅在macOS上）
        import subprocess
        import sys
        if sys.platform == "darwin":  # macOS
            try:
                subprocess.run(["open", str(image_path)], check=True)
                print("已在默认图像查看器中打开标注图像")
            except subprocess.CalledProcessError:
                print("无法自动打开图像，请手动查看上述路径的图像文件")
        else:
            print("请手动查看上述路径的标注图像文件")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


