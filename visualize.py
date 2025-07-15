import os
import json
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import glob
import csv

def load_font():
    """尝试加载系统中的中文字体"""
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc"
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path
    print("警告: 未找到中文字体，标签可能无法正确显示。")
    return None

def draw_box_pil(img, box, label, color=(0, 255, 0), thickness=2):
    """使用PIL在图像上绘制边界框和标签"""
    draw = ImageDraw.Draw(img)
    x_min, y_min, x_max, y_max = map(int, box)
    
    for i in range(thickness):
        draw.rectangle([(x_min + i, y_min + i), (x_max - i, y_max - i)], outline=color)
    
    font_path = load_font()
    font_size = 20
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except IOError:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    
    text_bg_y_min = y_min - text_height - 10
    if text_bg_y_min < 0:
        text_bg_y_min = y_max + 10
        
    draw.rectangle([(x_min, text_bg_y_min), (x_min + text_width + 4, text_bg_y_min + text_height + 5)], fill=color)
    draw.text((x_min + 2, text_bg_y_min + 2), label, fill=(255, 255, 255), font=font)
    
    return img

def draw_summary_text(img, text, position=(10, 10), font_size=25, color=(255, 255, 0)):
    """在图片上绘制汇总信息文本"""
    # 确保图像是RGBA模式以支持透明背景
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    draw = ImageDraw.Draw(img)
    font_path = load_font()
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except IOError:
        font = ImageFont.load_default()

    # 创建一个临时图层用于绘制带背景的文本
    text_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_layer)

    # 计算文本框大小并绘制背景
    text_bbox = text_draw.textbbox(position, text, font=font)
    bg_rect = (text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5)
    text_draw.rectangle(bg_rect, fill=(0, 0, 0, 128)) # 半透明黑色背景

    # 绘制文本
    text_draw.text(position, text, font=font, fill=color)

    # 将文本图层复合到原始图像上
    out_img = Image.alpha_composite(img, text_layer)
    return out_img.convert('RGB')


def visualize_yolo_txt(txt_path, output_dir, image_dir, class_names):
    """可视化YOLO格式的TXT标注文件"""
    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    image_path = next((os.path.join(image_dir, base_name + ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'] if os.path.exists(os.path.join(image_dir, base_name + ext))), None)
    
    if not image_path:
        print(f"警告: 找不到YOLO标注 {base_name}.txt 对应的图像文件。")
        return

    try:
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f.readlines() if line.strip()]

        if not lines:
            img = draw_summary_text(img, "人数: 0")
            img.save(os.path.join(output_dir, base_name + "_visualized.jpg"))
            return

        for line in lines:
            parts = line.split()
            class_id, x_center, y_center, width, height = map(float, parts)
            class_id = int(class_id)
            
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)
            
            label = class_names[class_id] if class_names and class_id < len(class_names) else f"未知{class_id}"
            img = draw_box_pil(img, [x_min, y_min, x_max, y_max], label)
        
        img = draw_summary_text(img, f"人数: {len(lines)}")
        img.save(os.path.join(output_dir, base_name + "_visualized.jpg"))
    except Exception as e:
        print(f"处理YOLO文件 {txt_path} 时出错: {e}")

def visualize_labelme_json(json_path, output_dir, image_dir):
    """可视化labelme格式的JSON标注文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_filename = data["imagePath"]
        image_path = os.path.join(image_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"警告: 找不到LabelMe标注 {image_filename} 对应的图像文件。")
            return
            
        img = Image.open(image_path).convert("RGB")
        shapes = data.get("shapes", [])
        for shape in shapes:
            if shape.get("shape_type") == "rectangle":
                points = shape.get("points", [])
                if len(points) == 2:
                    box = [points[0][0], points[0][1], points[1][0], points[1][1]]
                    img = draw_box_pil(img, box, shape.get("label", "unknown"))
        
        img = draw_summary_text(img, f"人数: {len(shapes)}")
        img.save(os.path.join(output_dir, os.path.splitext(image_filename)[0] + "_visualized.jpg"))
    except Exception as e:
        print(f"处理LabelMe文件 {json_path} 时出错: {e}")

def visualize_from_csv(csv_path, output_dir, image_dir):
    """从CSV文件中读取分析结果并可视化"""
    try:
        if not os.path.exists(csv_path):
            print(f"错误: CSV文件不存在 {csv_path}")
            return

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        for row in tqdm(rows, desc="从CSV生成可视化"):
            image_filename = row.get("image_filename")
            
            if not image_filename:
                print(f"警告: CSV中有一行缺少 'image_filename'，已跳过。")
                continue
            
            image_source = row.get("image_source", "")
            possible_image_paths = [
                os.path.join(image_dir, "images", "train", image_filename),
                os.path.join(image_dir, "images", "val", image_filename),
                os.path.join(image_dir, "frames", image_source, image_filename),
                os.path.join(image_dir, image_filename)
            ]
            
            image_path = next((p for p in possible_image_paths if p and os.path.exists(p)), None)

            if not image_path:
                print(f"警告: 找不到CSV中记录的图像 {image_filename}。")
                continue
                
            img = Image.open(image_path).convert("RGB")
            personnel_count = row.get("personnel_count", "N/A")
            
            try:
                analysis_result = json.loads(row["raw_response"])
                producing_personnel = analysis_result.get("producing_personnel", [])
                personnel_count = len(producing_personnel)
                for person in producing_personnel:
                    box = person.get("bounding_box")
                    label = person.get("label", "unknown")
                    if box and len(box) == 4:
                        img = draw_box_pil(img, box, label)

            except (json.JSONDecodeError, TypeError):
                print(f"无法解析 {image_filename} 的JSON数据，仅显示CSV中的人数。")

            img = draw_summary_text(img, f"人数: {personnel_count}")
            img.save(os.path.join(output_dir, os.path.splitext(image_filename)[0] + "_visualized.jpg"))

    except Exception as e:
        print(f"处理CSV文件 {csv_path} 时出错: {e}")

def process_yolo_dataset(yolo_dataset_dir, output_dir):
    """处理YOLO数据集，自动查找labels和images目录"""
    print(f"处理YOLO数据集: {yolo_dataset_dir}")

    yaml_path = os.path.join(yolo_dataset_dir, "dataset.yaml")
    class_names = []
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "names:" in content:
                    names_line = content.split("names:")[1].strip().split("\n")[0]
                    class_names = [name.strip().strip("'\"[] ") for name in names_line.replace("'", "").split(",")]
                    print(f"从YAML加载类别: {class_names}")
        except Exception as e:
            print(f"读取YAML失败: {e}")

    for subset in ["train", "val"]:
        labels_dir = os.path.join(yolo_dataset_dir, "labels", subset)
        images_dir = os.path.join(yolo_dataset_dir, "images", subset)
        output_subset_dir = os.path.join(output_dir, subset)
        os.makedirs(output_subset_dir, exist_ok=True)
        if os.path.exists(labels_dir) and os.path.exists(images_dir):
            print(f"处理YOLO {subset} 集...")
            for txt_file in tqdm(glob.glob(os.path.join(labels_dir, "*.txt")), desc=f"可视化 {subset}"):
                visualize_yolo_txt(txt_file, output_subset_dir, images_dir, class_names)
        else:
            print(f"未找到YOLO {subset} 集的 'labels' 或 'images' 目录。")

def main():
    # ===============================================================
    #                 --- 用户配置区域 ---
    # ===============================================================
    
    # 1. 选择处理模式: "yolo", "labelme", 或 "csv"
    process_mode = "csv"

    # 2. 根据模式设置路径
    
    # --- YOLO模式配置 ---
    # 指向YOLO数据集的根目录 (包含images, labels, dataset.yaml的文件夹)
    yolo_dataset_dir = r"D:\hyj_python\hyj_projiect\Qwen-auto\yolo_dataset1-20250715_095101"
    
    # --- LabelMe模式配置 ---
    # 指向 .json 文件所在的目录
    labelme_json_dir = r"D:\path\to\your\labelme_jsons"
    # 指向对应的原始图片目录,如果是视频，则指向output\frames\Recording目录
    labelme_image_dir = r"D:\path\to\your\original_images"

    # --- CSV模式配置 ---
    # 指向 summary.csv 文件的完整路径
    summary_csv_path = r"D:\hyj_python\hyj_projiect\Qwen-auto\output-20250715_110550\summary.csv"
    # CSV模式下，图片直接指向images目录，如果是视频，则指向output\frames\Recording目录
    csv_image_dir = r"D:\hyj_python\hyj_projiect\Qwen-auto\output-20250715_110550\frames\Recording 2025-07-11 154109"
    
    # 3. 设置输出目录
    output_dir = "可视化结果"

    # ===============================================================
    #                 --- 执行区域 ---
    # ===============================================================
    
    print(f"--- 可视化流程启动 (模式: {process_mode}) ---")
    
    os.makedirs(output_dir, exist_ok=True)

    if process_mode == "yolo":
        if not os.path.isdir(yolo_dataset_dir):
            print(f"错误: YOLO数据集目录不存在或不是一个目录: {yolo_dataset_dir}")
            return
        process_yolo_dataset(yolo_dataset_dir, output_dir)

    elif process_mode == "labelme":
        if not os.path.isdir(labelme_json_dir) or not os.path.isdir(labelme_image_dir):
            print(f"错误: LabelMe的JSON目录或图片目录不存在。")
            return
        print("处理LabelMe JSON文件...")
        json_files = glob.glob(os.path.join(labelme_json_dir, "*.json"))
        for json_file in tqdm(json_files, desc="可视化LabelMe"):
            visualize_labelme_json(json_file, output_dir, labelme_image_dir)

    elif process_mode == "csv":
        if not os.path.isfile(summary_csv_path) or not os.path.isdir(csv_image_dir):
            print(f"错误: CSV文件或其对应的图片目录不存在。")
            return
        visualize_from_csv(summary_csv_path, output_dir, csv_image_dir)
    else:
        print(f"错误: 未知的处理模式 '{process_mode}'")

    print(f"--- 可视化完成 --- \n结果已保存至: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()