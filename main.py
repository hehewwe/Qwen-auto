from ast import arg
import os
import base64
import requests
import json
import csv
import shutil
import argparse
import cv2
from PIL import Image
from tqdm import tqdm
import datetime

# --- 命令行参数更改  ---
parser = argparse.ArgumentParser(description='通义千问多模态模型分析工具')
parser.add_argument('--input', type=str, default=r"D:\hyj_python\hyj_projiect\Qwen-auto\images",
                    help='输入路径，可以是图片文件夹或视频文件夹.')
parser.add_argument('--input_type', type=str, choices=['image', 'video'], default='image',
                    help='输入类型：image(图片) 或 video(视频)，默认为image')
parser.add_argument('--frames', type=int, default=20, help='从每个视频抽取的帧数，默认为5帧')
parser.add_argument('--output', type=str, default='output1', help='输出文件夹路径，默认为"output"')
parser.add_argument('--yolo_output', type=str, default='yolo_dataset1', help='YOLO数据集输出路径，默认为"yolo_dataset"')
parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例，默认为0.2(20%)')
parser.add_argument('--api_key', type=str, default=None, help='您的灵积模型服务（DashScope）API密钥。如果未提供，将尝试从环境变量 DASHSCOPE_API_KEY 读取。')

args = parser.parse_args()

# --- 获取API Key ---
API_KEY = args.api_key or os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    print("错误：未提供API密钥。")
    print("您可以通过以下任一方式提供密钥：")
    print("  1. 使用命令行参数: --api_key YOUR_API_KEY")
    print("  2. 设置名为 DASHSCOPE_API_KEY 的环境变量。")
    exit()

# --- 生成唯一的输出文件夹名称 ---
def generate_unique_folder_name(base_name):
    """
    为输出文件夹生成唯一名称，添加时间戳
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}-{timestamp}"

# 使用时间戳创建唯一的输出文件夹
output_folder = generate_unique_folder_name(args.output)
yolo_output_folder = generate_unique_folder_name(args.yolo_output)

print(f"将使用以下输出文件夹:")
print(f"- 主输出文件夹: {output_folder}")
print(f"- YOLO数据集文件夹: {yolo_output_folder}")

# --- 设置输出路径 ---
frames_folder = os.path.join(output_folder, "frames")
label_folder = os.path.join(output_folder, "labels")
summary_csv_path = os.path.join(output_folder, "summary.csv")

# --- YOLO格式输出设置 ---
yolo_images_train = os.path.join(yolo_output_folder, "images", "train")
yolo_labels_train = os.path.join(yolo_output_folder, "labels", "train")
yolo_images_val = os.path.join(yolo_output_folder, "images", "val")
yolo_labels_val = os.path.join(yolo_output_folder, "labels", "val")

# 创建输出文件夹
os.makedirs(label_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(yolo_images_train, exist_ok=True)
os.makedirs(yolo_labels_train, exist_ok=True)
os.makedirs(yolo_images_val, exist_ok=True)
os.makedirs(yolo_labels_val, exist_ok=True)

# 设置输入路径
input_folder = args.input

# --- 初始化结果存储 ---
summary_data = []
# --- 设置类别 ---
# 英文类别名，用于YOLO和labelme。模型也将返回这些英文标签。
class_names = [
    "sw",              # 撒网收网
    "by",           # 搬运渔获或物资
    "tw",                  # 投喂饲料
    "wh",    # 设备维护
    "qt" # 其他生产活动
]
validation_split = args.val_split  # 验证集比例

# 视频抽帧函数
def extract_frames_from_video(video_path, output_dir, num_frames=5):
    """
    从视频中均匀抽取指定数量的帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 帧保存目录
        num_frames: 要抽取的帧数
    
    Returns:
        抽取的帧文件路径列表
    """
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 创建视频专属的帧目录
    video_frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return []
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"视频信息: {video_path}")
    print(f"总帧数: {total_frames}, FPS: {fps:.2f}, 时长: {duration:.2f}秒")
    
    # 确保不会抽取超过视频总帧数的帧
    num_frames = min(num_frames, total_frames)
    if num_frames <= 0:
        print(f"警告: 视频 {video_path} 没有足够的帧可供抽取")
        return []
    
    # 计算抽帧间隔
    interval = total_frames / num_frames
    
    frame_paths = []
    
    # 使用tqdm显示进度
    print(f"正在从视频中抽取 {num_frames} 帧...")
    for i in range(num_frames):
        # 计算当前帧位置
        frame_pos = int(i * interval)
        
        # 设置视频读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print(f"警告: 无法读取视频 {video_path} 的第 {frame_pos} 帧")
            continue
        
        # 保存帧为图片
        frame_filename = f"{video_name}_frame_{i:04d}.jpg"
        frame_path = os.path.join(video_frames_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    
    # 释放视频资源
    cap.release()
    
    print(f"已从视频 {video_path} 抽取 {len(frame_paths)} 帧")
    return frame_paths

# 创建YOLO数据集配置文件
def create_yolo_dataset_yaml():
    yaml_content = f"""# YOLO数据集配置
path: {os.path.abspath(yolo_output_folder)}  # 数据集根目录
train: images/train  # 训练图像相对路径
val: images/val  # 验证图像相对路径

# 类别
nc: {len(class_names)}  # 类别数量
names: {class_names}  # 类别名称
"""
    
    with open(os.path.join(yolo_output_folder, "dataset.yaml"), 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"已创建YOLO数据集配置文件: {os.path.join(yolo_output_folder, 'dataset.yaml')}")

# --- 新增：图片预处理函数 ---
def resize_image_in_place(image_path, max_size=1280):
    """
    如果图片的尺寸超过最大限制，则在保持长宽比的情况下进行缩放，并覆盖原始文件。
    
    Args:
        image_path (str): 图片文件路径。
        max_size (int): 允许的最大宽度或高度。
    """
    try:
        with Image.open(image_path) as img:
            # 检查图片格式是否支持覆盖保存
            if img.format.upper() not in ['JPEG', 'PNG', 'BMP']:
                # print(f"跳过不支持覆盖保存的图片格式: {img.format} ({os.path.basename(image_path)})")
                return

            width, height = img.size
            if width > max_size or height > max_size:
                print(f"正在缩放图片: {os.path.basename(image_path)} (原始尺寸 {width}x{height})")
                
                # 计算新尺寸，保持长宽比
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                # 使用高质量的LANCZOS算法进行缩放
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 覆盖保存
                resized_img.save(image_path)
                # print(f"图片已缩放并保存: {os.path.basename(image_path)} (新尺寸 {new_width}x{new_height})")

    except Exception as e:
        print(f"缩放图片时发生错误 {os.path.basename(image_path)}: {e}")

# 处理单个图片文件
def process_image(image_path, is_validation, target_images_folder, target_labels_folder):
    filename = os.path.basename(image_path)
    print(f"\n正在处理图片: {filename} [{'验证集' if is_validation else '训练集'}]")
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    # DashScope 通义千问VL API 地址
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    payload = {
        "model": "qwen2.5-vl-72b-instruct",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": f"data:image/jpeg;base64,{base64_image}"
                        },
                        {
                            "text": (
                                "你是一个顶级的图像分析专家，任务是分析渔排上的生产活动，并为后续的自动化标注提供结构化数据。\n\n"
                                "分析规则:\n"
                                "1. 生产活动定义: 任何与渔业生产直接相关的动作，如操作渔具、搬运、投喂、维护等。\n"
                                "2. 非生产活动: 仅站立、观望、休息等不视为生产活动。\n"
                                "3. 识别对象: 只识别正在进行生产活动的人员。\n"
                                f"4. 活动类别: 为每个生产人员选择一个最合适的活动标签。标签必须从以下列表中选择: {class_names}\n\n"
                                "输出要求:\n"
                                "请严格按照以下JSON格式返回分析结果，不要添加任何额外的解释性文字或Markdown标记。返回的结果必须是一个完整的、可被解析的JSON对象。\n"
                                "{\n"
                                "  \"overall_producing\": \"是/否/无法判定\",\n"
                                "  \"producing_personnel\": [\n"
                                "    {\n"
                                "      \"id\": 1,\n"
                                "      \"label\": \"casting_net\",\n"
                                "      \"description\": \"对此人的具体活动描述，例如：正在用力向水中撒网。\",\n"
                                "      \"bounding_box\": [x_min, y_min, x_max, y_max]\n"
                                "    }\n"
                                "  ]\n"
                                "}\n\n"
                                "字段说明:\n"
                                "- `overall_producing`: 根据画面整体情况，判断是否存在生产活动 ('是'/'否'/'无法判定')。\n"
                                "- `producing_personnel`: 一个列表，包含所有正在进行生产活动的人员。如果无人生产，则返回空列表 `[]`。\n"
                                "  - `id`: 人员的唯一序号，从1开始。\n"
                                "  - `label`: 生产活动的类别标签，必须是预定义列表中的一个英文字符串。\n"
                                "  - `description`: 对该人员生产活动的详细中文描述。\n"
                                "  - `bounding_box`: 包含四元整数的列表 `[x_min, y_min, x_max, y_max]`，代表该人员在图像中的像素坐标边界框。**此字段至关重要。**\n\n"
                                "请开始分析提供的图片。"
                            )
                        }
                    ]
                }
            ]
        },
        "parameters": {
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if "output" in result and "choices" in result["output"] and result["output"]["choices"]:
                raw_content = result["output"]["choices"][0]["message"]["content"]
                print(f"模型分析结果：\n{raw_content}")

                # ---  解析模型返回的JSON，并生成报告和标签文件 ---
                try:
                    json_string = ""
                    if isinstance(raw_content, list) and raw_content:
                        text_content = raw_content[0].get("text", "")
                        json_string = text_content
                    elif isinstance(raw_content, str):
                        json_string = raw_content

                    if json_string.startswith("```json"):
                        json_string = json_string[7:]
                    if json_string.endswith("```"):
                        json_string = json_string[:-3]
                    json_string = json_string.strip()

                    if not json_string:
                        raise json.JSONDecodeError("无法从模型响应中提取有效的JSON内容。", "", 0)

                    analysis_result = json.loads(json_string)
                    overall_producing = analysis_result.get("overall_producing", "未知")
                    producing_personnel = analysis_result.get("producing_personnel", [])

                    # 添加到汇总数据
                    image_source = os.path.dirname(image_path).split(os.path.sep)[-1]  # 获取图像来源（文件夹名）
                    summary_data.append({
                        "image_filename": filename,
                        "image_source": image_source,
                        "overall_producing": overall_producing,
                        "personnel_count": len(producing_personnel),
                        "dataset": "验证集" if is_validation else "训练集",
                        "raw_response": json_string
                    })
                    
                    # 复制图片到YOLO数据集目录
                    target_image_path = os.path.join(target_images_folder, filename)
                    shutil.copy2(image_path, target_image_path)
                    
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                    
                    # 生成labelme格式标签
                    if producing_personnel:
                        labelme_data = {
                            "version": "5.0.1",
                            "flags": {},
                            "shapes": [],
                            "imagePath": filename,
                            "imageData": None,  
                            "imageHeight": img_height,
                            "imageWidth": img_width
                        }

                        # 同时生成YOLO格式标签
                        yolo_labels = []
                        
                        for person in producing_personnel:
                            box = person.get("bounding_box")
                            label = person.get("label")

                            # 验证模型返回的label是否有效
                            if label not in class_names:
                                print(f"警告: 文件 {filename} 中，模型返回了无效的标签 '{label}'，将自动归类为 '{class_names[-1]}'。")
                                label = class_names[-1] # 默认为最后一个类别

                            if box and len(box) == 4:
                                # Labelme格式
                                shape = {
                                    "label": label,
                                    "points": [
                                        [box[0], box[1]],
                                        [box[2], box[3]]
                                    ],
                                    "group_id": None,
                                    "shape_type": "rectangle",
                                    "flags": {}
                                }
                                labelme_data["shapes"].append(shape)
                                
                                # YOLO格式 - 转换为归一化坐标 [class_id, x_center, y_center, width, height]
                                x_min, y_min, x_max, y_max = box
                                x_center = (x_min + x_max) / 2.0 / img_width
                                y_center = (y_min + y_max) / 2.0 / img_height
                                width = (x_max - x_min) / img_width
                                height = (y_max - y_min) / img_height
                                
                                # 确保值在0-1范围内
                                x_center = max(0, min(1, x_center))
                                y_center = max(0, min(1, y_center))
                                width = max(0, min(1, width))
                                height = max(0, min(1, height))
                                
                                # 获取类别ID
                                class_id = class_names.index(label)
                                yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
                        
                        # 保存labelme格式
                        if labelme_data["shapes"]:
                            label_filename = os.path.splitext(filename)[0] + ".json"
                            label_filepath = os.path.join(label_folder, label_filename)
                            with open(label_filepath, 'w', encoding='utf-8') as f:
                                json.dump(labelme_data, f, ensure_ascii=False, indent=4)
                            print(f"已生成labelme标签文件: {label_filepath}")
                        
                        # 保存YOLO格式
                        if yolo_labels:
                            yolo_label_filename = os.path.splitext(filename)[0] + ".txt"
                            yolo_label_filepath = os.path.join(target_labels_folder, yolo_label_filename)
                            with open(yolo_label_filepath, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(yolo_labels))
                            print(f"已生成YOLO标签文件: {yolo_label_filepath}")
                    else:
                        # 如果没有标注对象，仍然创建空的YOLO标签文件
                        yolo_label_filename = os.path.splitext(filename)[0] + ".txt"
                        yolo_label_filepath = os.path.join(target_labels_folder, yolo_label_filename)
                        with open(yolo_label_filepath, 'w', encoding='utf-8') as f:
                            pass  # 创建空文件
                        print(f"已创建空YOLO标签文件: {yolo_label_filepath}")
                        
                    return True
                        
                except json.JSONDecodeError as e:
                    print(f"警告: 模型返回的内容不是有效的JSON格式，无法处理。错误: {e}")
                    # 添加到汇总数据
                    image_source = os.path.dirname(image_path).split(os.path.sep)[-1]  # 获取图像来源（文件夹名）
                    summary_data.append({
                        "image_filename": filename,
                        "image_source": image_source,
                        "overall_producing": "解析失败",
                        "personnel_count": 0,
                        "dataset": "验证集" if is_validation else "训练集",
                        "raw_response": str(raw_content) 
                    })
                    
                    # 仍然复制图片到YOLO数据集，但不创建标签文件
                    target_image_path = os.path.join(target_images_folder, filename)
                    shutil.copy2(image_path, target_image_path)
                    return True
            else:
                print("未能从响应中提取有效内容，完整响应：", result)
        else:
            print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
    except Exception as e:
        print(f"发生异常: {e}")
    
    return False

# 主处理流程
def main():
    all_image_files = []
    
    # 根据输入类型进行处理
    if args.input_type == 'image':
        # 直接处理图片文件夹
        print(f"正在处理图片文件夹: {input_folder}")
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(input_folder, filename)
                all_image_files.append(image_path)
    
    elif args.input_type == 'video':
        # 处理视频文件夹，先抽帧
        print(f"正在处理视频文件夹: {input_folder}")
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                video_path = os.path.join(input_folder, filename)
                # 从视频中抽取帧
                frame_paths = extract_frames_from_video(video_path, frames_folder, args.frames)
                all_image_files.extend(frame_paths)
    
    # --- 新增步骤：在处理前统一缩放所有图片 ---
    print("\n--- 步骤1: 图片预处理 - 检查并缩放图片尺寸 (最大边1280px) ---")
    if all_image_files:
        for image_path in tqdm(all_image_files, desc="预处理图片尺寸"):
            resize_image_in_place(image_path, max_size=1280)
    print("--- 图片尺寸预处理完成 ---\n")
    
    # 处理所有图片
    total_images = len(all_image_files)
    val_count = int(total_images * validation_split)
    train_count = total_images - val_count
    
    print(f"找到 {total_images} 张图片，将分配 {train_count} 张用于训练，{val_count} 张用于验证")
    
    # 使用tqdm显示总体进度
    if all_image_files:
        for idx, image_path in enumerate(tqdm(all_image_files, desc="处理图片")):
            # 决定图片是训练集还是验证集
            is_validation = idx < val_count
            target_images_folder = yolo_images_val if is_validation else yolo_images_train
            target_labels_folder = yolo_labels_val if is_validation else yolo_labels_train
            
            # 处理单个图片
            process_image(image_path, is_validation, target_images_folder, target_labels_folder)
    
    # 创建YOLO数据集配置文件
    create_yolo_dataset_yaml()
    
    # 写入CSV总结报告
    if summary_data:
        print(f"\n正在写入总结报告: {summary_csv_path}")
        try:
            with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["image_filename", "image_source", "overall_producing", "personnel_count", "dataset", "raw_response"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
            print("报告生成完毕。")
        except Exception as e:
            print(f"写入CSV文件时发生错误: {e}")
    
    print(f"\n处理完成! YOLO格式数据集已生成在 '{os.path.abspath(yolo_output_folder)}' 目录")
    print(f"训练集: {train_count} 张图片")
    print(f"验证集: {val_count} 张图片")
    print(f"类别: {class_names}")
    
    # 创建一个README文件，记录处理信息
    readme_path = os.path.join(output_folder, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入路径: {os.path.abspath(input_folder)}\n")
        f.write(f"输入类型: {args.input_type}\n")
        if args.input_type == 'video':
            f.write(f"每个视频抽取帧数: {args.frames}\n")
        f.write(f"总图片数: {total_images}\n")
        f.write(f"训练集数量: {train_count}\n")
        f.write(f"验证集数量: {val_count}\n")
        f.write(f"验证集比例: {args.val_split}\n")
        f.write(f"YOLO数据集路径: {os.path.abspath(yolo_output_folder)}\n")
    
    print(f"处理信息已保存到: {readme_path}")

if __name__ == "__main__":
    main()