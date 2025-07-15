好的，没问题。我已经记下了您关于自我介绍的指示。

下面是您提供的文本，已按照清晰的Markdown格式重新组织：

# Qwen-auto 智能生产活动分析工具 - 使用说明

---

## 首次使用设置

### 步骤 1: 安装必要的库
本工具依赖一些Python库来运行。

### 步骤 2: 准备API密钥
本工具需要调用阿里云的“灵积模型服务（DashScope）”来分析图片，因此您需要一个API密钥。

- **获取密钥**: 请前往 [阿里云灵积模型服务控制台](https://dashscope.console.aliyun.com/) 创建并获取您的API密钥（API-KEY）。

您有两种方式让本程序使用您的密钥：
- **方式一 (推荐)**: 在运行命令时通过 `--api_key` 参数直接提供。这是最简单直接的方式。
- **方式二 (可选)**: 将密钥设置为系统环境变量。变量名为 `DASHSCOPE_API_KEY`。程序在找不到命令行参数时会尝试读取该变量。

---

## 如何运行分析
分析功能由 `main.py` 脚本执行。您可以分析图片或视频。

### 示例 1: 分析一个文件夹里的所有图片
假设您的图片都存放在 `D:\project\test_images` 文件夹下，您想使用密钥 `sk-xxxxxxxx` 来进行分析。

**运行命令：**
```bash
python main.py --api_key "sk-xxxxxxxx" --input "D:\project\test_images" --input_type image
```

### 示例 2: 分析一个文件夹里的所有视频
假设您的视频都存放在 `D:\project\source_videos` 文件夹下，并且您希望从每个视频中抽取 `30` 帧进行分析。

**运行命令：**
```bash
python main.py --api_key "sk-xxxxxxxx" --input "D:\project\source_videos" --input_type video --frames 30
```

### 命令参数说明
- `--api_key`: 您的密钥。
- `--input`: 存放图片或视频的文件夹路径。
- `--input_type`: 指定要处理的是 `image` (图片) 还是 `video` (视频)。
- `--frames`: (仅用于视频) 指定从每个视频中抽取的帧数。

---

## 如何可视化标注结果
分析完成后，您可以使用 `visualize.py` 脚本来检查生成的标注质量。

- **步骤 1: 打开并配置 `visualize.py` 文件**
  - 用代码编辑器打开 `visualize.py`。找到文件底部的 **用户配置区域**。
- **步骤 2: 修改配置**
- **步骤 3: 运行可视化脚本**

---

## 输出文件结构简介

- **`output-xxxxxxxx.../`**: 主输出文件夹。
  - `frames/`: (仅视频) 存放从视频中抽取的帧图片。
  - `labels/`: 存放 `LabelMe` 格式的 `.json` 标注文件。
  - `summary.csv`: 每次分析的汇总报告。
- **`yolo_dataset-xxxxxxxx.../`**: `YOLO` 格式的数据集。
  - `images/train/`, `images/val/`: 训练集和验证集的图片。
  - `labels/train/`, `labels/val/`: 对应的YOLO格式 `.txt` 标签。
  - `dataset.yaml`: YOLO训练所需的配置文件。
- **`可视化结果/`**: 存放可视化结果图片。
Ran tool
