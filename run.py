import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="渔排生产活动分析与可视化工具集",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="可执行命令")

    # --- 'analyze' 命令: 调用 main.py ---
    parser_analyze = subparsers.add_parser(
        "analyze",
        help="运行数据分析与自动标注流程 (调用 main.py)。",
        description="此命令会启动 main.py 脚本来处理图片或视频，并生成标注数据。"
    )
    # 添加 'main.py' 支持的所有参数
    parser_analyze.add_argument('--input', type=str, required=True, help='输入路径，可以是图片文件夹或视频文件夹。')
    parser_analyze.add_argument('--input_type', type=str, choices=['image', 'video'], default='image', help='输入类型: image 或 video。')
    parser_analyze.add_argument('--frames', type=int, default=20, help='从每个视频抽取的帧数。')
    parser_analyze.add_argument('--output', type=str, default='output', help='主输出文件夹的基础名称。')
    parser_analyze.add_argument('--yolo_output', type=str, default='yolo_dataset', help='YOLO数据集文件夹的基础名称。')
    parser_analyze.add_argument('--val_split', type=float, default=0.2, help='验证集比例。')

    # --- 'visualize' 命令: 调用 visualize.py ---
    parser_visualize = subparsers.add_parser(
        "visualize",
        help="运行标注结果可视化流程 (调用 visualize.py)。",
        description="此命令会启动 visualize.py 脚本来检查标注质量。\n注意: visualize.py 中的路径需要您手动修改。"
    )

    args, unknown = parser.parse_known_args()

    if args.command == "analyze":
        # 构建将传递给 main.py 的命令
        command = [
            sys.executable, 'main.py',
            '--input', args.input,
            '--input_type', args.input_type,
            '--frames', str(args.frames),
            '--output', args.output,
            '--yolo_output', args.yolo_output,
            '--val_split', str(args.val_split)
        ]
        print(f"正在执行: {' '.join(command)}")
        subprocess.run(command, check=True)

    elif args.command == "visualize":
        # 直接执行 visualize.py
        command = [sys.executable, 'visualize.py']
        print(f"正在执行: {' '.join(command)}")
        print("请确保您已在 visualize.py 文件中配置了正确的模式和路径。")
        subprocess.run(command, check=True)

if __name__ == "__main__":
    main()