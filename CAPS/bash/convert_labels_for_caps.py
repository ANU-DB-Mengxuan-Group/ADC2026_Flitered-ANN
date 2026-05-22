#!/usr/bin/env python3
"""
转换 ACORN 格式的 label 文件为 CAPS 格式

ACORN 格式 (label_base.txt):
7,19
9,25
5,23,27

CAPS 格式 (label_NHQ_base.txt):
132687 30
7,19
9,25
5,23,27

注意: CAPS 源码使用逗号作为分隔符，所以保持逗号分隔

用法:
  python convert_labels_for_caps.py <input_file> <output_file>
  python convert_labels_for_caps.py /path/to/label_base.txt /path/to/label_NHQ_base.txt
"""

import sys
import os

def convert_labels(input_file, output_file):
    # 读取所有行
    with open(input_file, 'r') as f:
        lines = f.readlines()

    num_points = len(lines)

    # 找出最大的属性值（用于确定 num_attributes）
    max_attr = 0
    converted_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            converted_lines.append("")
            continue

        # 保持逗号分隔（CAPS 源码使用逗号作为分隔符）
        attrs = line.split(',')
        for attr in attrs:
            try:
                val = int(attr)
                if val > max_attr:
                    max_attr = val
            except:
                pass

        # 保持原始格式（逗号分隔）
        converted_lines.append(line)

    # num_attributes 是最大属性值 + 1（因为从0开始）
    num_attributes = max_attr + 1

    # 写入输出文件
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w') as f:
        # 写入头部
        f.write(f"{num_points} {num_attributes}\n")
        # 写入转换后的行
        for line in converted_lines:
            f.write(line + '\n')

    print(f"Converted {num_points} points, {num_attributes} attributes")
    print(f"Output: {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_labels_for_caps.py <input_file> <output_file>")
        print("Example: python convert_labels_for_caps.py label_base.txt label_NHQ_base.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)

    convert_labels(input_file, output_file)

if __name__ == "__main__":
    main()
