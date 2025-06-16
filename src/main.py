from pyvrp import read, solve
import os

DATA_DIR = "data"

def read_data_files(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".TXT"):
                file_path = os.path.join(root, file)
                print(f"读取文件: {file_path}")
                with open(file_path, "r") as f:
                    # 只读取前5行作为示例
                    for i, line in enumerate(f):
                        print(line.strip())
                        if i >= 4:
                            break
                print("-" * 40)

if __name__ == "__main__":
    read_data_files(DATA_DIR)