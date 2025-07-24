#!/usr/bin/env python3
"""
批量测试脚本
按顺序测试 data/homberger_800_customer_instances 文件夹下的所有样例
"""

import os
import subprocess
import sys
from pathlib import Path
import glob

def main():
    # 设置路径
    data_dir = "data/homberger_800_customer_instances"
    output_dir = "output/results0716"
    src_main = "./src/main.py"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有.TXT文件并按名称排序
    pattern = os.path.join(data_dir, "*.TXT")
    all_txt_files = sorted(glob.glob(pattern))
    
    # 过滤掉以_1.TXT和_5.TXT结尾的文件
    # txt_files = [f for f in all_txt_files if not (f.endswith("_1.TXT") or f.endswith("_5.TXT"))]
    txt_files = [f for f in all_txt_files]
    
    if not all_txt_files:
        print(f"在 {data_dir} 中未找到.TXT文件")
        return
    
    # skipped_files = [f for f in all_txt_files if f.endswith("_1.TXT") or f.endswith("_5.TXT")]
    # if skipped_files:
    #     print(f"跳过的文件: {[Path(f).name for f in skipped_files]}")
    
    print(f"找到 {len(all_txt_files)} 个测试文件，将测试 {len(txt_files)} 个文件")
    
    # 逐个测试每个文件
    for i, txt_file in enumerate(txt_files, 1):
        # 获取文件名（不包含路径和扩展名）
        filename = Path(txt_file).stem
        output_file = os.path.join(output_dir, f"{filename}.txt")
        
        print(f"\n[{i}/{len(txt_files)}] 正在测试: {txt_file}")
        print(f"结果将保存到: {output_file}")
        
        # 构建命令
        cmd = [
            sys.executable,  # 使用当前Python解释器
            src_main,
            txt_file,
            "--runtime", "1800",
            "--runs", "10",
            "--num_subproblems", "8",
            "--decomposition_freq", "1500",
            "--subproblem_iters", "2000"
        ]
        
        try:
            # 运行命令并实时显示输出
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"测试文件: {txt_file}\n")
                f.write(f"命令: {' '.join(cmd)}\n")
                f.write("=" * 50 + "\n\n")
                f.flush()
                
                # 使用Popen实现实时输出
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1,
                    universal_newlines=True
                )
                
                # 实时读取并显示输出
                output_lines = []
                if process.stdout:
                    while True:
                        line = process.stdout.readline()
                        if line == '' and process.poll() is not None:
                            break
                        if line:
                            # 同时输出到终端和文件
                            print(line, end='')
                            f.write(line)
                            f.flush()
                            output_lines.append(line)
                
                # 等待进程结束
                return_code = process.wait()
                
                if return_code != 0:
                    error_msg = f"\n\n错误: 命令执行失败，退出码: {return_code}\n"
                    f.write(error_msg)
                    print(error_msg)
                    print(f"  ❌ 测试失败，退出码: {return_code}")
                else:
                    print(f"  ✅ 测试完成")
                    
        except Exception as e:
            print(f"  ❌ 执行出错: {e}")
            # 仍然创建输出文件记录错误
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"测试文件: {txt_file}\n")
                f.write(f"命令: {' '.join(cmd)}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"执行错误: {e}\n")
    
    print(f"\n所有测试完成！结果已保存到 {output_dir} 文件夹")

if __name__ == "__main__":
    main() 