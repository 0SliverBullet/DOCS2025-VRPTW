#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import shlex

# 将所有要运行的命令存储在一个列表中
commands = [
    """python main.py ../data/homberger_200_customer_instances/RC1_2_1.txt \
    --parallel_mode --num_strategies 8 --sync_frequency 1500 --runtime 1800 \
    --runs 10 --num_subproblems 2 --decomposition_freq 1500 --subproblem_iters 2000""",
    
    """python main.py ../data/homberger_200_customer_instances/RC1_2_2.txt \
    --parallel_mode --num_strategies 8 --sync_frequency 1500 --runtime 1800 \
    --runs 10 --num_subproblems 2 --decomposition_freq 1500 --subproblem_iters 2000""",
    
    """python main.py ../data/homberger_200_customer_instances/RC1_2_3.txt \
    --parallel_mode --num_strategies 8 --sync_frequency 1500 --runtime 1800 \
    --runs 10 --num_subproblems 2 --decomposition_freq 1500 --subproblem_iters 2000""",

    """python main.py ../data/homberger_200_customer_instances/RC1_2_4.txt \
    --parallel_mode --num_strategies 8 --sync_frequency 1500 --runtime 1800 \
    --runs 10 --num_subproblems 2 --decomposition_freq 1500 --subproblem_iters 2000""",

    """python main.py ../data/homberger_200_customer_instances/RC1_2_5.txt \
    --parallel_mode --num_strategies 8 --sync_frequency 1500 --runtime 1800 \
    --runs 10 --num_subproblems 2 --decomposition_freq 1500 --subproblem_iters 2000""",

    """python main.py ../data/homberger_400_customer_instances/RC1_4_1.txt \
    --parallel_mode --num_strategies 8 --sync_frequency 1500 --runtime 1800 \
    --runs 10 --num_subproblems 4 --decomposition_freq 1500 --subproblem_iters 2000""",

    """python main.py ../data/homberger_400_customer_instances/RC1_4_2.txt \
    --parallel_mode --num_strategies 8 --sync_frequency 1500 --runtime 1800 \
    --runs 10 --num_subproblems 4 --decomposition_freq 1500 --subproblem_iters 2000""",

    """python main.py ../data/homberger_400_customer_instances/RC1_4_3.txt \
    --parallel_mode --num_strategies 8 --sync_frequency 1500 --runtime 1800 \
    --runs 10 --num_subproblems 4 --decomposition_freq 1500 --subproblem_iters 2000""",

    """python main.py ../data/homberger_400_customer_instances/RC1_4_4.txt \
    --parallel_mode --num_strategies 8 --sync_frequency 1500 --runtime 1800 \
    --runs 10 --num_subproblems 4 --decomposition_freq 1500 --subproblem_iters 2000""",

    """python main.py ../data/homberger_400_customer_instances/RC1_4_5.txt \
    --parallel_mode --num_strategies 8 --sync_frequency 1500 --runtime 1800 \
    --runs 10 --num_subproblems 4 --decomposition_freq 1500 --subproblem_iters 2000""",
]

# 依次执行每条命令
for i, command in enumerate(commands):
    print("="*80)
    print(f"🚀 [ {i+1} / {len(commands)} ] 正在执行: {command.replace(chr(92), '').replace(chr(10), '')}") # 打印时移除换行符和斜杠
    print("="*80)
    try:
        # 使用 subprocess.run() 执行命令
        # check=True 表示如果命令返回非零退出码（即出错），脚本将抛出异常并停止
        # shell=True 允许我们以单个字符串的形式运行复杂的 shell 命令
        subprocess.run(command, shell=True, check=True)
        print(f"✅ 命令 [ {i+1} / {len(commands)} ] 执行成功！")
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令执行失败，返回码: {e.returncode}")
        print("脚本已终止。")
        break # 如果出错则停止循环

print("\n🎉 所有命令已执行完毕！")