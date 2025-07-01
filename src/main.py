"""
Copyright (c) 2025, the Route Seeker developers.
Members of the Route Seeker developers include:
- Zubin Zheng <zhengzb2021@mail.sustech.edu.cn>
All rights reserved.
This file is part of Route Seeker, a Python Program for solving Vehicle Routing Problems with Time Windows.
Distributed under the MIT License. See LICENSE for more information.
"""
from cli_parser import parse_args
from read import read_instance
from Model import Model
from pyvrp.stop import MaxRuntime

if __name__ == "__main__":

    # 解析传入的命令行参数
    args = parse_args()

    # 使用传入的参数读取实例和设置求解时间
    INSTANCE = read_instance(
        args.instance_path, 
        instance_format="solomon", 
        round_func="exact"
    )
    
    # 设置种子递增常量
    SEED_INCREMENT = 100
    
    print(f"Solving {args.instance_path} with a max runtime of {args.runtime} seconds...")
    print(f"Number of runs: {args.runs}")
    
    best_result = None
    best_distance = float('inf')
    
    # 循环控制求解器运行次数
    for run in range(args.runs):
        current_seed = args.seed + run * SEED_INCREMENT
        print(f"\n--- Run {run + 1}/{args.runs} (seed: {current_seed}) ---")
        
        # 为每次运行创建新的模型实例，确保完全独立
        model = Model.from_data(INSTANCE)
        result = model.solve(stop=MaxRuntime(args.runtime), seed=current_seed, display=True)
        
        # 检查是否有可行解
        if result.best.is_feasible():
            distance = round(result.best.distance() / 1000, 2)
            print(f"Found a solution with # vehicles: {result.best.num_routes()}, distance: {distance:.2f}.")
            
            # 更新最佳结果
            if distance < best_distance:
                best_distance = distance
                best_result = result
                print(f"*** New best solution found! ***")
        else:
            print("No feasible solution found within the given time limit.")
    
    # 输出最终结果
    print(f"\n=== Final Results ===")
    if best_result is not None:
        print(f"Best solution found over {args.runs} runs:")
        print(f"  Vehicles: {best_result.best.num_routes()}")
        print(f"  Distance: {best_distance:.2f}")
    else:
        print("No feasible solutions found in any of the runs.")