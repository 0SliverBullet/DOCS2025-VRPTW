"""
Copyright (c) 2025, the Route Seeker developers.
Members of the Route Seeker developers include:
- Zubin Zheng <zhengzb2021@mail.sustech.edu.cn>
All rights reserved.
This file is part of Route Seeker, a Python Program for solving Vehicle Routing Problems with Time Windows.
Distributed under the MIT License. See LICENSE for more information.
"""
import concurrent.futures
from cli_parser import parse_args
from read import read_instance
from pyvrp import Model, Result, ProblemData
from pyvrp.stop import MaxRuntime
import statistics
import time

# 这是一个跨平台兼容的工作函数。
# 通过显式传递instance_data，它可以在Windows(spawn)和Linux/macOS(fork)上可靠地工作。
def solve_worker(instance_data: ProblemData, seed: int, runtime: int) -> Result:
    """
    单个CPU核心执行的工作函数。
    它使用传入的实例数据创建一个模型，并用给定的随机种子进行求解。
    """
    model = Model.from_data(instance_data)
    
    # 在并行计算中，关闭每个核心的详细日志输出，避免信息混乱
    result = model.solve(stop=MaxRuntime(runtime), seed=seed, display=True)
    
    cost = result.cost() if result.is_feasible() else "N/A"
    print(f"  Core with seed {seed:2d} finished. Cost: {cost}")
    return result


if __name__ == "__main__":
    # --- 参数解析部分保持不变 ---
    args = parse_args()
    
    # --- 主逻辑部分保持不变 ---
    instance_data_main = read_instance(
        args.instance_path, instance_format="solomon", round_func="exact"
    )

    # --- 这里的NUM_CORES和INIT_SEED是从命令行参数中获取的 ---
    
    NUM_CORES = args.num_cores
    INIT_SEED = args.seed
    
    # 设置种子递增常量
    SEED_INCREMENT = 1000  # 为8核设置更大的种子增量，避免种子重叠
    
    print(f"Solving {args.instance_path} on {NUM_CORES} cores...")
    print(f"Max runtime per core: {args.runtime} seconds")
    print(f"Number of runs: {args.runs}\n")
    
    best_result = None
    best_distance = float('inf')
    
    # 收集所有运行的结果用于统计
    all_distances = []
    all_vehicle_counts = []
    all_run_times = []
    feasible_runs = 0
    
    # 循环控制求解器运行次数
    for run in range(args.runs):
        current_base_seed = args.seed + run * SEED_INCREMENT
        print(f"--- Run {run + 1}/{args.runs} (base seed: {current_base_seed}) ---")
        
        # 记录开始时间
        start_time = time.time()
        
        all_results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            seeds = [current_base_seed + i for i in range(NUM_CORES)]
            # 将数据作为参数传递，这是实现跨平台兼容的关键
            futures = {
                executor.submit(solve_worker, instance_data_main, seed, args.runtime): seed 
                for seed in seeds
            }
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as exc:
                    print(f"A worker process generated an exception: {exc}")
        
        # 记录结束时间并计算运行时间
        end_time = time.time()
        run_time = round(end_time - start_time, 2)
        all_run_times.append(run_time)

        feasible_results = [res for res in all_results if res.is_feasible()]

        if not feasible_results:
            print("No feasible solution found by any core within the given time limit.")
            print(f"Run time: {run_time:.2f} seconds")
        else:
            run_best_result = min(feasible_results, key=lambda res: res.cost())
            distance = round(run_best_result.best.distance() / 1000, 2)
            vehicle_count = run_best_result.best.num_routes()
            
            print(f"Found a solution with # vehicles: {vehicle_count}, distance: {distance:.2f}.")
            print(f"Run time: {run_time:.2f} seconds")
            
            # 收集统计数据
            all_distances.append(distance)
            all_vehicle_counts.append(vehicle_count)
            feasible_runs += 1
            
            # 更新最佳结果
            if distance < best_distance:
                best_distance = distance
                best_result = run_best_result
                print(f"*** New best solution found! ***")
    
    # 输出最终结果
    print(f"\n=== Final Results ===")
    print(f"Total runs: {args.runs}")
    print(f"Feasible solutions found: {feasible_runs}")
    
    if best_result is not None:
        print(f"\nBest solution found over {args.runs} runs:")
        print(f"  Vehicles: {best_result.best.num_routes()}")
        print(f"  Distance: {best_distance:.2f}")
        
        # 输出统计信息
        if args.runs > 1:
            print(f"\nStatistics over {args.runs} runs:")
            
            # 运行时间统计
            avg_runtime = statistics.mean(all_run_times)
            std_runtime = statistics.stdev(all_run_times) if len(all_run_times) > 1 else 0.0
            print(f"  Run Time - Mean: {avg_runtime:.2f}s, Std Dev: {std_runtime:.2f}s")
            print(f"  Run Time - Min: {min(all_run_times):.2f}s, Max: {max(all_run_times):.2f}s")
            
            if feasible_runs > 1:
                print(f"\nStatistics over {feasible_runs} feasible runs:")
                
                # 距离统计
                avg_distance = statistics.mean(all_distances)
                std_distance = statistics.stdev(all_distances) if len(all_distances) > 1 else 0.0
                print(f"  Distance - Mean: {avg_distance:.2f}, Std Dev: {std_distance:.2f}")
                print(f"  Distance - Min: {min(all_distances):.2f}, Max: {max(all_distances):.2f}")
                
                # 车辆数统计
                avg_vehicles = statistics.mean(all_vehicle_counts)
                std_vehicles = statistics.stdev(all_vehicle_counts) if len(all_vehicle_counts) > 1 else 0.0
                print(f"  Vehicles - Mean: {avg_vehicles:.2f}, Std Dev: {std_vehicles:.2f}")
                print(f"  Vehicles - Min: {min(all_vehicle_counts)}, Max: {max(all_vehicle_counts)}")
            elif feasible_runs == 1:
                print(f"\nOnly one feasible solution found, no solution quality statistics to compute.")
        else:
            print(f"\nSingle run completed in {all_run_times[0]:.2f} seconds.")
    else:
        print("No feasible solutions found in any of the runs.")