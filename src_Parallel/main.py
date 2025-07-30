"""
Copyright (c) 2025, the Route Seeker developers.
Members of the Route Seeker developers include:
- Haoze Lv <12232421@mail.sustech.edu.cn>
- Zubin Zheng <zhengzb2021@mail.sustech.edu.cn>
All rights reserved.
This file is part of Route Seeker, a Python Program for solving Vehicle Routing Problems with Time Windows.
Distributed under the MIT License. See LICENSE for more information.
"""
from cli_parser import parse_args
from read import read_instance
from Model import Model
from solve import SolveParams, solve_parallel
from parallel_hgs_solver import ParallelSolveParams
from GeneticAlgorithm import GeneticAlgorithmParams
from parameter_configs import get_predefined_strategies
from result_saver import save_single_run_result
from pyvrp.stop import MaxRuntime
import statistics
import time

def is_better_solution(result1, result2):
    """
    比较两个Result对象，判断result1是否比result2更好
    比较标准：
    1. 车辆总数，车辆数少的优先
    2. 车辆总数相同时，比较总行驶距离，距离短的优先
    3. 总行驶距离相同时，比较总时间，时间短的优先
    """
    # 比较车辆数
    vehicles1 = result1.best.num_routes()
    vehicles2 = result2.best.num_routes()
    if vehicles1 != vehicles2:
        return vehicles1 < vehicles2
    
    # 车辆数相同，比较距离
    distance1 = result1.best.distance()
    distance2 = result2.best.distance()
    if distance1 != distance2:
        return distance1 < distance2
    
    # 距离相同，比较时间
    duration1 = result1.best.duration()
    duration2 = result2.best.duration()
    return duration1 < duration2

if __name__ == "__main__":

    # 解析传入的命令行参数
    args = parse_args()

    # 使用传入的参数读取实例和设置求解时间
    INSTANCE = read_instance(
        args.instance_path, 
        instance_format="solomon", 
        # round_func="exact"
        # round_func="dimacs"
        round_func="docs"
    )
    
    # 检查是否启用并行模式
    if args.parallel_mode:
        print("Parallel multi-strategy mode enabled!")
        
        # 创建并行求解参数
        parallel_params = ParallelSolveParams(
            num_strategies=args.num_strategies,
            sync_frequency=args.sync_frequency,
            decomposition_frequency=args.decomposition_freq,
            num_subproblems=args.num_subproblems,
            subproblem_iters=args.subproblem_iters,
            strategy_configs=get_predefined_strategies()[:args.num_strategies],
            collect_stats=True,
            display=True
        )
        
        print(f"Solving {args.instance_path} with parallel multi-strategy HGS...")
        print(f"Runtime: {args.runtime} seconds")
        print(f"Number of runs: {args.runs}")
        print(f"Number of strategies: {args.num_strategies}")
        print(f"Sync frequency: {args.sync_frequency} iterations") 
        print(f"Decomposition frequency: {args.decomposition_freq} iterations")
        print(f"Subproblems: {args.num_subproblems}")
        
        # 设置种子递增常量
        SEED_INCREMENT = 100
        
        best_result = None
        
        # 收集所有运行的结果用于统计
        all_distances = []
        all_vehicle_counts = []
        all_durations = []
        all_run_times = []
        all_feasible_results = []  # 保存所有可行的Result对象
        feasible_runs = 0
        
        # 循环控制并行求解器运行次数
        for run in range(args.runs):
            current_seed = args.seed + run * SEED_INCREMENT
            print(f"\n--- Parallel Run {run + 1}/{args.runs} (seed: {current_seed}) ---")
            
            # 记录开始时间
            start_time = time.time()
            
            # 为每次运行创建新的模型实例，确保完全独立
            model = Model.from_data(INSTANCE)
            result, synchronizer = solve_parallel(
                model.data(),
                stop=MaxRuntime(args.runtime),
                seed=current_seed,
                collect_stats=True,
                display=True,
                params=parallel_params
            )
            
            # 记录结束时间并计算运行时间
            end_time = time.time()
            run_time = round(end_time - start_time, 2)
            all_run_times.append(run_time)
            
            # 检查是否有可行解
            if result.best and result.best.is_feasible():
                distance = round(result.best.distance() / 100, 2)
                vehicle_count = result.best.num_routes()
                duration = round(result.best.duration() / 100, 2)
                print(f"Found a solution with # vehicles: {vehicle_count}, distance: {distance:.2f}, duration: {duration:.2f}.")
                print(f"Run time: {run_time:.2f} seconds")
                
                # 收集统计数据
                all_distances.append(distance)
                all_vehicle_counts.append(vehicle_count)
                all_durations.append(duration)
                all_feasible_results.append(result)
                feasible_runs += 1
                
                # 更新最佳结果
                if best_result is None or is_better_solution(result, best_result):
                    best_result = result
                    print(f"*** New best solution found! ***")
                    
                    # 保存当前最佳结果
                    try:
                        save_single_run_result(args, result, synchronizer, parallel_params, run_time)
                    except Exception as e:
                        print(f"Warning: Failed to save results: {e}")
                        print("Continuing without saving results...")
            else:
                print("No feasible solution found within the given time limit.")
                print(f"Run time: {run_time:.2f} seconds")
        
        # 输出最终结果
        print(f"\n=== Final Parallel Results ===")
        print(f"Total runs: {args.runs}")
        print(f"Feasible solutions found: {feasible_runs}")
        
        if best_result is not None:
            best_distance = round(best_result.best.distance() / 100, 2)
            best_duration = round(best_result.best.duration() / 100, 2)

            # 输出统计信息
            if args.runs > 1:
                print(f"\nStatistics over {args.runs} parallel runs:")
                
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
                    
                    # 时间统计
                    avg_duration = statistics.mean(all_durations)
                    std_duration = statistics.stdev(all_durations) if len(all_durations) > 1 else 0.0
                    print(f"  Duration - Mean: {avg_duration:.2f}, Std Dev: {std_duration:.2f}")
                    print(f"  Duration - Min: {min(all_durations):.2f}, Max: {max(all_durations):.2f}")
                elif feasible_runs == 1:
                    print(f"\nOnly one feasible solution found, no solution quality statistics to compute.")
            else:
                print(f"\nSingle parallel run completed in {all_run_times[0]:.2f} seconds.")
            
            print(f"\nBest solution found over {args.runs} parallel runs:")
            print(f"  Vehicles: {best_result.best.num_routes()}")
            print(f"  Distance: {best_distance:.2f}")
            print(f"  Duration: {best_duration:.2f}")
            # 输出最佳解的详细信息和路径
            print(f"\n" + "="*60)
            print("BEST SOLUTION DETAILS:")
            print("="*60)
            print(best_result)
        else:
            print("No feasible solutions found in any of the parallel runs.")
        
        exit(0)  # 并行模式执行完毕退出
    
    # 原有的单策略模式逻辑
    print("Single-strategy mode (original implementation)")
    
    # 创建自定义的遗传算法参数
    genetic_params = GeneticAlgorithmParams(
        num_subproblems=args.num_subproblems,
        decomposition_frequency=args.decomposition_freq,
        subproblem_iters=args.subproblem_iters,
        # 保持其他参数为默认值
    )

    # 创建求解参数
    solve_params = SolveParams(genetic=genetic_params)
    
    # 设置种子递增常量
    SEED_INCREMENT = 100
    
    print(f"Solving {args.instance_path} with a max runtime of {args.runtime} seconds...")
    print(f"Number of runs: {args.runs}")
    print(f"Number of subproblems: {args.num_subproblems}")
    print(f"Decomposition frequency: {args.decomposition_freq}")
    print(f"Subproblem iterations: {args.subproblem_iters}")
    
    best_result = None
    
    # 收集所有运行的结果用于统计
    all_distances = []
    all_vehicle_counts = []
    all_durations = []
    all_run_times = []
    all_feasible_results = []  # 保存所有可行的Result对象
    feasible_runs = 0
    
    # 循环控制求解器运行次数
    for run in range(args.runs):
        current_seed = args.seed + run * SEED_INCREMENT
        print(f"\n--- Run {run + 1}/{args.runs} (seed: {current_seed}) ---")
        
        # 记录开始时间
        start_time = time.time()
        
        # 为每次运行创建新的模型实例，确保完全独立
        model = Model.from_data(INSTANCE)
        result = model.solve(stop=MaxRuntime(args.runtime), seed=current_seed, display=True, params=solve_params)
        
        # 记录结束时间并计算运行时间
        end_time = time.time()
        run_time = round(end_time - start_time, 2)
        all_run_times.append(run_time)
        
        # 检查是否有可行解
        if result.best.is_feasible():
            distance = round(result.best.distance() / 100, 2)
            vehicle_count = result.best.num_routes()
            duration = round(result.best.duration() / 100, 2)
            print(f"Found a solution with # vehicles: {vehicle_count}, distance: {distance:.2f}, duration: {duration:.2f}.")
            print(f"Run time: {run_time:.2f} seconds")
            
            # 收集统计数据
            all_distances.append(distance)
            all_vehicle_counts.append(vehicle_count)
            all_durations.append(duration)
            all_feasible_results.append(result)
            feasible_runs += 1
            
            # 更新最佳结果
            if best_result is None or is_better_solution(result, best_result):
                best_result = result
                print(f"*** New best solution found! ***")
        else:
            print("No feasible solution found within the given time limit.")
            print(f"Run time: {run_time:.2f} seconds")
    
    # 输出最终结果
    print(f"\n=== Final Results ===")
    print(f"Total runs: {args.runs}")
    print(f"Feasible solutions found: {feasible_runs}")
    
    if best_result is not None:
        best_distance = round(best_result.best.distance() / 100, 2)
        best_duration = round(best_result.best.duration() / 100, 2)

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
                
                # 时间统计
                avg_duration = statistics.mean(all_durations)
                std_duration = statistics.stdev(all_durations) if len(all_durations) > 1 else 0.0
                print(f"  Duration - Mean: {avg_duration:.2f}, Std Dev: {std_duration:.2f}")
                print(f"  Duration - Min: {min(all_durations):.2f}, Max: {max(all_durations):.2f}")
            elif feasible_runs == 1:
                print(f"\nOnly one feasible solution found, no solution quality statistics to compute.")
        else:
            print(f"\nSingle run completed in {all_run_times[0]:.2f} seconds.")
        

        print(f"\nBest solution found over {args.runs} runs:")
        print(f"  Vehicles: {best_result.best.num_routes()}")
        print(f"  Distance: {best_distance:.2f}")
        print(f"  Duration: {best_duration:.2f}")
        # 输出最佳解的详细信息和路径
        print(f"\n" + "="*60)
        print("BEST SOLUTION DETAILS:")
        print("="*60)
        print(best_result)
    else:
        print("No feasible solutions found in any of the runs.")