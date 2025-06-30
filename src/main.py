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
from pyvrp import Result, ProblemData
from Model import Model
from pyvrp.stop import MaxRuntime

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
    
    NUM_CORES = args.num_subproblems
    INIT_SEED = args.seed
    print(f"Solving {args.instance_path} on {NUM_CORES} cores...")
    print(f"Max runtime per core: {args.runtime} seconds.\n")
    
    all_results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        seeds = [INIT_SEED + i for i in range(NUM_CORES)]
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

    feasible_results = [res for res in all_results if res.is_feasible()]

    if not feasible_results:
        print("\nNo feasible solution found by any core within the given time limit.")
    else:
        best_result = min(feasible_results, key=lambda res: res.cost())
        distance = round(best_result.best.distance() / 1000, 2)
        
        print("\n" + "="*50)
        print("Best solution found across all cores:")
        print(f"  - Number of vehicles: {best_result.best.num_routes()}")
        print(f"  - Total distance: {distance:.2f}")
        # print(f"  - Route visits: {best_result.best.routes()[0].visits()}")
        print("="*50)