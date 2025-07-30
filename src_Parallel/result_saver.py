"""
Result saver for parallel multi-strategy HGS solver.
Handles saving of parallel execution results including strategy performance.
"""

import os
import json
from datetime import datetime
from typing import Union

# Handle imports that may not be available
try:
    from parallel_hgs_solver import ParallelSolveParams
    from solution_synchronizer import SolutionSynchronizer
    from pyvrp.Result import Result
except ImportError:
    # Fallback types for when PyVRP is not available
    ParallelSolveParams = object
    SolutionSynchronizer = object
    Result = object


def save_parallel_results(args, result: Union[Result, object], synchronizer: Union[SolutionSynchronizer, object], 
                         parallel_params: Union[ParallelSolveParams, object], total_runtime: float):
    """
    保存并行多策略求解的结果到文件，包含完整的并行执行信息
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    result : Result
        最佳求解结果
    synchronizer : SolutionSynchronizer
        同步器，包含运行统计信息
    parallel_params : ParallelSolveParams
        并行求解参数
    total_runtime : float
        总运行时间
    """
    
    # 获取实例名称
    instance_name = os.path.splitext(os.path.basename(args.instance_path))[0]
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建结果文件夹结构：results_Parallel/instance_name/
    results_base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results_Parallel")
    instance_dir = os.path.join(results_base_dir, instance_name)
    os.makedirs(instance_dir, exist_ok=True)
    
    # 创建结果文件名
    result_filename = f"{instance_name}_{timestamp}.json"
    result_path = os.path.join(instance_dir, result_filename)
    
    # 获取同步统计信息
    sync_stats = synchronizer.get_statistics_summary()
    
    # 准备结果数据
    result_data = {
        "instance_info": {
            "instance_name": instance_name,
            "instance_path": args.instance_path,
            "timestamp": timestamp,
            "runtime": args.runtime,
            "total_runtime": round(total_runtime, 2),
            "solver_type": "parallel_multi_strategy_hgs"
        },
        "parallel_configuration": {
            # 并行参数
            "parallel_params": {
                "num_strategies": parallel_params.num_strategies,
                "sync_frequency": parallel_params.sync_frequency,
                "decomposition_frequency": parallel_params.decomposition_frequency,
                "num_subproblems": parallel_params.num_subproblems,
                "subproblem_iters": parallel_params.subproblem_iters,
                "collect_stats": parallel_params.collect_stats,
                "display": parallel_params.display
            },
            # 命令行参数
            "command_line_args": {
                "parallel_mode": getattr(args, 'parallel_mode', True),
                "num_strategies": getattr(args, 'num_strategies', parallel_params.num_strategies),
                "sync_frequency": getattr(args, 'sync_frequency', parallel_params.sync_frequency),
                "decomposition_freq": args.decomposition_freq,
                "num_subproblems": args.num_subproblems,
                "subproblem_iters": args.subproblem_iters,
                "seed": args.seed,
                "num_cores": args.num_cores
            },
            # 策略配置
            "strategy_configs": []
        },
        "synchronization_statistics": sync_stats,
        "best_solution": None,
        "detailed_solution": None,
        "strategy_performance": []
    }
    
    # 添加策略配置信息
    if parallel_params.strategy_configs:
        for i, config in enumerate(parallel_params.strategy_configs):
            strategy_info = {
                "strategy_id": i,
                "strategy_name": config.strategy_name,
                "repair_probability": config.repair_probability,
                "nb_iter_no_improvement": config.nb_iter_no_improvement,
                "min_population_size": config.min_population_size,
                "generation_size": config.generation_size,
                "lb_diversity": config.lb_diversity,
                "ub_diversity": config.ub_diversity,
                "nb_elite": config.nb_elite,
                "nb_close": config.nb_close,
                "weight_wait_time": config.weight_wait_time,
                "weight_time_warp": config.weight_time_warp,
                "nb_granular": config.nb_granular,
                "symmetric_proximity": config.symmetric_proximity,
                "symmetric_neighbours": config.symmetric_neighbours,
                "penalty_increase": config.penalty_increase,
                "penalty_decrease": config.penalty_decrease
            }
            result_data["parallel_configuration"]["strategy_configs"].append(strategy_info)
    
    # 添加最佳解信息
    if result and result.best:
        best_distance = round(result.best.distance() / 100, 2)
        best_duration = round(result.best.duration() / 100, 2)

        result_data["best_solution"] = {
            "vehicles": result.best.num_routes(),
            "distance": best_distance,
            "duration": best_duration,
            "is_feasible": result.best.is_feasible(),
            "cost": result.cost() if hasattr(result, 'cost') else None,
            "iterations": result.iterations if hasattr(result, 'iterations') else None
        }
        
        # 添加详细解信息
        routes_info = []
        for i, route in enumerate(result.best.routes()):
            route_info = {
                "route_id": i + 1,
                "visits": list(route.visits()),
                "distance": round(route.distance() / 100, 1),
                "duration": round(route.duration() / 100, 1),
                "start_time": route.start_time(),
                "end_time": route.end_time()
            }
            routes_info.append(route_info)
        
        result_data["detailed_solution"] = {
            "routes": routes_info,
            "total_routes": len(routes_info)
        }
    
    # 添加策略性能数据（来自同步历史）
    if synchronizer.sync_history:
        for sync_result in synchronizer.sync_history[-10:]:  # 保存最后10次同步的数据
            strategy_perf = {
                "sync_iteration": synchronizer.global_iteration,
                "global_best_strategy_id": sync_result.global_best_strategy_id,
                "global_best_strategy_name": sync_result.global_best_stats.strategy_name,
                "improvement_found": sync_result.improvement_found,
                "decomposition_triggered": sync_result.decomposition_triggered,
                "sync_time": round(sync_result.sync_time, 3),
                "strategies_performance": []
            }
            
            for stats in sync_result.strategies_stats:
                perf_data = {
                    "strategy_id": stats.strategy_id,
                    "strategy_name": stats.strategy_name,
                    "best_vehicles": stats.best_vehicles,
                    "best_distance": stats.best_distance,
                    "best_duration": stats.best_duration,
                    "is_feasible": stats.is_feasible,
                    "iterations_no_improvement": stats.iterations_no_improvement
                }
                strategy_perf["strategies_performance"].append(perf_data)
            
            result_data["strategy_performance"].append(strategy_perf)
    
    # 保存到JSON文件
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n并行求解结果已保存到: {result_path}")
    
    # 保存文本报告
    report_filename = f"{instance_name}_{timestamp}_report.txt"
    report_path = os.path.join(instance_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"并行多策略HGS求解结果报告\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"实例信息:\n")
        f.write(f"  实例名称: {instance_name}\n")
        f.write(f"  实例路径: {args.instance_path}\n")
        f.write(f"  运行时间: {timestamp}\n")
        f.write(f"  最大运行时间: {args.runtime}秒\n")
        f.write(f"  实际运行时间: {total_runtime:.2f}秒\n")
        f.write(f"  求解器类型: 并行多策略HGS\n\n")
        
        f.write(f"并行配置:\n")
        f.write(f"  策略数量: {parallel_params.num_strategies}\n")
        f.write(f"  同步频率: {parallel_params.sync_frequency}迭代\n")
        f.write(f"  分解频率: {parallel_params.decomposition_frequency}迭代\n")
        f.write(f"  子问题数量: {parallel_params.num_subproblems}\n")
        f.write(f"  子问题迭代: {parallel_params.subproblem_iters}\n")
        f.write(f"  CPU核心利用: {args.num_cores}核\n")
        f.write(f"  随机种子: {args.seed}\n\n")
        
        f.write(f"策略配置:\n")
        if parallel_params.strategy_configs:
            for i, config in enumerate(parallel_params.strategy_configs):
                f.write(f"  策略 {i+1}: {config.strategy_name}\n")
                f.write(f"    修复概率: {config.repair_probability}\n")
                f.write(f"    收敛耐心: {config.nb_iter_no_improvement}\n")
                f.write(f"    种群大小: {config.min_population_size}\n")
                f.write(f"    惩罚增加: {config.penalty_increase}\n")
        f.write(f"\n")
        
        f.write(f"同步统计:\n")
        f.write(f"  总同步次数: {sync_stats.get('total_synchronizations', 0)}\n")
        f.write(f"  发现改进: {sync_stats.get('total_improvements', 0)}次\n")
        f.write(f"  触发分解: {sync_stats.get('total_decompositions', 0)}次\n")
        f.write(f"  改进率: {sync_stats.get('improvement_rate', 0):.1%}\n")
        f.write(f"  平均同步时间: {sync_stats.get('average_sync_time', 0):.3f}秒\n\n")
        
        if result and result.best:
            f.write(f"最优解:\n")
            f.write(f"  车辆数: {result.best.num_routes()}\n")
            f.write(f"  总距离: {best_distance}\n")
            f.write(f"  总时间: {best_duration}\n")
            f.write(f"  可行性: {result.best.is_feasible()}\n")
            if 'global_best_strategy' in sync_stats:
                f.write(f"  最佳策略: {sync_stats['global_best_strategy']}\n")
            f.write(f"\n")
            
            f.write(f"路径详情:\n")
            for i, route in enumerate(result.best.routes()):
                f.write(f"  路径 {i+1}: {list(route.visits())} (距离: {route.distance()/100:.2f})\n")
        else:
            f.write(f"未找到可行解\n")
    
    print(f"文本报告已保存到: {report_path}")
    
    # 保存配置文件
    config_filename = f"{instance_name}_{timestamp}_config.json"
    config_path = os.path.join(instance_dir, config_filename)
    
    config_data = {
        "instance_path": args.instance_path,
        "parallel_configuration": result_data["parallel_configuration"],
        "description": f"Parallel multi-strategy configuration for {instance_name} at {timestamp}"
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"配置文件已保存到: {config_path}")
    
    # 生成复现脚本
    script_filename = f"{instance_name}_{timestamp}_reproduce.sh"
    script_path = os.path.join(instance_dir, script_filename)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# 并行多策略HGS复现脚本\n")
        f.write(f"# 生成时间: {timestamp}\n\n")
        f.write(f"python src_Parallel/main.py {args.instance_path} \\\n")
        f.write(f"    --parallel_mode \\\n")
        f.write(f"    --runtime {args.runtime} \\\n")
        f.write(f"    --num_strategies {getattr(args, 'num_strategies', parallel_params.num_strategies)} \\\n")
        f.write(f"    --sync_frequency {getattr(args, 'sync_frequency', parallel_params.sync_frequency)} \\\n")
        f.write(f"    --decomposition_freq {args.decomposition_freq} \\\n")
        f.write(f"    --num_subproblems {args.num_subproblems} \\\n")
        f.write(f"    --subproblem_iters {args.subproblem_iters} \\\n")
        f.write(f"    --seed {args.seed}\n")
    
    # 设置脚本可执行权限
    os.chmod(script_path, 0o755)
    print(f"复现脚本已保存到: {script_path}")
    
    return result_path


def create_results_summary(instance_name: str, results_dir: str):
    """
    为特定实例创建结果汇总文件
    
    Parameters
    ----------
    instance_name : str
        实例名称
    results_dir : str
        结果目录路径
    """
    
    # 查找该实例的所有结果文件
    json_files = []
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.json') and not file.endswith('_config.json'):
                json_files.append(file)
    
    if not json_files:
        return
    
    json_files.sort()  # 按时间排序
    
    # 创建汇总文件
    summary_filename = f"{instance_name}_summary.md"
    summary_path = os.path.join(results_dir, summary_filename)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# {instance_name} 并行求解结果汇总\n\n")
        f.write(f"总计运行次数: {len(json_files)}\n\n")
        f.write(f"| 时间戳 | 车辆数 | 距离 | 运行时间 | 策略数 | 同步次数 | 改进次数 |\n")
        f.write(f"|--------|--------|------|----------|--------|----------|----------|\n")
        
        best_result = None
        best_vehicles = float('inf')
        best_distance = float('inf')
        
        for json_file in json_files:
            json_path = os.path.join(results_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                
                timestamp = data['instance_info']['timestamp']
                runtime = data['instance_info'].get('total_runtime', 0)
                
                if data['best_solution']:
                    vehicles = data['best_solution']['vehicles']
                    distance = data['best_solution']['distance']
                    
                    # 更新最佳结果
                    if vehicles < best_vehicles or (vehicles == best_vehicles and distance < best_distance):
                        best_vehicles = vehicles
                        best_distance = distance
                        best_result = data
                else:
                    vehicles = "N/A"
                    distance = "N/A"
                
                strategies = data['parallel_configuration']['parallel_params']['num_strategies']
                sync_count = data['synchronization_statistics'].get('total_synchronizations', 0)
                improvements = data['synchronization_statistics'].get('total_improvements', 0)
                
                f.write(f"| {timestamp} | {vehicles} | {distance} | {runtime:.2f}s | {strategies} | {sync_count} | {improvements} |\n")
                
            except Exception as e:
                print(f"Warning: Failed to process {json_file}: {e}")
        
        f.write(f"\n## 最佳结果\n\n")
        if best_result:
            f.write(f"- **时间戳**: {best_result['instance_info']['timestamp']}\n")
            f.write(f"- **车辆数**: {best_result['best_solution']['vehicles']}\n")
            f.write(f"- **距离**: {best_result['best_solution']['distance']}\n")
            f.write(f"- **运行时间**: {best_result['instance_info']['total_runtime']:.2f}秒\n")
            if 'global_best_strategy' in best_result['synchronization_statistics']:
                f.write(f"- **最佳策略**: {best_result['synchronization_statistics']['global_best_strategy']}\n")
        else:
            f.write("未找到可行解\n")
    
    print(f"结果汇总已保存到: {summary_path}")


def save_single_run_result(args, result: Union[Result, object], synchronizer: Union[SolutionSynchronizer, object], 
                          parallel_params: Union[ParallelSolveParams, object], total_runtime: float):
    """
    保存单次并行运行的结果（完整版本的包装）
    """
    result_path = save_parallel_results(args, result, synchronizer, parallel_params, total_runtime)
    
    # 获取实例名称和结果目录
    instance_name = os.path.splitext(os.path.basename(args.instance_path))[0]
    results_base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results_Parallel")
    instance_dir = os.path.join(results_base_dir, instance_name)
    
    # 创建或更新结果汇总
    create_results_summary(instance_name, instance_dir)
    
    return result_path