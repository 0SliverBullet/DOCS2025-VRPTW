import argparse

def parse_args():
    """
    解析用于VRPTW求解器的命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="Route Seeker: A VRPTW solver with Barycenter Clustering Decomposition."
    )

    # 必需参数：实例文件路径
    parser.add_argument(
        "instance_path", 
        type=str, 
        help="Path to the instance file (e.g., data/C1_2_1.TXT)."
    )

    # 可选参数：总运行时间
    parser.add_argument(
        "--runtime",
        "-t",
        type=int,
        default=1800,
        help="Total solver runtime in seconds. Default is 1800."
    )

    # 可选参数：运行次数
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=1,
        help="Number of times to run the solver. Default is 1."
    )

    # 可选参数：CPU核心数量
    parser.add_argument(    
        "--num_cores",
        "-c",
        type=int,
        default=8,
        help="Number of CPU cores to use for parallel processing. Default is 8."
    )

    # 可选参数：子问题数量
    parser.add_argument(
        "--num_subproblems",
        type=int,
        default=8,
        help="Number of subproblems to create (should match CPU cores). Default is 8."
    )
    
    # 您可能还想把之前添加的其他参数也放进来
    parser.add_argument(
        "--decomposition_freq", 
        type=int, 
        default=4_000, 
        help="Iterations between decomposition phases."
    )
    
    parser.add_argument(
        "--subproblem_iters", 
        type=int, 
        default=1_000, 
        help="Number of iterations for solving subproblems."
    )
    
    parser.add_argument(
        "--max_customers_per_subproblem",
        type=int,
        default=None,
        help="(Optional) Set a maximum number of customers per subproblem for balancing."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for all operations."
    )
    
    # <<< 新增并行求解相关参数 >>>
    parser.add_argument(
        "--parallel_mode",
        action="store_true",
        default=False,
        help="Enable parallel multi-strategy HGS mode."
    )
    
    parser.add_argument(
        "--sync_frequency",
        type=int,
        default=1000,
        help="Synchronization frequency between strategies (in iterations). Default is 1500."
    )
    
    parser.add_argument(
        "--num_strategies",
        type=int,
        default=8,
        help="Number of parallel strategies to run (should match CPU cores). Default is 8."
    )
    
    parser.add_argument(
        "--strategy_configs",
        type=str,
        default=None,
        help="Path to custom strategy configurations file (optional)."
    )

    return parser.parse_args()