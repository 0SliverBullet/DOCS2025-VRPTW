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
        default=5000, 
        help="Iterations between decomposition phases."
    )
    
    parser.add_argument(
        "--subproblem_iters", 
        type=int, 
        default=10000, 
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

    return parser.parse_args()