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
from pyvrp import Model
from pyvrp.plotting import (
    plot_coordinates,
    plot_instance,
    plot_result,
    plot_route_schedule,
)
from pyvrp.stop import MaxIterations, MaxRuntime


if __name__ == "__main__":

    # 解析传入的命令行参数
    args = parse_args()

    # 使用传入的参数读取实例和设置求解时间
    INSTANCE = read_instance(
        args.instance_path, 
        instance_format="solomon", 
        round_func="exact"
    )
    
    model = Model.from_data(INSTANCE)
    
    print(f"Solving {args.instance_path} with a max runtime of {args.runtime} seconds...")
    
    result = model.solve(stop=MaxRuntime(args.runtime), seed=42, display=True)
    
    # 检查是否有可行解
    if result.best.is_feasible():
        distance = round(result.best.distance() / 1000, 2)
        print(f"Found a solution with # vehicles: {result.best.num_routes()}, distance: {distance:.2f}.")
    else:
        print("No feasible solution found within the given time limit.")