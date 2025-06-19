# Copyright (c) 2025, the Route Seeker developers.
# Members of the Route Seeker developers include:
# - Zubin Zheng <zhengzb2021@mail.sustech.edu.cn>
# All rights reserved.
# This file is part of Route Seeker, a Python Program for solving Vehicle Routing Problems with Time Windows.
# Distributed under the MIT License. See LICENSE for more information.
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
    INSTANCE = read_instance("data/homberger_200_customer_instances/C1_2_1.TXT", instance_format="solomon", round_func="dimacs")
    model = Model.from_data(INSTANCE)
    result = model.solve(stop=MaxRuntime(30), seed=42, display=True)
    cost = result.cost() / 10
    print(f"Found a solution with cost: {cost}.")