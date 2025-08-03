import math
import os
import glob

def truncate(number, decimal_places):
    """
    直接截断一个数字到指定的小数位数，不进行四舍五入。
    """
    if decimal_places < 0:
        raise ValueError("Decimal places must be non-negative.")
    if decimal_places == 0:
        return math.trunc(number)
    
    factor = 10.0 ** decimal_places
    return math.trunc(number * factor) / factor

def load_instance(filename, decimal_places=None):
    """加载VRP实例文件，支持控制欧式距离精度"""
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # 解析车辆容量信息
    Q = int(lines[4].split()[1])
    
    # 解析客户信息
    customers = []
    for line in lines[9:]:
        if stripped := line.strip():
            customers.append(stripped.split())
    
    # 解析仓库信息
    depot = customers[0]
    coords = [(int(depot[1]), int(depot[2]))]
    demands = [0]
    e_windows = [int(depot[4])]
    l_windows = [int(depot[5])]
    service_times = [0.0]

    # 解析客户信息
    for cust in customers[1:]:
        coords.append((int(cust[1]), int(cust[2])))
        demands.append(int(cust[3]))
        e_windows.append(int(cust[4]))
        l_windows.append(int(cust[5]))
        service_times.append(float(cust[6]))
    
    # 构建时间矩阵（欧几里得距离）
    num_locations = len(coords)
    time_matrix = [[0.0] * num_locations for _ in range(num_locations)]
    
    for i in range(num_locations):
        x1, y1 = coords[i]
        for j in range(i + 1, num_locations):
            x2, y2 = coords[j]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if decimal_places is not None:
                # 复赛
                distance = round(distance, decimal_places) 
                # 初赛
                # 【修改点】将 round() 替换为 truncate()
                # distance = truncate(distance, decimal_places)
            time_matrix[i][j] = distance
            time_matrix[j][i] = distance
    
    return {
        "time_matrix": time_matrix,
        "time_windows": list(zip(e_windows, l_windows)),
        "demands": demands,
        "service_times": service_times,
        "vehicle_capacity": Q,
        "num_locations": num_locations,
        "depot": 0
    }

def parse_solution_file(filename):
    """解析结果文件，提取路径信息"""
    routes = []
    current_route = []
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Route for Vehicle"):
                if current_route:
                    routes.append(current_route)
                    current_route = []
            elif "Customers:" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    nodes = parts[1].split("->")
                    current_route = [int(node.strip()) for node in nodes if node.strip()]
    
    if current_route:
        routes.append(current_route)
    
    return routes

def validate_solution(data, routes):
    """验证解决方案是否满足所有约束条件"""
    num_locations = data["num_locations"]
    depot_idx = data["depot"]
    Q = data["vehicle_capacity"]
    depot_e, depot_l = data["time_windows"][depot_idx]
    visited = [False] * num_locations
    visited[depot_idx] = True
    all_feasible = True

    # 检查客户点访问情况
    for route in routes:
        # 验证路径格式
        if route[0] != depot_idx or route[-1] != depot_idx:
            print(f"路径不以仓库开始或结束: {route}")
            all_feasible = False
        
        # 标记访问节点并检查重复访问
        for node in route[1:-1]:
            if not (0 <= node < num_locations):
                print(f"节点 {node} 超出范围")
                all_feasible = False
            elif visited[node]:
                print(f"节点 {node} 被重复访问")
                all_feasible = False
            else:
                visited[node] = True
    
    # 检查未访问的客户点
    for i in range(num_locations):
        if i != depot_idx and not visited[i]:
            print(f"节点 {i} 未被访问")
            all_feasible = False

    # 检查每条路径的容量和时间窗约束
    for idx, route in enumerate(routes, start=1):
        # 容量约束检查
        total_demand = sum(data["demands"][node] for node in route)
        if total_demand > Q:
            print(f"路径 {idx} 需求 {total_demand} 超过容量 {Q}")
            all_feasible = False
        
        # 时间窗约束检查
        current_time = depot_e  # 从仓库出发时间
        for i in range(1, len(route)):
            prev_node = route[i-1]
            curr_node = route[i]
            
            # 计算行驶时间并更新当前时间
            current_time += data["time_matrix"][prev_node][curr_node]
            
            # 获取当前节点信息
            e, l = data["time_windows"][curr_node]
            service_time = data["service_times"][curr_node]
            
            # 检查时间窗
            if current_time > l:
                print(f"路径 {idx} 节点 {curr_node} 到达时间 {current_time:.2f} 晚于最晚时间 {l}")
                all_feasible = False
            
            # 提前到达则等待
            if current_time < e:
                current_time = e
            
            # 执行服务
            current_time += service_time
        
        # 检查返回仓库时间
        if current_time > depot_l:
            print(f"路径 {idx} 返回仓库时间 {current_time:.2f} 晚于最晚时间 {depot_l}")
            all_feasible = False
    
    return all_feasible

# if __name__ == "__main__":
#     DATA_FILES = [
#         "C1_2_1.TXT", "C1_2_2.TXT", "C1_2_3.TXT", "C1_2_4.TXT", "C1_2_5.TXT",
#         "C1_8_1.TXT", "C1_8_2.TXT", "C1_8_3.TXT", "C1_8_4.TXT", "C1_8_5.TXT"
#     ]
    
#     for data_file in DATA_FILES:
#         print(f"\n验证实例: {data_file}")
        
#         # 加载问题实例（使用1位小数精度）
#         data_path = os.path.join("data/homberger_200_customer_instances", data_file)

#         instance_data = load_instance(data_path, decimal_places=1)
        
#         # 加载解决方案
#         sol_path = os.path.join("solutions/result0724", data_file)
#         if not os.path.exists(sol_path):
#             print(f"结果文件不存在: {sol_path}")
#             continue
        
#         routes = parse_solution_file(sol_path)
        
#         # 验证解决方案
#         if validate_solution(instance_data, routes):
#             print("解决方案满足所有约束条件!")
#         else:
#             print("解决方案违反约束条件!")

if __name__ == "__main__":
    # DATA_FILES 列表保持不变
    DATA_FILES = [
        "RC1_2_1.txt", "RC1_2_2.txt", "RC1_2_3.txt", "RC1_2_4.txt", "RC1_2_5.txt",
        "RC1_4_1.txt", "RC1_4_2.txt", "RC1_4_3.txt", "RC1_4_4.txt", "RC1_4_5.txt"
    ]
    
    # 【第1步：新增】定义所有可能存放数据文件的目录
    DATA_DIRECTORIES = [
        "data/homberger_200_customer_instances",
        "data/homberger_400_customer_instances"
    ]
    
    for data_file in DATA_FILES:
        print(f"\n验证实例: {data_file}")
        
        # 【第2步：修改】动态查找 data_path，而不是写死
        found_path = None
        for directory in DATA_DIRECTORIES:
            potential_path = os.path.join(directory, data_file)
            if os.path.exists(potential_path):
                found_path = potential_path
                break  # 找到后就跳出内部循环
        
        # 【第3步：新增】检查文件是否真的找到了
        if not found_path:
            print(f"错误: 在所有指定目录中都找不到文件 {data_file}")
            continue # 跳过这个文件，继续处理下一个

        # 现在 found_path 就是正确的路径，可能是200目录也可能是800目录
        # 使用这个找到的路径来加载实例
        data_path = found_path
        instance_data = load_instance(data_path, decimal_places=2)  # 使用2位小数精度
        
        base_name, ext = os.path.splitext(data_file)
        sol_filename = base_name + ".txt" # 直接指定小写 .txt
        
        sol_path = os.path.join("solutions/results0806", sol_filename)

        if not os.path.exists(sol_path):
            print(f"结果文件不存在: {sol_path}")
            continue
        
        routes = parse_solution_file(sol_path)
        
        if validate_solution(instance_data, routes):
            print("解决方案满足所有约束条件!")
        else:
            print("解决方案违反约束条件!")