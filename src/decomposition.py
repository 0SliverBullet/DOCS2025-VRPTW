import numpy as np
import os
from sklearn.cluster import KMeans
from pyvrp import ProblemData, Solution, VehicleType

# 设置环境变量解决KMeans内存泄漏警告
os.environ['OMP_NUM_THREADS'] = '1'

def barycenter_clustering_decomposition(
    elite_solution: Solution,
    original_data: ProblemData,
    num_clusters: int,
    max_customers_per_cluster: int | None = None, # <-- 新增参数：每个簇的客户上限
    random_state: int = 42  # <-- TODO完成：random_state作为参数
):
    """
    使用基于路线质心的k-means聚类策略将问题分解为子问题，并可选地进行均衡调整。

    参数:
    elite_solution: 用于分解的精英解。
    original_data: 原始的、完整的VRP问题数据。
    num_clusters: 要创建的子问题（聚类）的数量 (k)。
    max_customers_per_cluster: (可选) 每个聚类中允许的最大客户点数量。
    random_state: 用于K-means聚类的随机种子。

    返回:
    一个 ProblemData 对象的列表，每个对象代表一个子问题。
    """

    # 1. 计算每条非空路线的质心及其客户数量
    routes = [route for route in elite_solution.routes()]
    if not routes:
        return []

    # print(f"Elite solution routes in decomposition ({len(routes)} routes):")
    # for route_idx, route in enumerate(routes):
    #     print(f"  Route {route_idx + 1}: {list(route.visits())}")
    #     print()
    # exit(0)

    barycenters = []
    route_customer_counts = []
    for route in routes:
        x_coords = [original_data.location(c).x for c in route.visits()]
        y_coords = [original_data.location(c).y for c in route.visits()]
        barycenters.append([np.mean(x_coords), np.mean(y_coords)])
        route_customer_counts.append(len(route.visits()))

    barycenters = np.array(barycenters)
    route_customer_counts = np.array(route_customer_counts)

    # 2. 检查并调整聚类数量
    k = min(num_clusters, len(routes))
    if k <= 1:
        print("Warning: Number of clusters is <= 1. Decomposition will not be performed.")
        return []

    # 3. K-means 聚类
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(barycenters)

    # 4. 根据聚类标签对路线进行初始分组
    route_groups = [[] for _ in range(k)]
    for idx, label in enumerate(labels):
        route_groups[label].append(idx)  # 存储路线的索引，而非对象


    # --- 新增逻辑：聚类均衡化 ---
    if max_customers_per_cluster is not None:
        print("Balancing clusters...")
        # 修复：确保cluster_customer_counts是整数列表，而不是numpy数组
        cluster_customer_counts = [int(sum(route_customer_counts[route_idx] for route_idx in group)) for group in route_groups]

        # <<< 变化1：将K-means的簇中心复制一份，以便后续动态更新 >>>
        current_cluster_centers = kmeans.cluster_centers_.copy()

        MAX_BALANCE_ITERATIONS = 100
        TOLERANCE = 1.05 
        
        for _ in range(MAX_BALANCE_ITERATIONS):
            overloaded_clusters = [
                (i, count) for i, count in enumerate(cluster_customer_counts) if count > max_customers_per_cluster * TOLERANCE
            ]
            
            if not overloaded_clusters:
                print("Clusters are balanced.")
                break 

            # 修复：确保key返回的是可比较的整数类型
            overloaded_idx = max(overloaded_clusters, key=lambda item: int(item[1]))[0]
            
            # <<< 变化2：使用动态更新的簇中心进行计算 >>>
            cluster_center = current_cluster_centers[overloaded_idx]
            routes_in_cluster = route_groups[overloaded_idx]
            
            if len(routes_in_cluster) <= 1:
                continue 

            distances = [np.linalg.norm(barycenters[route_idx] - cluster_center) for route_idx in routes_in_cluster]
            route_to_move_local_idx = np.argmax(distances)
            route_to_move_global_idx = routes_in_cluster.pop(route_to_move_local_idx)

            candidate_clusters = [i for i, count in enumerate(cluster_customer_counts) if i != overloaded_idx]
            
            if not candidate_clusters: 
                route_groups[overloaded_idx].append(route_to_move_global_idx)
                print("Warning: Could not find a candidate cluster to move the route to.")
                break

            # <<< 变化3：同样使用动态更新的簇中心来寻找最佳新家 >>>
            target_distances = [
                np.linalg.norm(barycenters[route_to_move_global_idx] - current_cluster_centers[i])
                for i in candidate_clusters
            ]
            
            best_new_cluster_idx_local = np.argmin(target_distances)
            best_new_cluster_global_idx = candidate_clusters[best_new_cluster_idx_local]

            # 更新数据
            route_groups[best_new_cluster_global_idx].append(route_to_move_global_idx)
            
            # 更新两个受影响簇的客户总数
            moved_customers_count = int(route_customer_counts[route_to_move_global_idx])
            cluster_customer_counts[overloaded_idx] -= moved_customers_count
            cluster_customer_counts[best_new_cluster_global_idx] += moved_customers_count
            
            # <<< 变化4 (核心修正)：重新计算被移动路线的旧簇和新簇的中心点 >>>
            
            # 更新旧簇 (overloaded_idx) 的中心点
            if route_groups[overloaded_idx]: # 确保簇不为空
                current_cluster_centers[overloaded_idx] = np.mean(barycenters[route_groups[overloaded_idx]], axis=0)
                
            # 更新新簇 (best_new_cluster_global_idx) 的中心点
            current_cluster_centers[best_new_cluster_global_idx] = np.mean(barycenters[route_groups[best_new_cluster_global_idx]], axis=0)

        else: 
            print(f"Warning: Balancing reached max iterations ({MAX_BALANCE_ITERATIONS}) but may not be perfect.")


     # 5. 为每个路线分组创建子问题
    subproblems = []
    subproblem_mappings = []  # 存储映射信息
    og_clients = original_data.clients()
    og_depots = original_data.depots()
    
    # 将路线索引转换回路线对象
    final_route_groups = [[routes[idx] for idx in group] for group in route_groups]

    for cluster_idx, group in enumerate(final_route_groups):
        if not group:
            continue
            
        # 收集该组中涉及的所有客户和depot
        client_locs = []
        depot_locs = []
        
        for route in group:
            # route.visits() 只包含客户，直接收集
            for client_loc_idx in route.visits():
                if client_loc_idx not in client_locs:
                    client_locs.append(client_loc_idx)
            
            # 单独收集depot
            start_depot_idx = route.start_depot()
            end_depot_idx = route.end_depot()
            if start_depot_idx not in depot_locs:
                depot_locs.append(start_depot_idx)
            if end_depot_idx not in depot_locs:
                depot_locs.append(end_depot_idx)

        if not client_locs:
            continue
        
        # 创建新的索引映射：depot在前，客户在后
        all_loc_indices = sorted(depot_locs) + sorted(client_locs)
        old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(all_loc_indices)}
        
        # 创建客户编号映射记录
        client_mapping = {}
        for new_idx, old_idx in enumerate(all_loc_indices):
            if old_idx != 0:  # 非depot
                client_mapping[new_idx] = old_idx
        
        # 创建ProblemData对象（如果需要在内存中使用）
        num_sub_locs = len(all_loc_indices)
        sub_depots = [og_depots[loc_idx] for loc_idx in sorted(depot_locs)]
        sub_clients = [og_clients[loc_idx - len(og_depots)] for loc_idx in sorted(client_locs)]
        
        first_route_type_idx = group[0].vehicle_type()
        og_veh_type = original_data.vehicle_type(first_route_type_idx)
        sub_veh_type = VehicleType(num_available=len(group), capacity=og_veh_type.capacity)
        
        # 创建距离和持续时间矩阵
        og_dist_mat = original_data.distance_matrix(profile=og_veh_type.profile)
        og_dur_mat = original_data.duration_matrix(profile=og_veh_type.profile)
        
        sub_dist_mat = np.zeros((num_sub_locs, num_sub_locs), dtype=og_dist_mat.dtype)
        sub_dur_mat = np.zeros((num_sub_locs, num_sub_locs), dtype=og_dur_mat.dtype)

        for old_frm, new_frm in old_to_new_map.items():
            for old_to, new_to in old_to_new_map.items():
                sub_dist_mat[new_frm, new_to] = og_dist_mat[old_frm, old_to]
                sub_dur_mat[new_frm, new_to] = og_dur_mat[old_frm, old_to]

        sub_data = ProblemData(
            clients=sub_clients,
            depots=sub_depots,
            vehicle_types=[sub_veh_type],
            distance_matrices=[sub_dist_mat],  # type: ignore
            duration_matrices=[sub_dur_mat],   # type: ignore
            groups=[],
        )
        subproblems.append(sub_data)
        
        # 保存映射关系到字典
        subproblem_mappings.append({
            'client_mapping': client_mapping,
            'cluster_id': cluster_idx + 1,
            'old_to_new_map': old_to_new_map
        })
         
    return subproblems, subproblem_mappings



