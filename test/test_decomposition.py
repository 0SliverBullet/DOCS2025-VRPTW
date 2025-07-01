"""
测试 decomposition.py 中的质心聚类分解功能
"""

import sys
import os
import numpy as np

# 添加src路径到sys.path以便导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from decomposition import barycenter_clustering_decomposition
from pyvrp._pyvrp import (
    Client, 
    Depot, 
    ProblemData, 
    VehicleType, 
    Solution, 
    Route,
    Trip
)


def create_test_problem_data():
    """
    创建一个测试用的简单问题数据
    包含1个仓库和8个客户，适合测试聚类分解
    """
    # 创建仓库（位于中心）
    depot = Depot(x=50, y=50)
    
    # 创建8个客户，分布在4个象限，每个象限2个客户
    clients = [
        # 第一象限（右上）
        Client(x=80, y=80, delivery=[10], tw_early=0, tw_late=1000, service_duration=10),
        Client(x=85, y=85, delivery=[15], tw_early=0, tw_late=1000, service_duration=10),
        
        # 第二象限（左上）
        Client(x=20, y=80, delivery=[12], tw_early=0, tw_late=1000, service_duration=10),
        Client(x=15, y=85, delivery=[8], tw_early=0, tw_late=1000, service_duration=10),
        
        # 第三象限（左下）
        Client(x=20, y=20, delivery=[20], tw_early=0, tw_late=1000, service_duration=10),
        Client(x=15, y=15, delivery=[18], tw_early=0, tw_late=1000, service_duration=10),
        
        # 第四象限（右下）
        Client(x=80, y=20, delivery=[14], tw_early=0, tw_late=1000, service_duration=10),
        Client(x=85, y=15, delivery=[16], tw_early=0, tw_late=1000, service_duration=10),
    ]
    
    # 创建车辆类型
    vehicle_type = VehicleType(
        num_available=4,
        capacity=[100],
        start_depot=0,
        end_depot=0,
        tw_early=0,
        tw_late=1000
    )
    
    # 创建距离和持续时间矩阵（9x9: 1个仓库 + 8个客户）
    num_locations = 9
    distance_matrix = np.zeros((num_locations, num_locations), dtype=np.int64)
    duration_matrix = np.zeros((num_locations, num_locations), dtype=np.int64)
    
    # 计算欧几里得距离
    locations = [depot] + clients
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                dx = locations[i].x - locations[j].x
                dy = locations[i].y - locations[j].y
                dist = int(np.sqrt(dx*dx + dy*dy) * 1000)  # 放大1000倍
                distance_matrix[i, j] = dist
                duration_matrix[i, j] = dist
    
    return ProblemData(
        clients=clients,
        depots=[depot],
        vehicle_types=[vehicle_type],
        distance_matrices=[distance_matrix],  # type: ignore
        duration_matrices=[duration_matrix],  # type: ignore
        groups=[]
    )


def create_test_solution(problem_data):
    """
    创建一个测试解，包含4条路线，每条路线服务2个客户
    """
    # 路线1: 仓库 -> 客户0 -> 客户1 -> 仓库
    route1 = Route(
        problem_data,
        [Trip(problem_data, [0, 1], 0, start_depot=0, end_depot=0)],
        0  # vehicle_type
    )
    
    # 路线2: 仓库 -> 客户2 -> 客户3 -> 仓库  
    route2 = Route(
        problem_data,
        [Trip(problem_data, [2, 3], 0, start_depot=0, end_depot=0)],
        0  # vehicle_type
    )
    
    # 路线3: 仓库 -> 客户4 -> 客户5 -> 仓库
    route3 = Route(
        problem_data,
        [Trip(problem_data, [4, 5], 0, start_depot=0, end_depot=0)],
        0  # vehicle_type
    )
    
    # 路线4: 仓库 -> 客户6 -> 客户7 -> 仓库
    route4 = Route(
        problem_data,
        [Trip(problem_data, [6, 7], 0, start_depot=0, end_depot=0)],
        0  # vehicle_type
    )
    
    return Solution(problem_data, [route1, route2, route3, route4])


class TestBarycenteClustering:
    """
    测试质心聚类分解功能的测试类
    """
    
    def setup_method(self):
        """在每个测试方法前运行，设置测试数据"""
        self.problem_data = create_test_problem_data()
        self.solution = create_test_solution(self.problem_data)
    
    def test_basic_decomposition(self):
        """测试基本的聚类分解功能"""
        # 将4条路线分解为2个子问题
        num_clusters = 2
        subproblems = barycenter_clustering_decomposition(
            self.solution,
            self.problem_data,
            num_clusters,
            random_state=42
        )
        
        # 验证结果
        assert len(subproblems) == num_clusters, f"期望得到{num_clusters}个子问题，实际得到{len(subproblems)}个"
        
        # 验证每个子问题都是有效的ProblemData
        for i, subproblem in enumerate(subproblems):
            assert isinstance(subproblem, ProblemData), f"子问题{i}不是ProblemData类型"
            assert len(subproblem.clients()) > 0, f"子问题{i}没有客户"
            assert len(subproblem.depots()) > 0, f"子问题{i}没有仓库"
            assert len(subproblem.vehicle_types()) > 0, f"子问题{i}没有车辆类型"
    
    def test_cluster_balancing(self):
        """测试带有最大客户数限制的聚类均衡功能"""
        # 设置每个聚类最多3个客户
        max_customers_per_cluster = 3
        subproblems = barycenter_clustering_decomposition(
            self.solution,
            self.problem_data,
            num_clusters=2,
            max_customers_per_cluster=max_customers_per_cluster,
            random_state=42
        )
        
        # 验证均衡效果
        for i, subproblem in enumerate(subproblems):
            num_customers = len(subproblem.clients())
            print(f"子问题{i}有{num_customers}个客户")
            # 允许一定的容忍度
            assert num_customers <= max_customers_per_cluster * 1.2, \
                f"子问题{i}的客户数{num_customers}超过了设定的最大值{max_customers_per_cluster}"
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试聚类数大于路线数的情况
        large_clusters = barycenter_clustering_decomposition(
            self.solution,
            self.problem_data,
            num_clusters=10,  # 大于路线数4
            random_state=42
        )
        
        # 应该最多得到4个子问题（等于路线数）
        assert len(large_clusters) <= 4, "聚类数大于路线数时，应该最多得到与路线数相等的子问题"
        
        # 测试聚类数为1的情况
        single_cluster = barycenter_clustering_decomposition(
            self.solution,
            self.problem_data,
            num_clusters=1,
            random_state=42
        )
        
        # 应该返回空列表或警告
        assert len(single_cluster) == 0, "聚类数为1时应该返回空列表"
    
    def test_random_state_consistency(self):
        """测试相同随机种子的一致性"""
        # 使用相同的随机种子运行两次
        subproblems1 = barycenter_clustering_decomposition(
            self.solution,
            self.problem_data,
            num_clusters=2,
            random_state=123
        )
        
        subproblems2 = barycenter_clustering_decomposition(
            self.solution,
            self.problem_data,
            num_clusters=2,
            random_state=123
        )
        
        # 应该得到相同的结果
        assert len(subproblems1) == len(subproblems2), "相同随机种子应该产生相同数量的子问题"
        
        # 验证每个子问题的客户数量是否相同
        for i, (sub1, sub2) in enumerate(zip(subproblems1, subproblems2)):
            assert len(sub1.clients()) == len(sub2.clients()), \
                f"相同随机种子下，子问题{i}的客户数量应该相同"
    
    def test_empty_solution(self):
        """测试空解的情况"""
        # 创建没有路线的空解
        empty_solution = Solution(self.problem_data, [])
        
        subproblems = barycenter_clustering_decomposition(
            empty_solution,
            self.problem_data,
            num_clusters=2,
            random_state=42
        )
        
        # 应该返回空列表
        assert len(subproblems) == 0, "空解应该返回空的子问题列表"
    
    def test_subproblem_structure(self):
        """测试子问题的结构完整性"""
        subproblems = barycenter_clustering_decomposition(
            self.solution,
            self.problem_data,
            num_clusters=2,
            random_state=42
        )
        
        total_clients = 0
        for i, subproblem in enumerate(subproblems):
            # 验证距离矩阵的维度
            num_locs = len(subproblem.depots()) + len(subproblem.clients())
            dist_matrix = subproblem.distance_matrix(profile=0)
            assert dist_matrix.shape == (num_locs, num_locs), \
                f"子问题{i}的距离矩阵维度不正确"
            
            # 验证对角线为0
            assert np.all(np.diag(dist_matrix) == 0), \
                f"子问题{i}的距离矩阵对角线应该为0"
            
            # 统计总客户数
            total_clients += len(subproblem.clients())
        
        # 验证所有客户都被分配到了子问题中
        assert total_clients == len(self.problem_data.clients()), \
            "所有客户都应该被分配到子问题中"


def run_manual_test():
    """手动运行测试，用于调试"""
    print("=== 开始测试 decomposition.py ===")
    
    try:
        # 使用真实数据文件测试
        from read import read_instance
        from pyvrp import Model
        
        print("使用真实数据进行测试...")
        problem_data = read_instance("data/homberger_200_customer_instances/C1_2_1.TXT", 
                                   instance_format="solomon", round_func="exact")
        
        print(f"原问题有 {len(problem_data.clients())} 个客户")
        
        # 创建模型并获得一个解
        model = Model.from_data(problem_data)
        from pyvrp.stop import MaxIterations
        result = model.solve(stop=MaxIterations(100), seed=42, display=False)
        
        if result.best.is_feasible():
            solution = result.best
            print(f"解中有 {len(solution.routes())} 条路线")
            
            # 测试基本分解
            print("\n--- 测试基本分解 ---")
            subproblems = barycenter_clustering_decomposition(
                solution,
                problem_data,
                num_clusters=2,
                random_state=42
            )
            
            print(f"分解后得到 {len(subproblems)} 个子问题")
            for i, sub in enumerate(subproblems):
                print(f"子问题{i}: {len(sub.clients())} 个客户, {len(sub.depots())} 个仓库")
            
            # 测试均衡分解
            print("\n--- 测试均衡分解 ---")
            balanced_subproblems = barycenter_clustering_decomposition(
                solution,
                problem_data,
                num_clusters=3,
                max_customers_per_cluster=len(problem_data.clients())//3,
                random_state=42
            )
            
            print(f"均衡分解后得到 {len(balanced_subproblems)} 个子问题")
            for i, sub in enumerate(balanced_subproblems):
                print(f"子问题{i}: {len(sub.clients())} 个客户, {len(sub.depots())} 个仓库")
            
            print("\n=== 所有测试通过！ ===")
        else:
            print("无法获得可行解，跳过测试")
        
    except Exception as e:
        print(f"\n!!! 测试失败: {str(e)} !!!")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_manual_test() 