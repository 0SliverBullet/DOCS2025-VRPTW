# Parallel Multi-Strategy HGS Solver

这是基于原始Route Seekers VRPTW求解器的并行多策略增强版本，通过在8个CPU核心上同时运行不同参数配置的HGS算法来提高求解效率和解质量。

## 核心改进

### 🚀 并行架构设计
- **8个并行策略**：同时运行8个不同参数配置的HGS实例，充分利用8核CPU
- **周期性同步**：每1500迭代进行策略间同步，共享最优解
- **智能分解**：每4000迭代触发barycenter聚类分解，将最优解分解为子问题并行求解
- **CPU资源管理**：确保8个CPU核心在任何时候都不超载

### ⚡ 性能优化
- **CPU利用率**：从原版的12.5%提升到接近100%
- **搜索多样化**：8种不同策略探索不同搜索空间，提高全局最优概率
- **鲁棒性增强**：多策略并行降低陷入局部最优的风险

## 文件结构

### 🆕 新增核心文件
- `parameter_configs.py` - 8种多样化HGS参数配置
- `parallel_hgs_solver.py` - 并行多策略求解器核心
- `solution_synchronizer.py` - 解同步和选择策略管理
- `test_parallel.py` - 测试套件

### 🔧 增强的原有文件  
- `main.py` - 集成并行求解模式选择
- `cli_parser.py` - 添加并行相关命令行参数
- `solve.py` - 添加并行求解入口函数
- `GeneticAlgorithm.py` - 添加外部解注入接口

## 使用方法

### 启用并行模式
```bash
# 基本并行模式
python main.py data/homberger_200_customer_instances/C1_2_1.TXT --parallel_mode --runtime 30

# 自定义并行参数
python main.py data/homberger_800_customer_instances/C1_8_2.TXT \
    --parallel_mode \
    --runtime 1800 \
    --num_strategies 8 \
    --sync_frequency 1500 \
    --decomposition_freq 4000 \
    --num_subproblems 8 \
    --subproblem_iters 1000
```

### 传统单策略模式（向后兼容）
```bash
# 不使用--parallel_mode标志即为原始模式
python main.py data/homberger_200_customer_instances/C1_2_1.TXT --runtime 30
```

## 新增命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--parallel_mode` | False | 启用并行多策略模式 |
| `--num_strategies` | 8 | 并行策略数量（应匹配CPU核心数） |
| `--sync_frequency` | 1500 | 策略同步频率（迭代次数） |
| `--strategy_configs` | None | 自定义策略配置文件路径（可选） |

## 8种预定义策略

1. **Conservative** - 高修复概率，稳定参数
2. **Aggressive** - 低修复概率，快速收敛  
3. **Large Population** - 注重多样性的大种群
4. **Small Population** - 注重局部搜索的小种群
5. **High Intensification** - 高局部搜索强度
6. **Patient** - 长期收敛耐心
7. **Balanced** - 平衡参数配置
8. **Dynamic** - 动态惩罚管理

## 算法流程

```
1. 启动8个并行HGS实例（不同参数配置）
2. 每1500迭代：
   - 暂停所有HGS实例
   - 收集各实例的最佳解
   - 选择全局最优解
   - 将全局最优解分发给所有实例
   - 恢复所有HGS实例
3. 每4000迭代（与同步重合时）：
   - 对全局最优解进行barycenter聚类分解
   - 用8核并行求解8个子问题
   - 合并子问题解决方案
   - 将改进解注入所有HGS实例
```

## 预期性能提升

- **CPU利用率**：从12.5% → 接近100%
- **解质量**：多策略并行应能找到更好的解
- **鲁棒性**：对不同实例特征更加鲁棒
- **可扩展性**：支持轻松添加新的策略配置

## 测试运行

```bash
# 运行测试套件
cd src_Parallel
python test_parallel.py

# 快速测试并行模式（30秒）
python main.py ../data/homberger_200_customer_instances/C1_2_1.TXT --parallel_mode --runtime 30

# 完整测试（30分钟）
python main.py ../data/homberger_800_customer_instances/C1_8_2.TXT --parallel_mode --runtime 1800
```

## 依赖要求

需要原始项目的所有依赖：
- PyVRP
- vrplib  
- numpy
- concurrent.futures (标准库)
- threading (标准库)
- queue (标准库)

## 注意事项

1. **内存使用**：并行模式会使用更多内存（8个独立种群）
2. **调试模式**：测试时可以减少`num_strategies`以便调试
3. **最佳实践**：`num_strategies`应等于CPU核心数以获得最佳性能
4. **兼容性**：完全向后兼容原始单策略模式

## 实现状态

✅ **已完成**：
- 核心并行架构
- 8种策略配置
- 同步机制
- 分解集成
- CLI接口  
- 基础测试

⚠️ **需要运行时环境**：
- PyVRP库安装
- vrplib库安装  
- 完整依赖环境

## 贡献

基于Route Seekers团队的原始实现，并行多策略增强由AI助手开发。