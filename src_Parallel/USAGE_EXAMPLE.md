# 并行多策略HGS求解器 - 使用示例

## 快速开始

### 1. 测试基础功能（无需PyVRP）
```bash
cd src_Parallel

# 运行独立测试（测试核心逻辑）
python test_standalone.py

# 运行完整测试（会跳过依赖PyVRP的部分）
python test_parallel.py
```

### 2. 设置PyVRP环境（用于实际求解）
```bash
# 安装PyVRP依赖（需要在原始src/目录的Python环境中）
pip install pyvrp vrplib numpy

# 或者使用conda
conda install pyvrp vrplib numpy
```

### 3. 运行并行求解器

#### 启用并行模式（8策略并行）：
```bash
# 200客户实例 - 30秒快速测试
python main.py ../data/homberger_200_customer_instances/C1_2_1.TXT --parallel_mode --runtime 30

# 800客户实例 - 完整30分钟求解
python main.py ../data/homberger_800_customer_instances/C1_8_2.TXT --parallel_mode --runtime 1800

# 自定义参数
python main.py ../data/homberger_800_customer_instances/C1_8_3.TXT \
    --parallel_mode \
    --runtime 1800 \
    --num_strategies 8 \
    --sync_frequency 1500 \
    --decomposition_freq 4000 \
    --num_subproblems 8
```

#### 传统单策略模式（向后兼容）：
```bash
# 原始单核模式
python main.py ../data/homberger_200_customer_instances/C1_2_1.TXT --runtime 30
```

## 参数说明

### 并行模式专用参数
- `--parallel_mode`: 启用8核并行多策略模式
- `--num_strategies`: 并行策略数量（默认8，建议等于CPU核心数）
- `--sync_frequency`: 策略同步频率，单位：迭代次数（默认1500）

### 分解相关参数  
- `--decomposition_freq`: 分解触发频率，单位：迭代次数（默认4000）
- `--num_subproblems`: 分解子问题数量（默认8）
- `--subproblem_iters`: 子问题求解迭代次数（默认1000）

### 通用参数
- `--runtime`: 总求解时间，单位：秒（默认1800）
- `--seed`: 随机种子（默认42）

## 输出示例

### 并行模式输出
```
Parallel multi-strategy mode enabled!
Solving ../data/C1_2_1.TXT with parallel multi-strategy HGS...
Runtime: 30 seconds
Number of strategies: 8
Sync frequency: 1500 iterations
Decomposition frequency: 4000 iterations
Subproblems: 8

================================================================================
PARALLEL MULTI-STRATEGY HGS CONFIGURATION
================================================================================
Number of strategies: 8
Synchronization frequency: 1500 iterations
Decomposition frequency: 4000 iterations
Subproblems for decomposition: 8

Strategy Details:
--------------------------------------------------------------------------------
Strategy 1: conservative
  Repair Prob: 1.0, No Improve: 25000, Pop Size: 40
Strategy 2: aggressive
  Repair Prob: 0.6, No Improve: 15000, Pop Size: 25
... (8种策略详情)

============================================================
SYNCHRONIZATION SUMMARY - Iteration 1500
============================================================
Global Best Strategy: conservative (ID: 0)
Best Solution: 20 vehicles, distance: 2704.6, duration: 2075.6
Feasible: True
*** GLOBAL IMPROVEMENT FOUND! ***
Synchronization time: 0.003s

Strategy Performance:
----------------------------------------
BEST conservative: 20v, 2704.6d, feasible: True
     aggressive: 21v, 2750.3d, feasible: True
     balanced: 20v, 2720.1d, feasible: True
...
============================================================

=== Parallel Execution Results ===
Total runtime: 30.45 seconds
Best solution found:
  Vehicles: 20
  Distance: 2704.6
  Duration: 2075.6
  Feasible: True

============================================================
BEST SOLUTION DETAILS:
============================================================
Route 1: 0 -> 5 -> 3 -> 7 -> 8 -> 6 -> 4 -> 2 -> 1 -> 0
Route 2: 0 -> 15 -> 13 -> 17 -> 18 -> 16 -> 14 -> 12 -> 11 -> 0
...
```

## 性能对比

| 模式 | CPU利用率 | 策略数量 | 预期性能 |
|------|----------|----------|----------|
| 传统模式 | ~12.5% | 1 | 基准 |
| 并行模式 | ~100% | 8 | 明显提升 |

## 故障排除

### 1. 导入错误
```bash
ModuleNotFoundError: No module named 'pyvrp'
```
**解决**：安装PyVRP依赖或运行独立测试
```bash
pip install pyvrp vrplib
# 或者只测试核心逻辑
python test_standalone.py
```

### 2. 内存不足
并行模式使用更多内存，如遇问题可减少策略数量：
```bash
python main.py data.txt --parallel_mode --num_strategies 4
```

### 3. CPU核心数不匹配
建议设置`--num_strategies`等于系统CPU核心数：
```bash
# 查看CPU核心数
python -c "import os; print(f'CPU cores: {os.cpu_count()}')"

# 使用对应数量的策略
python main.py data.txt --parallel_mode --num_strategies 8
```

## 预期改进

相比原始单策略实现：
- ✅ **CPU利用率**: 12.5% → 接近100%
- ✅ **解质量**: 多策略并行探索更大搜索空间
- ✅ **鲁棒性**: 降低陷入局部最优的概率  
- ✅ **可扩展性**: 支持添加新的策略配置

## 技术支持

1. 运行测试：`python test_standalone.py`
2. 检查文档：`README_Parallel.md`  
3. 查看策略配置：`parameter_configs.py`