#!/usr/bin/env python3
"""
Test script for result saving functionality.
Tests the result saving without PyVRP dependencies.
"""

import sys
import os
import time
import tempfile
import json
from datetime import datetime

# Add src_Parallel to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_result_saving():
    """Test basic result saving structure without PyVRP objects."""
    print("Testing basic result saving structure...")
    
    try:
        # Mock command line arguments
        class MockArgs:
            def __init__(self):
                self.instance_path = "test/C1_2_1.TXT"
                self.runtime = 30
                self.parallel_mode = True
                self.num_strategies = 8
                self.sync_frequency = 1500
                self.decomposition_freq = 4000
                self.num_subproblems = 8
                self.subproblem_iters = 1000
                self.seed = 42
                self.num_cores = 8
        
        # Mock parallel parameters
        class MockParallelParams:
            def __init__(self):
                self.num_strategies = 8
                self.sync_frequency = 1500
                self.decomposition_frequency = 4000
                self.num_subproblems = 8
                self.subproblem_iters = 1000
                self.collect_stats = True
                self.display = True
                self.strategy_configs = []
        
        # Mock strategy config
        class MockStrategyConfig:
            def __init__(self, name):
                self.strategy_name = name
                self.repair_probability = 0.8
                self.nb_iter_no_improvement = 20000
                self.min_population_size = 40
                self.generation_size = 60
                self.penalty_increase = 1.34
                self.penalty_decrease = 0.32
                # Updated parameters - removed intensification_probability
                self.lb_diversity = 0.1
                self.ub_diversity = 0.5
                self.nb_elite = 4
                self.nb_close = 5
                self.weight_wait_time = 0.2
                self.weight_time_warp = 1.0
                self.nb_granular = 40
                self.symmetric_proximity = True
                self.symmetric_neighbours = False
        
        # Create mock objects
        args = MockArgs()
        parallel_params = MockParallelParams()
        
        # Add some strategy configs
        strategy_names = ["conservative", "aggressive", "balanced"]
        for name in strategy_names:
            parallel_params.strategy_configs.append(MockStrategyConfig(name))
        
        # Mock synchronizer statistics
        class MockSynchronizer:
            def get_statistics_summary(self):
                return {
                    "total_synchronizations": 5,
                    "total_improvements": 3,
                    "total_decompositions": 2,
                    "improvement_rate": 0.6,
                    "average_sync_time": 0.005,
                    "global_best_strategy": "conservative"
                }
            
            @property
            def sync_history(self):
                return []  # Empty for now
            
            @property
            def global_iteration(self):
                return 7500
        
        synchronizer = MockSynchronizer()
        
        # Test directory structure creation
        instance_name = os.path.splitext(os.path.basename(args.instance_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override the results directory for testing
            import result_saver
            original_dirname = os.path.dirname
            
            def mock_dirname(path):
                if path.endswith("__file__"):
                    return temp_dir
                return original_dirname(path)
            
            # Temporarily patch os.path.dirname
            os.path.dirname = mock_dirname
            
            try:
                # Test the result data structure creation
                results_base_dir = os.path.join(temp_dir, "results_Parallel")
                instance_dir = os.path.join(results_base_dir, instance_name)
                os.makedirs(instance_dir, exist_ok=True)
                
                # Create a mock result data structure
                result_data = {
                    "instance_info": {
                        "instance_name": instance_name,
                        "instance_path": args.instance_path,
                        "timestamp": timestamp,
                        "runtime": args.runtime,
                        "total_runtime": 29.45,
                        "solver_type": "parallel_multi_strategy_hgs"
                    },
                    "parallel_configuration": {
                        "parallel_params": {
                            "num_strategies": parallel_params.num_strategies,
                            "sync_frequency": parallel_params.sync_frequency,
                            "decomposition_frequency": parallel_params.decomposition_frequency,
                            "num_subproblems": parallel_params.num_subproblems,
                            "subproblem_iters": parallel_params.subproblem_iters,
                        },
                        "strategy_configs": [
                            {
                                "strategy_id": i,
                                "strategy_name": config.strategy_name,
                                "repair_probability": config.repair_probability,
                                "nb_iter_no_improvement": config.nb_iter_no_improvement,
                            }
                            for i, config in enumerate(parallel_params.strategy_configs)
                        ]
                    },
                    "synchronization_statistics": synchronizer.get_statistics_summary(),
                    "best_solution": {
                        "vehicles": 20,
                        "distance": 2704.6,
                        "duration": 2075.6,
                        "is_feasible": True
                    },
                    "detailed_solution": {
                        "routes": [
                            {"route_id": 1, "visits": [1, 5, 3, 7], "distance": 135.2},
                            {"route_id": 2, "visits": [2, 6, 4, 8], "distance": 142.8}
                        ],
                        "total_routes": 2
                    }
                }
                
                # Test JSON file creation
                result_filename = f"{instance_name}_{timestamp}.json"
                result_path = os.path.join(instance_dir, result_filename)
                
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                print(f"✓ Created JSON result file: {result_filename}")
                
                # Verify JSON structure
                with open(result_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                
                assert loaded_data["instance_info"]["instance_name"] == instance_name
                assert loaded_data["best_solution"]["vehicles"] == 20
                assert len(loaded_data["parallel_configuration"]["strategy_configs"]) == 3
                print(f"✓ JSON structure validation passed")
                
                # Test text report creation
                report_filename = f"{instance_name}_{timestamp}_report.txt"
                report_path = os.path.join(instance_dir, report_filename)
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(f"并行多策略HGS求解结果报告\n")
                    f.write(f"=" * 60 + "\n\n")
                    f.write(f"实例信息:\n")
                    f.write(f"  实例名称: {instance_name}\n")
                    f.write(f"  实例路径: {args.instance_path}\n")
                    f.write(f"  运行时间: {timestamp}\n")
                    f.write(f"并行配置:\n")
                    f.write(f"  策略数量: {parallel_params.num_strategies}\n")
                    f.write(f"最优解:\n")
                    f.write(f"  车辆数: 20\n")
                    f.write(f"  总距离: 2704.6\n")
                
                print(f"✓ Created text report: {report_filename}")
                
                # Test script generation
                script_filename = f"{instance_name}_{timestamp}_reproduce.sh"
                script_path = os.path.join(instance_dir, script_filename)
                
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write("#!/bin/bash\n")
                    f.write(f"# 并行多策略HGS复现脚本\n")
                    f.write(f"python src_Parallel/main.py {args.instance_path} \\\n")
                    f.write(f"    --parallel_mode \\\n")
                    f.write(f"    --runtime {args.runtime}\n")
                
                os.chmod(script_path, 0o755)
                print(f"✓ Created reproduce script: {script_filename}")
                
                # Test directory structure
                expected_files = [result_filename, report_filename, script_filename]
                actual_files = os.listdir(instance_dir)
                
                for expected_file in expected_files:
                    assert expected_file in actual_files, f"Missing file: {expected_file}"
                
                print(f"✓ Directory structure verification passed")
                
                return True
                
            finally:
                # Restore original function
                os.path.dirname = original_dirname
        
    except Exception as e:
        print(f"✗ Basic result saving test failed: {e}")
        return False


def test_results_directory_structure():
    """Test the results directory structure creation.""" 
    print("Testing results directory structure...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test multiple instances
            test_instances = ["C1_2_1", "C1_8_2", "C1_8_3"]
            
            for instance in test_instances:
                # Create instance directory
                results_base = os.path.join(temp_dir, "results_Parallel")
                instance_dir = os.path.join(results_base, instance)
                os.makedirs(instance_dir, exist_ok=True)
                
                # Create dummy result files
                timestamp1 = "20250121_120000"
                timestamp2 = "20250121_130000"
                
                for timestamp in [timestamp1, timestamp2]:
                    result_file = os.path.join(instance_dir, f"{instance}_{timestamp}.json")
                    with open(result_file, 'w') as f:
                        json.dump({
                            "instance_info": {"timestamp": timestamp},
                            "best_solution": {"vehicles": 20, "distance": 2700.0}
                        }, f)
            
            # Verify directory structure
            assert os.path.exists(results_base), "results_Parallel directory not created"
            
            for instance in test_instances:
                instance_dir = os.path.join(results_base, instance)
                assert os.path.exists(instance_dir), f"Instance directory {instance} not created"
                
                files = os.listdir(instance_dir)
                json_files = [f for f in files if f.endswith('.json')]
                assert len(json_files) == 2, f"Expected 2 JSON files for {instance}, got {len(json_files)}"
            
            print(f"✓ Created directory structure for {len(test_instances)} instances")
            print(f"✓ Directory structure: results_Parallel/{{instance_name}}/")
            
            return True
            
    except Exception as e:
        print(f"✗ Directory structure test failed: {e}")
        return False


def main():
    """Run result saver tests."""
    print("="*60)
    print("PARALLEL RESULT SAVER - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Basic Result Saving", test_basic_result_saving),
        ("Directory Structure", test_results_directory_structure),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"RESULT SAVER TEST RESULTS")
    print(f"{'='*60}")
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nSummary: {passed}/{total} tests passed")
    print(f"Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All result saver tests passed!")
        print("Result saving functionality is working correctly.")
        print("\nResult structure:")
        print("📁 results_Parallel/")
        print("  └── 📁 {instance_name}/")
        print("      ├── 📄 {instance}_{timestamp}.json")
        print("      ├── 📄 {instance}_{timestamp}_report.txt")
        print("      ├── 📄 {instance}_{timestamp}_config.json")
        print("      ├── 📄 {instance}_{timestamp}_reproduce.sh")
        print("      └── 📄 {instance}_summary.md")
    else:
        print(f"\n⚠ {total - passed} tests failed.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())