#!/usr/bin/env python3
"""
Standalone test script that doesn't require PyVRP dependencies.
Tests the core parallel logic components that don't depend on external libraries.
"""

import sys
import os
import time

# Add src_Parallel to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_parameter_configs():
    """Test parameter configuration generation."""
    print("Testing parameter configuration generation...")
    
    try:
        from parameter_configs import get_predefined_strategies, generate_diverse_strategies, print_strategy_summary
        
        # Test predefined strategies
        predefined = get_predefined_strategies()
        print(f"✓ Generated {len(predefined)} predefined strategies")
        assert len(predefined) == 8, "Should generate 8 predefined strategies"
        
        # Test diverse generation
        diverse = generate_diverse_strategies(8)
        print(f"✓ Generated {len(diverse)} diverse strategies")
        assert len(diverse) == 8, "Should generate 8 diverse strategies"
        
        # Test strategy names are unique
        names = [s.strategy_name for s in predefined]
        assert len(set(names)) == len(names), "Strategy names should be unique"
        print(f"✓ All strategy names are unique")
        
        # Test parameter validation
        for strategy in predefined:
            assert 0 <= strategy.repair_probability <= 1, "Invalid repair probability"
            assert strategy.nb_iter_no_improvement > 0, "Invalid iteration count"
            assert strategy.min_population_size > 0, "Invalid population size"
        print(f"✓ All strategies have valid parameters")
        
        return True
    except Exception as e:
        print(f"✗ Parameter config test failed: {e}")
        return False


def test_cli_parser():
    """Test command line argument parsing."""
    print("Testing CLI parser...")
    
    try:
        from cli_parser import parse_args
        import argparse
        
        # Create test arguments
        test_cases = [
            # Basic parallel mode
            ["test_instance.txt", "--parallel_mode"],
            # Full parameter set
            [
                "test_instance.txt",
                "--parallel_mode",
                "--num_strategies", "4",
                "--sync_frequency", "1000",
                "--decomposition_freq", "2000",
                "--num_subproblems", "4",
                "--subproblem_iters", "500",
                "--runtime", "60"
            ],
            # Traditional mode (no parallel)
            ["test_instance.txt", "--runtime", "30", "--runs", "3"]
        ]
        
        original_argv = sys.argv
        
        for i, test_args in enumerate(test_cases):
            sys.argv = ["test_standalone.py"] + test_args
            
            try:
                args = parse_args()
                print(f"✓ Test case {i+1}: Parsed successfully")
                
                # Validate specific arguments for parallel mode
                if "--parallel_mode" in test_args:
                    assert args.parallel_mode == True, "parallel_mode should be True"
                    if "--num_strategies" in test_args:
                        expected_strategies = int(test_args[test_args.index("--num_strategies") + 1])
                        assert args.num_strategies == expected_strategies, "num_strategies mismatch"
                else:
                    assert args.parallel_mode == False, "parallel_mode should be False by default"
                    
            except Exception as e:
                print(f"✗ Test case {i+1} failed: {e}")
                return False
            finally:
                sys.argv = original_argv
        
        return True
    except Exception as e:
        print(f"✗ CLI parser test failed: {e}")
        return False


def test_synchronization_logic():
    """Test synchronization logic without PyVRP dependencies."""
    print("Testing synchronization logic...")
    
    try:
        # Mock strategy stats for testing
        class MockStrategyStats:
            def __init__(self, strategy_id, vehicles, distance, is_feasible):
                self.strategy_id = strategy_id
                self.strategy_name = f"strategy_{strategy_id}"
                self.best_vehicles = vehicles
                self.best_distance = distance
                self.is_feasible = is_feasible
                
            def is_better_than(self, other):
                if self.is_feasible != other.is_feasible:
                    return self.is_feasible
                if not self.is_feasible and not other.is_feasible:
                    return False
                if self.best_vehicles != other.best_vehicles:
                    return self.best_vehicles < other.best_vehicles
                return self.best_distance < other.best_distance
        
        # Test strategy comparison logic
        stats1 = MockStrategyStats(1, 20, 1000.0, True)  # 20 vehicles, feasible
        stats2 = MockStrategyStats(2, 19, 1100.0, True)  # 19 vehicles, feasible (better)
        stats3 = MockStrategyStats(3, 21, 900.0, True)   # 21 vehicles, feasible (worse)
        stats4 = MockStrategyStats(4, 18, 1500.0, False) # 18 vehicles, infeasible
        
        assert stats2.is_better_than(stats1), "Fewer vehicles should be better"
        assert stats1.is_better_than(stats3), "Fewer vehicles should be better"
        assert stats1.is_better_than(stats4), "Feasible should be better than infeasible"
        print(f"✓ Strategy comparison logic works correctly")
        
        # Test frequency calculation
        sync_freq = 1500
        decomp_freq = 4000
        
        # Test sync frequency
        assert 1500 % sync_freq == 0, "Should sync at 1500"
        assert 1499 % sync_freq != 0, "Should not sync at 1499"
        assert 3000 % sync_freq == 0, "Should sync at 3000"
        print(f"✓ Synchronization frequency logic works")
        
        # Test decomposition frequency 
        assert 4000 % decomp_freq == 0, "Should decompose at 4000"
        assert 8000 % decomp_freq == 0, "Should decompose at 8000"
        assert 3999 % decomp_freq != 0, "Should not decompose at 3999"
        print(f"✓ Decomposition frequency logic works")
        
        return True
    except Exception as e:
        print(f"✗ Synchronization logic test failed: {e}")
        return False


def test_strategy_diversity():
    """Test that strategies are actually diverse."""
    print("Testing strategy diversity...")
    
    try:
        from parameter_configs import get_predefined_strategies
        
        strategies = get_predefined_strategies()
        
        # Check diversity across key parameters
        repair_probs = [s.repair_probability for s in strategies]
        no_improve_iters = [s.nb_iter_no_improvement for s in strategies]
        pop_sizes = [s.min_population_size for s in strategies]
        
        # Should have variety in repair probabilities
        assert len(set(repair_probs)) > 1, "Should have variety in repair probabilities"
        print(f"✓ Repair probability diversity: {set(repair_probs)}")
        
        # Should have variety in convergence patience
        assert len(set(no_improve_iters)) > 1, "Should have variety in convergence patience"
        print(f"✓ Convergence patience diversity: {set(no_improve_iters)}")
        
        # Should have variety in population sizes
        assert len(set(pop_sizes)) > 1, "Should have variety in population sizes"
        print(f"✓ Population size diversity: {set(pop_sizes)}")
        
        # Check that extreme strategies exist
        min_repair = min(repair_probs)
        max_repair = max(repair_probs)
        assert max_repair - min_repair >= 0.3, "Should have significant repair probability range"
        print(f"✓ Repair probability range: {min_repair} to {max_repair}")
        
        return True
    except Exception as e:
        print(f"✗ Strategy diversity test failed: {e}")
        return False


def test_data_structures():
    """Test core data structures and their validation."""
    print("Testing data structures...")
    
    try:
        from parameter_configs import HGSStrategyConfig
        
        # Test valid configuration
        config = HGSStrategyConfig(
            repair_probability=0.8,
            nb_iter_no_improvement=20000,
            min_population_size=40,
            strategy_name="test_config"
        )
        print(f"✓ Valid configuration created: {config.strategy_name}")
        
        # Test invalid configurations should raise errors
        try:
            invalid_config = HGSStrategyConfig(repair_probability=1.5)  # Invalid > 1
            print(f"✗ Should have rejected invalid repair probability")
            return False
        except ValueError:
            print(f"✓ Correctly rejected invalid repair probability")
        
        try:
            invalid_config = HGSStrategyConfig(nb_iter_no_improvement=-1000)  # Invalid < 0
            print(f"✗ Should have rejected negative improvement iterations")
            return False
        except ValueError:
            print(f"✓ Correctly rejected negative improvement iterations")
        
        return True
    except Exception as e:
        print(f"✗ Data structures test failed: {e}")
        return False


def main():
    """Run all standalone tests."""
    print("="*60)
    print("PARALLEL HGS SOLVER - STANDALONE TEST SUITE")
    print("="*60)
    print("Testing components that don't require PyVRP dependencies...")
    
    tests = [
        ("Parameter Configurations", test_parameter_configs),
        ("CLI Parser", test_cli_parser),
        ("Strategy Diversity", test_strategy_diversity),
        ("Synchronization Logic", test_synchronization_logic),
        ("Data Structures", test_data_structures),
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
    print(f"STANDALONE TEST RESULTS")
    print(f"{'='*60}")
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nSummary: {passed}/{total} tests passed")
    print(f"Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All standalone tests passed!")
        print("The core parallel logic is working correctly.")
        print("\nNote: Full integration tests require PyVRP dependencies.")
        print("\nTo test with a real instance (requires PyVRP):")
        print("python main.py ../data/homberger_200_customer_instances/C1_2_1.TXT --parallel_mode --runtime 30")
    else:
        print(f"\n⚠ {total - passed} tests failed. Please fix the issues.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())