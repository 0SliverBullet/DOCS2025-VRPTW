"""
Solution synchronizer for parallel multi-strategy HGS solver.
Handles synchronization, solution selection, and elite solution sharing.
"""

from __future__ import annotations
import time
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import concurrent.futures

from pyvrp._pyvrp import Solution, ProblemData


@dataclass
class StrategyStats:
    """Statistics for a single strategy during parallel execution."""
    
    strategy_id: int
    strategy_name: str
    current_iteration: int
    best_cost: float
    best_vehicles: int
    best_distance: float
    best_duration: float
    is_feasible: bool
    iterations_no_improvement: int
    last_improvement_iteration: int
    total_time: float
    avg_population_cost: float = 0.0  # Average cost of feasible solutions in current population
    avg_feasible_cost: float = 0.0    # Average cost of feasible solutions only
    feasible_population_count: int = 0  # Number of feasible solutions in population
    
    def is_better_than(self, other: 'StrategyStats') -> bool:
        """
        Compare this strategy's performance with another.
        Priority: feasibility -> vehicles -> distance -> duration
        """
        if self.is_feasible != other.is_feasible:
            return self.is_feasible  # Feasible solutions are better
        
        if not self.is_feasible and not other.is_feasible:
            return self.best_cost < other.best_cost  # Both infeasible, compare cost
            
        # Both feasible, compare by hierarchy
        if self.best_vehicles != other.best_vehicles:
            return self.best_vehicles < other.best_vehicles
            
        if abs(self.best_distance - other.best_distance) > 0.1:
            return self.best_distance < other.best_distance
            
        return self.best_duration < other.best_duration


@dataclass
class SynchronizationResult:
    """Result of a synchronization operation."""
    
    global_best_strategy_id: int
    global_best_solution: Optional[Solution]
    global_best_stats: StrategyStats
    improvement_found: bool
    decomposition_triggered: bool
    sync_time: float
    strategies_stats: List[StrategyStats]


class SolutionSynchronizer:
    """
    Manages synchronization of solutions across parallel HGS strategies.
    """
    
    def __init__(self, 
                 data: ProblemData,
                 sync_frequency: int = 1500,
                 decomposition_frequency: int = 4000,
                 num_subproblems: int = 8,
                 subproblem_iters: int = 1000):
        """
        Initialize the solution synchronizer.
        
        Parameters
        ----------
        data : ProblemData
            Problem data instance
        sync_frequency : int
            Frequency of synchronization (in iterations)
        decomposition_frequency : int
            Frequency of decomposition (in iterations)
        num_subproblems : int
            Number of subproblems for decomposition
        subproblem_iters : int
            Iterations for subproblem solving
        """
        self.data = data
        self.sync_frequency = sync_frequency
        self.decomposition_frequency = decomposition_frequency
        self.num_subproblems = num_subproblems
        self.subproblem_iters = subproblem_iters
        
        # State tracking
        self.global_iteration = 0
        self.last_sync_iteration = 0
        self.last_decomposition_iteration = 0
        self.global_best_solution: Optional[Solution] = None
        self.global_best_stats: Optional[StrategyStats] = None
        
        # Statistics
        self.sync_history: List[SynchronizationResult] = []
        self.improvement_history: List[Tuple[int, float, int]] = []  # (iteration, cost, vehicles)
        
    def should_synchronize(self, current_iteration: int) -> bool:
        """Check if synchronization should be triggered."""
        return (current_iteration - self.last_sync_iteration) >= self.sync_frequency
    
    def should_decompose(self, current_iteration: int) -> bool:
        """Check if decomposition should be triggered."""
        return (current_iteration - self.last_decomposition_iteration) >= self.decomposition_frequency
    
    def synchronize_solutions(self, 
                            strategies_data: List[Tuple[int, Solution, Dict[str, Any]]]) -> SynchronizationResult:
        """
        Synchronize solutions across all parallel strategies.
        
        Parameters
        ----------
        strategies_data : List[Tuple[int, Solution, Dict[str, Any]]]
            List of (strategy_id, current_best_solution, stats_dict) for each strategy
            
        Returns
        -------
        SynchronizationResult
            Result of the synchronization operation
        """
        sync_start_time = time.time()
        
        # Extract statistics from each strategy
        strategies_stats = []
        for strategy_id, solution, stats_dict in strategies_data:
            stats = self._extract_strategy_stats(strategy_id, solution, stats_dict)
            strategies_stats.append(stats)
        
        # Find the globally best strategy using proper comparison
        best_strategy_stats = strategies_stats[0]  # Initialize with first strategy
        for stats in strategies_stats[1:]:
            if stats.is_better_than(best_strategy_stats):
                best_strategy_stats = stats
        
        # Get the corresponding solution
        global_best_solution = None
        global_best_strategy_id = best_strategy_stats.strategy_id
        
        for strategy_id, solution, _ in strategies_data:
            if strategy_id == global_best_strategy_id:
                global_best_solution = solution
                break
        
        # Check for improvement
        improvement_found = self._is_global_improvement(best_strategy_stats)
        
        if improvement_found:
            self.global_best_solution = global_best_solution
            self.global_best_stats = best_strategy_stats
            self.improvement_history.append((
                self.global_iteration,
                best_strategy_stats.best_cost,
                best_strategy_stats.best_vehicles
            ))
        
        # Check if decomposition should be triggered
        decomposition_triggered = self.should_decompose(self.global_iteration)
        
        if decomposition_triggered:
            self.last_decomposition_iteration = self.global_iteration
        
        self.last_sync_iteration = self.global_iteration
        sync_time = time.time() - sync_start_time
        
        # Ensure we have a valid global_best_solution
        if global_best_solution is None:
            # Use first available solution as fallback
            if strategies_data:
                global_best_solution = strategies_data[0][1]
        
        # Create synchronization result
        sync_result = SynchronizationResult(
            global_best_strategy_id=global_best_strategy_id,
            global_best_solution=global_best_solution or strategies_data[0][1] if strategies_data else None,
            global_best_stats=best_strategy_stats,
            improvement_found=improvement_found,
            decomposition_triggered=decomposition_triggered,
            sync_time=sync_time,
            strategies_stats=strategies_stats
        )
        
        self.sync_history.append(sync_result)
        
        return sync_result
    
    def _extract_strategy_stats(self, 
                              strategy_id: int, 
                              solution: Solution, 
                              stats_dict: Dict[str, Any]) -> StrategyStats:
        """Extract strategy statistics from solution and stats dictionary."""
        
        # Calculate solution metrics
        is_feasible = solution.is_feasible()
        vehicles = solution.num_routes()
        distance = round(solution.distance() / 10, 1)
        duration = round(solution.duration() / 10, 1)
        # Calculate cost - PyVRP Solution doesn't have cost() method
        if is_feasible:
            cost = distance + duration  # Use distance + duration as cost approximation
        else:
            cost = float('inf')
        
        return StrategyStats(
            strategy_id=strategy_id,
            strategy_name=stats_dict.get('strategy_name', f'strategy_{strategy_id}'),
            current_iteration=stats_dict.get('current_iteration', 0),
            best_cost=cost,
            best_vehicles=vehicles,
            best_distance=distance,
            best_duration=duration,
            is_feasible=is_feasible,
            iterations_no_improvement=stats_dict.get('iterations_no_improvement', 0),
            last_improvement_iteration=stats_dict.get('last_improvement_iteration', 0),
            total_time=stats_dict.get('total_time', 0.0),
            avg_population_cost=stats_dict.get('avg_population_cost', 0.0),
            avg_feasible_cost=stats_dict.get('avg_feasible_cost', 0.0),
            feasible_population_count=stats_dict.get('feasible_population_count', 0)
        )
    
    def _is_global_improvement(self, new_stats: StrategyStats) -> bool:
        """Check if the new solution is a global improvement."""
        if self.global_best_stats is None:
            return True
        
        return new_stats.is_better_than(self.global_best_stats)
    
    def trigger_decomposition(self, 
                            elite_solution: Solution,
                            num_cores: int = 8) -> Optional[Solution]:
        """
        Trigger decomposition of the elite solution.
        
        Parameters
        ----------
        elite_solution : Solution
            Elite solution to decompose
        num_cores : int
            Number of CPU cores for parallel subproblem solving
            
        Returns
        -------
        Optional[Solution]
            Improved solution from decomposition, or None if failed
        """
        try:
            # Import decomposition function
            from decomposition import barycenter_clustering_decomposition
            
            print(f"Starting decomposition with {self.num_subproblems} subproblems...")
            
            # Perform decomposition
            subproblems, subproblem_mappings = barycenter_clustering_decomposition(
                elite_solution,
                self.data,
                self.num_subproblems,
                max_customers_per_cluster=len(self.data.clients()) // self.num_subproblems,
                random_state=self.global_iteration
            )
            
            if not subproblems:
                print("Decomposition failed to generate subproblems.")
                return None
            
            # Parallel solve subproblems
            improved_solution = self._solve_subproblems_parallel(
                subproblems, 
                subproblem_mappings, 
                num_cores
            )
            
            return improved_solution
            
        except Exception as e:
            print(f"Decomposition failed: {e}")
            return None
    
    def _solve_subproblems_parallel(self, 
                                  subproblems: List[ProblemData], 
                                  subproblem_mappings: List[Dict],
                                  num_cores: int) -> Optional[Solution]:
        """Solve subproblems in parallel and merge results."""
        
        try:
            # Import required modules
            from solve import solve as solve_subproblem
            from pyvrp.stop import MaxIterations
            
            def solve_subproblem_worker(args):
                idx, sub_data, mapping = args
                try:
                    stop_sub = MaxIterations(self.subproblem_iters)
                    result = solve_subproblem(sub_data, stop=stop_sub, seed=self.global_iteration + idx)
                    
                    if result.is_feasible():
                        return (idx, result.best, mapping, True)
                    else:
                        return (idx, None, mapping, False)
                except Exception as e:
                    print(f"Subproblem {idx} failed: {e}")
                    return (idx, None, mapping, False)
            
            # Prepare arguments for parallel execution
            worker_args = [
                (sub_idx, sub_data, subproblem_mappings[sub_idx]) 
                for sub_idx, sub_data in enumerate(subproblems)
            ]
            
            # Execute in parallel using ThreadPoolExecutor to avoid deadlock
            subproblem_solutions = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
                futures = [executor.submit(solve_subproblem_worker, args) for args in worker_args]
                
                # Add timeout to prevent infinite hanging
                for future in concurrent.futures.as_completed(futures, timeout=300):  # 5 minutes timeout
                    try:
                        sub_idx, solution, mapping, is_feasible = future.result(timeout=60)  # 1 minute per subproblem
                        if is_feasible and solution is not None:
                            subproblem_solutions.append((solution, mapping))
                    except concurrent.futures.TimeoutError:
                        print(f"Subproblem {sub_idx if 'sub_idx' in locals() else 'unknown'} timed out")
                    except Exception as exc:
                        print(f"Subproblem worker exception: {exc}")
            
            # Merge solutions
            if not subproblem_solutions:
                print("No feasible subproblem solutions found.")
                return None
            
            return self._merge_subproblem_solutions(subproblem_solutions)
            
        except Exception as e:
            print(f"Parallel subproblem solving failed: {e}")
            return None
    
    def _merge_subproblem_solutions(self, 
                                  subproblem_solutions: List[Tuple[Solution, Dict]]) -> Optional[Solution]:
        """Merge subproblem solutions back to the original problem."""
        
        try:
            from pyvrp._pyvrp import Route, Trip, Solution
            
            new_routes = []
            
            for solution, mapping in subproblem_solutions:
                # Create mapping from subproblem indices to original indices
                new_to_old_map = {
                    new_idx: old_idx 
                    for old_idx, new_idx in mapping['old_to_new_map'].items()
                }
                
                for route in solution.routes():
                    # Map client visits back to original indices
                    original_visits = [
                        new_to_old_map[client_idx] 
                        for client_idx in route.visits()
                    ]
                    
                    if not original_visits:
                        continue
                    
                    # Map depot indices
                    original_start_depot = new_to_old_map[route.start_depot()]
                    original_end_depot = new_to_old_map[route.end_depot()]
                    
                    # Create trip
                    trip = Trip(
                        self.data,
                        original_visits,
                        0,
                        start_depot=original_start_depot,
                        end_depot=original_end_depot
                    )
                    
                    # Create route
                    original_route = Route(self.data, [trip], 0)
                    new_routes.append(original_route)
            
            if not new_routes:
                print("No valid routes to merge.")
                return None
            
            # Create merged solution
            merged_solution = Solution(self.data, new_routes)
            
            print(f"Merged solution: {merged_solution.num_routes()} routes, "
                  f"distance: {merged_solution.distance() / 10:.1f}, "
                  f"feasible: {merged_solution.is_feasible()}")
            
            return merged_solution
            
        except Exception as e:
            print(f"Solution merging failed: {e}")
            return None
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the synchronization process."""
        
        if not self.sync_history:
            return {"message": "No synchronization data available"}
        
        # Calculate sync statistics
        sync_times = [result.sync_time for result in self.sync_history]
        improvements = [result.improvement_found for result in self.sync_history]
        decompositions = [result.decomposition_triggered for result in self.sync_history]
        
        stats = {
            "total_synchronizations": len(self.sync_history),
            "total_improvements": sum(improvements),
            "total_decompositions": sum(decompositions),
            "average_sync_time": statistics.mean(sync_times) if sync_times else 0,
            "improvement_rate": sum(improvements) / len(improvements) if improvements else 0,
            "decomposition_rate": sum(decompositions) / len(decompositions) if decompositions else 0,
        }
        
        # Add global best information
        if self.global_best_stats:
            stats.update({
                "global_best_vehicles": self.global_best_stats.best_vehicles,
                "global_best_distance": self.global_best_stats.best_distance,
                "global_best_duration": self.global_best_stats.best_duration,
                "global_best_feasible": self.global_best_stats.is_feasible,
                "global_best_strategy": self.global_best_stats.strategy_name,
            })
        
        # Add improvement history
        if self.improvement_history:
            stats["improvement_history"] = self.improvement_history[-10:]  # Last 10 improvements
        
        return stats
    
    def print_sync_summary(self, sync_result: SynchronizationResult) -> None:
        """Print a summary of the latest synchronization."""
        
        print(f"\n{'='*60}")
        print(f"SYNCHRONIZATION SUMMARY - Iteration {self.global_iteration}")
        print(f"{'='*60}")
        
        print(f"Global Best Strategy: {sync_result.global_best_stats.strategy_name} "
              f"(ID: {sync_result.global_best_strategy_id})")
        
        stats = sync_result.global_best_stats
        print(f"Best Solution: {stats.best_vehicles} vehicles, "
              f"distance: {stats.best_distance:.1f}, "
              f"duration: {stats.best_duration:.1f}")
        print(f"Feasible: {stats.is_feasible}")
        
        if sync_result.improvement_found:
            print("*** GLOBAL IMPROVEMENT FOUND! ***")
        
        if sync_result.decomposition_triggered:
            print("*** DECOMPOSITION TRIGGERED ***")
        
        print(f"Synchronization time: {sync_result.sync_time:.3f}s")
        
        # Strategy performance comparison
        print(f"\nStrategy Performance:")
        print("-" * 40)
        for stats in sync_result.strategies_stats:
            status = "BEST" if stats.strategy_id == sync_result.global_best_strategy_id else "    "
            if stats.feasible_population_count > 0:
                print(f"{status} {stats.strategy_name}: "
                      f"{stats.best_vehicles}v, {stats.best_distance:.1f}d, "
                      f"feasible: {stats.is_feasible}, "
                      f"avg_distance: {stats.avg_feasible_cost:.1f} "
                      f"({stats.feasible_population_count} feasible)")
            else:
                print(f"{status} {stats.strategy_name}: "
                      f"{stats.best_vehicles}v, {stats.best_distance:.1f}d, "
                      f"feasible: {stats.is_feasible}, "
                      f"avg_distance: N/A (no feasible solutions)")
        
        print(f"{'='*60}\n")