"""
Decomposition-based solution improvement module.
Provides functionality to improve solutions using problem decomposition and parallel subproblem solving.
"""

from __future__ import annotations
import time
import concurrent.futures
from typing import Optional, Tuple

from pyvrp._pyvrp import Solution, ProblemData
from pyvrp.solve import solve as solve_subproblem
from pyvrp.stop import MaxIterations
from pyvrp.search import LocalSearch
from pyvrp._pyvrp import CostEvaluator

try:
    from decomposition import barycenter_clustering_decomposition
    DECOMPOSITION_SUPPORTED = True
except ImportError:
    DECOMPOSITION_SUPPORTED = False
    print("Warning: decomposition module not found. Decomposition improvement disabled.")


def solve_subproblem_worker(sub_data, subproblem_mapping, idx, subproblem_iters, seed):
    """
    Worker function for solving a single subproblem in parallel.
    
    Parameters
    ----------
    sub_data : ProblemData
        Subproblem data
    subproblem_mapping : dict
        Subproblem mapping information
    idx : int
        Subproblem index
    subproblem_iters : int
        Number of iterations for subproblem solving
    seed : int
        Random seed
        
    Returns
    -------
    tuple
        (subproblem_index, solution, mapping_info, is_feasible)
    """
    try:
        print(f"Solving subproblem {idx} with {subproblem_iters} iterations...")
        stop_sub = MaxIterations(subproblem_iters)
        res = solve_subproblem(sub_data, stop=stop_sub, seed=seed)
        
        if res.is_feasible():
            print(f"Subproblem {idx} solved successfully with cost: {res.cost()}")
            return (idx, res.best, subproblem_mapping, True)
        else:
            print(f"Subproblem {idx} returned infeasible solution")
            return (idx, None, subproblem_mapping, False)
    except Exception as e:
        print(f"Error solving subproblem {idx}: {e}")
        return (idx, None, subproblem_mapping, False)


def decomposition_improve_solution(
    solution: Solution,
    data: ProblemData, 
    cost_evaluator: CostEvaluator,
    local_search: LocalSearch,
    num_subproblems: int = 8,
    subproblem_iters: int = 1000,
    random_seed: int = 0
) -> Optional[Solution]:
    """
    Improve a solution using decomposition-based approach.
    
    This function decomposes the given solution into subproblems, solves each 
    subproblem in parallel, and then merges the results back into a complete solution.
    
    Parameters
    ----------
    solution : Solution
        The solution to improve
    data : ProblemData
        Problem data
    cost_evaluator : CostEvaluator
        Cost evaluator for solution evaluation
    local_search : LocalSearch
        Local search method for solution improvement
    num_subproblems : int, default=8
        Number of subproblems to decompose into
    subproblem_iters : int, default=1000
        Number of iterations for solving each subproblem
    random_seed : int, default=0
        Random seed for reproducibility
        
    Returns
    -------
    Solution or None
        Improved solution if successful, None otherwise
    """
    if not DECOMPOSITION_SUPPORTED:
        print("Decomposition not supported - missing decomposition module")
        return None
    
    if not solution or not solution.is_feasible():
        print("Invalid or infeasible input solution for decomposition")
        return None
    
    print(f"Starting decomposition improvement with {num_subproblems} subproblems...")
    start_time = time.time()
    
    try:
        # Step 1: Decompose the solution into subproblems
        subproblems, subproblem_mappings = barycenter_clustering_decomposition(
            solution, 
            data, 
            num_subproblems,
            max_customers_per_cluster=len(data.clients()) // num_subproblems,
            random_state=random_seed
        )
        
        if not subproblems:
            print("Decomposition failed - no subproblems generated")
            return None
        
        print(f"Successfully decomposed into {len(subproblems)} subproblems")
        
        # Step 2: Solve subproblems in parallel
        subproblem_solutions = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_subproblems) as executor:
            # Submit all subproblem solving tasks
            futures = {
                executor.submit(
                    solve_subproblem_worker, 
                    sub_data, 
                    subproblem_mappings[idx], 
                    idx + 1, 
                    subproblem_iters, 
                    random_seed + idx
                ): idx 
                for idx, sub_data in enumerate(subproblems)
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, sol, mapping, is_feasible = future.result()
                    if is_feasible and sol is not None:
                        subproblem_solutions.append((sol, mapping))
                        print(f"Subproblem {idx} completed successfully")
                    else:
                        print(f"Subproblem {idx} failed or returned infeasible solution")
                except Exception as exc:
                    print(f"Subproblem worker generated an exception: {exc}")
        
        if not subproblem_solutions:
            print("No feasible subproblem solutions found")
            return None
        
        print(f"Successfully solved {len(subproblem_solutions)} out of {len(subproblems)} subproblems")
        
        # Step 3: Merge subproblem solutions
        merged_solution = _merge_subproblem_solutions(
            subproblem_solutions, data, cost_evaluator
        )
        
        if merged_solution is None:
            print("Failed to merge subproblem solutions")
            return None
        
        # Step 4: Apply local search to improve merged solution
        improved_solution = local_search(merged_solution, cost_evaluator)
        
        # Verify improvement
        original_cost = cost_evaluator.cost(solution)
        improved_cost = cost_evaluator.cost(improved_solution)
        
        elapsed_time = time.time() - start_time
        print(f"Decomposition improvement completed in {elapsed_time:.2f}s")
        print(f"Original cost: {original_cost}, Improved cost: {improved_cost}")
        
        if improved_cost < original_cost:
            print(f"Decomposition found improvement: {original_cost - improved_cost:.2f}")
            return improved_solution
        else:
            print("Decomposition did not improve the solution")
            return None
            
    except Exception as e:
        print(f"Error during decomposition improvement: {e}")
        return None


def _merge_subproblem_solutions(
    subproblem_solutions: list, 
    data: ProblemData, 
    cost_evaluator: CostEvaluator
) -> Optional[Solution]:
    """
    Merge subproblem solutions back into a complete solution.
    
    Parameters
    ----------
    subproblem_solutions : list
        List of (solution, mapping) tuples from subproblems
    data : ProblemData
        Original problem data
    cost_evaluator : CostEvaluator
        Cost evaluator for solution evaluation
        
    Returns
    -------
    Solution or None
        Merged solution if successful, None otherwise
    """
    try:
        new_routes = []
        
        for sol, mapping in subproblem_solutions:
            # Create mapping from subproblem indices to original problem indices
            new_to_old_map = {
                new_idx: old_idx 
                for old_idx, new_idx in mapping['old_to_new_map'].items()
            }
            
            for route in sol.routes():
                # Map route visits back to original problem indices
                original_visits = [
                    new_to_old_map[client_idx] 
                    for client_idx in route.visits()
                ]
                
                # Reconstruct route using correct PyVRP API
                from pyvrp._pyvrp import Route, Trip
                
                # Map depot indices
                original_start_depot = new_to_old_map[route.start_depot()]
                original_end_depot = new_to_old_map[route.end_depot()]
                
                # Create Trip object
                trip = Trip(
                    data, 
                    original_visits, 
                    0,  # trip index
                    start_depot=original_start_depot, 
                    end_depot=original_end_depot
                )
                
                # Create Route object using first vehicle type of original problem
                original_route = Route(
                    data,
                    [trip],
                    0  # use first vehicle type
                )
                new_routes.append(original_route)
        
        if not new_routes:
            print("Warning: No routes to merge")
            return None
        
        # Create merged solution
        merged_solution = Solution(data, new_routes)
        
        # Verify the merged solution
        cost = cost_evaluator.cost(merged_solution)
        if cost == float('inf') or cost > 1e15:
            print("Warning: Merged solution has invalid cost")
            return None
        
        print(f"Successfully merged {len(new_routes)} routes")
        print(f"Merged solution cost: {cost}")
        print(f"Merged solution is feasible: {merged_solution.is_feasible()}")
        
        return merged_solution
        
    except Exception as e:
        print(f"Error during solution merging: {e}")
        return None


def is_decomposition_supported() -> bool:
    """
    Check if decomposition functionality is available.
    
    Returns
    -------
    bool
        True if decomposition is supported, False otherwise
    """
    return DECOMPOSITION_SUPPORTED