"""
Parallel multi-strategy HGS solver for VRPTW.
Manages multiple HGS instances with different parameters running in parallel.
"""

from __future__ import annotations
import time
import threading
import queue
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import copy

from pyvrp._pyvrp import Solution, ProblemData, RandomNumberGenerator
from pyvrp.Result import Result
from pyvrp.stop import StoppingCriterion, MaxRuntime
from solve import SolveParams, solve, PersistentSolver
from parameter_configs import HGSStrategyConfig, get_predefined_strategies
from solution_synchronizer import SolutionSynchronizer
from decomposition_improver import decomposition_improve_solution, is_decomposition_supported
from solution_comparator import is_better_solution, SolutionTracker, get_solution_metrics


@dataclass
class ParallelSolveParams:
    """Parameters for parallel multi-strategy solving."""
    
    num_strategies: int = 8
    sync_frequency: int = 1500
    decomposition_frequency: int = 4000  
    num_subproblems: int = 8
    subproblem_iters: int = 2000
    strategy_configs: Optional[List[HGSStrategyConfig]] = None
    collect_stats: bool = True
    display: bool = True
    
    def __post_init__(self):
        if self.num_strategies < 1:
            raise ValueError("num_strategies must be >= 1")
        if self.sync_frequency < 1:
            raise ValueError("sync_frequency must be >= 1")
        if self.decomposition_frequency < 1:
            raise ValueError("decomposition_frequency must be >= 1")


class StrategyWorker:
    """Worker class for managing a single HGS strategy in parallel execution."""
    
    def __init__(self, 
                 strategy_id: int,
                 strategy_config: HGSStrategyConfig,
                 data: ProblemData,
                 seed: int):
        self.strategy_id = strategy_id
        self.strategy_config = strategy_config
        self.data = data
        self.seed = seed
        
        # State management
        self.is_running = False
        self.is_paused = False
        self.current_iteration = 0
        self.iterations_no_improvement = 1
        self.total_time = 0.0
        self.start_time = 0.0
        self.last_improvement_iteration = 0
        
        # Solution tracking
        self.current_best: Optional[Solution] = None
        self.solution_tracker = SolutionTracker()
        
        # Thread synchronization
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop_event = threading.Event()
        self.result_queue = queue.Queue()
        self.inject_queue = queue.Queue()
        
        # Persistent solver for continuous evolution
        self.persistent_solver: Optional[PersistentSolver] = None
        
    def run(self, stop_criterion: StoppingCriterion) -> Result:
        """Run the strategy worker using persistent population evolution."""
        self.is_running = True
        self.start_time = time.time()
        
        try:
            # Initialize persistent solver on first run
            if self.persistent_solver is None:
                self._initialize_persistent_solver()
            
            # Run continuous evolution with persistent population
            return self._run_persistent_evolution(stop_criterion)
        except Exception as e:
            print(f"Strategy {self.strategy_id} failed: {e}")
            from pyvrp.Statistics import Statistics
            dummy_solution = Solution.make_random(self.data, RandomNumberGenerator(self.seed))
            return Result(dummy_solution, Statistics(), 0, time.time() - self.start_time)
        finally:
            self.is_running = False
    
    def _initialize_persistent_solver(self):
        """Initialize the persistent solver with strategy-specific parameters."""
        try:
            solve_params = self._create_solve_params()
            self.persistent_solver = PersistentSolver(
                data=self.data,
                params=solve_params,
                seed=self.seed
            )
            print(f"Strategy {self.strategy_id}: Initialized persistent solver with population")
        except Exception as e:
            print(f"Strategy {self.strategy_id}: Failed to initialize persistent solver: {e}")
            self.persistent_solver = None
    
    def _run_persistent_evolution(self, stop_criterion: StoppingCriterion) -> Result:
        """Run continuous evolution using persistent population."""
        if self.persistent_solver is None:
            # Fallback to old method if persistent solver failed
            return self._run_enhanced_batch_fallback(stop_criterion)
        
        from pyvrp.Statistics import Statistics
        from pyvrp.stop import MaxRuntime
        
        best_result = None
        batch_count = 0
        max_batches = 1000
        batch_runtime = 5.0  # Small 2-second batches
        
        while not self.stop_event.is_set() and batch_count < max_batches:
            # Check stopping criterion
            if batch_count > 0 and self.current_best:
                # Use proper hierarchical comparison for stopping criterion
                # For now, use simple cost as stop_criterion expects a numeric value
                vehicles, distance, duration = get_solution_metrics(self.current_best)
                combined_cost = distance + duration  # Temporary for stop_criterion compatibility
                if stop_criterion(combined_cost):
                    break
            
            # Wait if paused
            self.pause_event.wait()
            if self.stop_event.is_set():
                break
            
            # Check for solution injection
            while not self.inject_queue.empty():
                try:
                    injected_solution = self.inject_queue.get_nowait()
                    if injected_solution.is_feasible():
                        # Inject into persistent solver
                        self.persistent_solver.inject_solution(injected_solution)
                        
                        # Update local tracking
                        if self.solution_tracker.update(injected_solution):
                            self.current_best = injected_solution
                except Exception as e:
                    print(f"Strategy {self.strategy_id}: Solution injection failed: {e}")
            
            # Run one batch on persistent population
            try:
                batch_stop = MaxRuntime(int(batch_runtime))
                result = self.persistent_solver.solve_batch(
                    stop=batch_stop,
                    collect_stats=True,
                    display=False
                )
                
                # Update best solution from persistent solver
                current_best = self.persistent_solver.get_current_best()
                if current_best and current_best.is_feasible():
                    if self.solution_tracker.update(current_best):
                        self.current_best = current_best
                        self.last_improvement_iteration = batch_count
                        best_result = result
                
                self.current_iteration += getattr(result, 'iterations', 1)
                batch_count += 1
                
            except Exception as e:
                print(f"Strategy {self.strategy_id} batch {batch_count} failed: {e}")
                break
        
        self.total_time = time.time() - self.start_time
        
        # Return best result found or create one from current state
        if best_result:
            return best_result
        elif self.current_best:
            # Ensure we have a valid solution
            solution = self.current_best
            if solution is None:
                solution = Solution.make_random(self.data, RandomNumberGenerator(self.seed))
            
            return Result(
                solution,
                Statistics(),
                self.current_iteration,
                self.total_time
            )
        else:
            # Create dummy result
            return Result(
                Solution.make_random(self.data, RandomNumberGenerator(self.seed)),
                Statistics(),
                self.current_iteration,
                self.total_time
            )
    
    def _run_enhanced_batch_fallback(self, stop_criterion: StoppingCriterion) -> Result:
        """Run enhanced batch approach with frequent small batches."""
        from pyvrp.Statistics import Statistics
        
        best_result = None
        batch_count = 0
        max_batches = 100
        batch_runtime = 2.0  # Small 2-second batches
        
        while not self.stop_event.is_set() and batch_count < max_batches:
            # Check stopping criterion
            if batch_count > 0 and self.current_best:  # Skip check on first iteration
                vehicles, distance, duration = get_solution_metrics(self.current_best)
                combined_cost = distance + duration  # Temporary for stop_criterion compatibility
                if stop_criterion(combined_cost):
                    break
            
            # Wait if paused
            self.pause_event.wait()
            if self.stop_event.is_set():
                break
            
            # Check for solution injection
            while not self.inject_queue.empty():
                try:
                    injected_solution = self.inject_queue.get_nowait()
                    if injected_solution.is_feasible():
                        if self.solution_tracker.update(injected_solution):
                            self.current_best = injected_solution
                except:
                    pass
            
            # Run one small batch
            try:
                batch_stop = MaxRuntime(int(batch_runtime))
                solve_params = self._create_solve_params()
                
                result = solve(
                    self.data,
                    stop=batch_stop,
                    seed=self.seed + batch_count * 100,  # Vary seed slightly
                    collect_stats=True,
                    display=False,
                    params=solve_params
                )
                
                # Update best solution
                if result.best and result.best.is_feasible():
                    if self.solution_tracker.update(result.best):
                        self.current_best = result.best
                        self.last_improvement_iteration = batch_count
                        best_result = result
                
                self.current_iteration += getattr(result, 'iterations', 1)  # Default to 1 if no iterations attribute
                batch_count += 1
                
            except Exception as e:
                print(f"Strategy {self.strategy_id} batch {batch_count} failed: {e}")
                break
        
        self.total_time = time.time() - self.start_time
        
        # Return best result found or create dummy
        if best_result:
            return best_result
        else:
            # Ensure we have a valid solution
            solution = self.current_best
            if solution is None:
                solution = Solution.make_random(self.data, RandomNumberGenerator(self.seed))
            
            return Result(
                solution,
                Statistics(), 
                self.current_iteration, 
                self.total_time
            )
    
    def _create_solve_params(self) -> SolveParams:
        """Create SolveParams from strategy configuration."""
        from GeneticAlgorithm import GeneticAlgorithmParams
        
        genetic_params = GeneticAlgorithmParams(
            repair_probability=self.strategy_config.repair_probability,
            nb_iter_no_improvement=self.strategy_config.nb_iter_no_improvement
        )
        
        return SolveParams(genetic=genetic_params)
    
    def pause(self):
        """Pause the strategy execution."""
        self.is_paused = True
        self.pause_event.clear()
    
    def resume(self):
        """Resume the strategy execution."""
        self.is_paused = False
        self.pause_event.set()
    
    def stop(self):
        """Stop the strategy execution."""
        self.stop_event.set()
        self.pause_event.set()
    
    def inject_solution(self, solution: Solution):
        """Inject an elite solution into this strategy."""
        try:
            self.inject_queue.put_nowait(solution)
        except queue.Full:
            pass
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics for this strategy."""
        avg_population_cost = 0.0
        avg_feasible_cost = 0.0
        feasible_population_count = 0
        
        # Try to get average population cost from persistent solver
        if (self.persistent_solver and 
            hasattr(self.persistent_solver, 'pm') and 
            hasattr(self.persistent_solver, 'pop')):
            try:
                # Get cost evaluator from penalty manager
                cost_evaluator = self.persistent_solver.pm.cost_evaluator()
                
                # Get all solutions from population
                population_costs = []
                feasible_costs = []
                # Access population directly from persistent solver
                pop = self.persistent_solver.pop
                
                # Try to get population size and iterate through solutions
                if hasattr(pop, '__len__') and len(pop) > 0:
                    # Try different methods to access population solutions
                    if hasattr(pop, 'solutions'):
                        solutions = pop.solutions()
                    elif hasattr(pop, '__iter__'):
                        solutions = list(pop)
                    else:
                        solutions = []
                    
                    # Calculate costs separately for feasible and infeasible solutions
                    for solution in solutions:
                        if solution is not None:
                            cost = cost_evaluator.cost(solution)
                            population_costs.append(cost)
                            
                            # Only consider feasible solutions for avg_feasible_cost
                            if solution.is_feasible():
                                # For feasible solutions, use only distance as simplified cost
                                feasible_cost = solution.distance() / 100
                                feasible_costs.append(feasible_cost)
                    
                    # Calculate averages
                    if population_costs:
                        avg_population_cost = sum(population_costs) / len(population_costs)
                    
                    if feasible_costs:
                        avg_feasible_cost = sum(feasible_costs) / len(feasible_costs)
                        feasible_population_count = len(feasible_costs)
                        
            except Exception as e:
                # If population cost calculation fails, use best solution cost as fallback
                if self.current_best and self.current_best.is_feasible():
                    try:
                        avg_feasible_cost = self.current_best.distance() / 100
                        feasible_population_count = 1
                    except:
                        avg_feasible_cost = 0.0
                        feasible_population_count = 0
        
        return {
            'strategy_name': self.strategy_config.strategy_name,
            'current_iteration': self.current_iteration,
            'iterations_no_improvement': self.iterations_no_improvement,
            'last_improvement_iteration': self.last_improvement_iteration,
            'total_time': time.time() - self.start_time if self.is_running else self.total_time,
            'avg_population_cost': avg_population_cost,
            'avg_feasible_cost': avg_feasible_cost,
            'feasible_population_count': feasible_population_count,
        }


class ParallelMultiStrategyHGS:
    """Parallel multi-strategy HGS solver."""
    
    def __init__(self, 
                 data: ProblemData,
                 params: ParallelSolveParams = ParallelSolveParams()):
        self.data = data
        self.params = params
        
        # Get strategy configurations
        if params.strategy_configs is None:
            self.strategy_configs = get_predefined_strategies()[:params.num_strategies]
        else:
            self.strategy_configs = params.strategy_configs[:params.num_strategies]
        
        # Ensure we have enough strategies
        while len(self.strategy_configs) < params.num_strategies:
            idx = len(self.strategy_configs) % len(get_predefined_strategies())
            config = copy.deepcopy(get_predefined_strategies()[idx])
            config.strategy_name = f"{config.strategy_name}_dup_{len(self.strategy_configs)}"
            self.strategy_configs.append(config)
        
        # Initialize synchronizer
        self.synchronizer = SolutionSynchronizer(
            data=data,
            sync_frequency=params.sync_frequency,
            decomposition_frequency=params.decomposition_frequency,
            num_subproblems=params.num_subproblems,
            subproblem_iters=params.subproblem_iters
        )
        
        # State tracking
        self.strategy_workers: List[StrategyWorker] = []
        self.global_best_solution: Optional[Solution] = None
        
        # Decomposition control
        self.global_best_tracker = SolutionTracker()
        self.syncs_without_improvement = 0
        self.decomposition_threshold = 5  # Trigger decomposition after 10 syncs without improvement
        
    def solve(self, 
              stop: StoppingCriterion,
              seed: int = 0,
              collect_stats: bool = True,
              display: bool = True) -> Tuple[Result, SolutionSynchronizer]:
        """Solve the problem using parallel multi-strategy approach."""
        print(f"Starting parallel multi-strategy HGS with {self.params.num_strategies} strategies...")
        
        if display:
            self._print_strategy_summary()
        
        self.start_time = time.time()
        
        try:
            # Create strategy workers
            self.strategy_workers = [
                StrategyWorker(
                    strategy_id=i,
                    strategy_config=config,
                    data=self.data,
                    seed=seed + i * 1000
                )
                for i, config in enumerate(self.strategy_configs)
            ]
            
            # Run parallel strategies with synchronization
            best_result = self._run_parallel_with_sync(stop, display)
            
            total_time = time.time() - self.start_time
            
            if display:
                self._print_final_summary(best_result, total_time)
            
            return best_result, self.synchronizer
            
        except Exception as e:
            print(f"Parallel solving failed: {e}")
            raise
    
    def _run_parallel_with_sync(self, stop: StoppingCriterion, display: bool) -> Result:
        """Run parallel strategies with periodic synchronization."""
        best_result = None
        sync_count = 0
        max_sync_count = 1000
        
        # Start all strategy workers in separate threads
        strategy_threads = []
        for worker in self.strategy_workers:
            thread = threading.Thread(
                target=self._run_continuous_strategy, 
                args=(worker, stop), 
                daemon=True
            )
            thread.start()
            strategy_threads.append(thread)
            time.sleep(0.1)  # Stagger starts
        
        try:
            # Main synchronization loop
            while not self._check_stopping_criterion(stop) and sync_count < max_sync_count:
                sync_count += 1
                current_time = time.time() - self.start_time
                
                if display:
                    print(f"Running synchronization {sync_count}, total time: {current_time:.2f}s")
                
                # Wait for sync interval
                time.sleep(10.0)  # Sync every 2 seconds
                
                # Perform synchronization
                sync_result = self._perform_synchronization(sync_count)
                
                if sync_result and display:
                    self.synchronizer.print_sync_summary(sync_result)
                
                # Update global best and handle decomposition counter
                if sync_result and sync_result.improvement_found and sync_result.global_best_solution:
                    # Note: _should_apply_decomposition already handled global best tracking
                    # Just update the reference for compatibility
                    self.global_best_solution = sync_result.global_best_solution
                    
                    # Create result from best strategy
                    for worker in self.strategy_workers:
                        if worker.current_best is sync_result.global_best_solution:
                            from pyvrp.Statistics import Statistics
                            # Ensure we have a valid solution
                            solution = worker.current_best
                            if solution is not None:
                                best_result = Result(
                                    solution,
                                    Statistics(),
                                    worker.current_iteration,
                                    worker.total_time
                                )
                                break
                    
                    # Inject best solution to all other strategies
                    self._inject_solution_to_all(sync_result.global_best_solution)
        
        finally:
            # Stop all strategy workers
            for worker in self.strategy_workers:
                worker.stop()
            
            # Wait for threads to finish
            for thread in strategy_threads:
                thread.join(timeout=5.0)
        
        final_result = best_result or self._get_best_result_from_workers()
        if final_result is None:
            # Create dummy result if no result found
            from pyvrp.Statistics import Statistics
            dummy_solution = Solution.make_random(self.data, RandomNumberGenerator(0))
            final_result = Result(dummy_solution, Statistics(), 0, 0.0)
        return final_result
    
    def _run_continuous_strategy(self, worker: StrategyWorker, stop: StoppingCriterion):
        """Run a single strategy worker continuously."""
        try:
            long_stop = MaxRuntime(3600)  # 1 hour max per worker
            worker.run(long_stop)
        except Exception as e:
            print(f"Strategy {worker.strategy_id} thread failed: {e}")
    
    def _perform_synchronization(self, sync_count: int):
        """Perform synchronization across all active strategy workers."""
        try:
            print(f"Starting synchronization {sync_count} - pausing all strategies...")
            
            # Step 1: Pause all strategy workers
            for worker in self.strategy_workers:
                worker.pause()
            
            # Brief wait to ensure all workers have paused
            time.sleep(0.25)
            
            # Step 2: Collect current solutions and stats from all workers
            strategies_data = []
            for worker in self.strategy_workers:
                if worker.current_best and worker.is_running:
                    strategies_data.append((
                        worker.strategy_id,
                        worker.current_best,
                        worker.get_current_stats()
                    ))
            
            if not strategies_data:
                print("No strategy data available for synchronization")
                # Resume all workers before returning
                for worker in self.strategy_workers:
                    worker.resume()
                return None
            
            # Step 3: Update synchronizer iteration count and perform basic synchronization
            self.synchronizer.global_iteration = sync_count * self.params.sync_frequency
            sync_result = self.synchronizer.synchronize_solutions(strategies_data)
            
            # Step 4: Check if we should apply decomposition improvement
            should_decompose = self._should_apply_decomposition(sync_result)
            
            if should_decompose and sync_result and sync_result.global_best_solution and is_decomposition_supported():
                print(f"Applying decomposition improvement to best solution...")
                print(f"(Triggered after {self.syncs_without_improvement} synchronizations without improvement)")
                
                improved_solution, decomposition_succeeded = self._apply_decomposition_improvement_with_status(
                    sync_result.global_best_solution, sync_count
                )

                # Reset counter only if decomposition clustering succeeded (regardless of final improvement)
                # This prevents frequent decomposition calls when clustering works but doesn't improve
                if decomposition_succeeded:
                    self.syncs_without_improvement = 0
                    print(f"Decomposition clustering succeeded - counter reset to prevent frequent calls")
                else:
                    print("Decomposition clustering failed - counter not reset, will retry sooner")

                if improved_solution:
                    print(f"Improved solution: {improved_solution.num_routes()}v {improved_solution.distance()/100:.2f}d {improved_solution.duration()/100:.2f}t")
                    # Update the sync result with improved solution
                    sync_result.global_best_solution = improved_solution
                    sync_result.improvement_found = True
                    print(f"Decomposition improvement successful!")
                else:
                    print("Decomposition attempted but no improvement found")
            
            # Step 5: Resume all strategy workers
            print(f"Resuming all strategies after synchronization {sync_count}")
            for worker in self.strategy_workers:
                worker.resume()
            
            return sync_result
            
        except Exception as e:
            print(f"Synchronization {sync_count} failed: {e}")
            # Ensure all workers are resumed even if synchronization fails
            for worker in self.strategy_workers:
                worker.resume()
            return None
    
    def _should_apply_decomposition(self, sync_result) -> bool:
        """
        Determine if decomposition improvement should be applied.
        Only apply after 10 synchronizations without global improvement.
        Uses proper hierarchical VRPTW solution comparison.
        """
        if not sync_result or not sync_result.global_best_solution:
            return False
        
        # Check if we have a new global best using hierarchical comparison
        current_solution = sync_result.global_best_solution
        
        if self.global_best_tracker.update(current_solution):
            # Found improvement - reset counter
            self.syncs_without_improvement = 0
            vehicles, distance, duration = get_solution_metrics(current_solution)
            print(f"New global best found! Vehicles: {vehicles}, Distance: {distance/100:.2f}, Duration: {duration/100:.2f}, counter reset.")
            return False  # Don't decompose when we just found improvement
        else:
            # No improvement - increment counter
            self.syncs_without_improvement += 1
            print(f"No global improvement. Counter: {self.syncs_without_improvement}/{self.decomposition_threshold}")
            
            # Apply decomposition if we've reached the threshold
            return self.syncs_without_improvement >= self.decomposition_threshold
    
    def _apply_decomposition_improvement(self, solution: Solution, sync_count: int) -> Optional[Solution]:
        """Apply decomposition improvement using the best performing strategy's components."""
        improved_solution, _ = self._apply_decomposition_improvement_with_status(solution, sync_count)
        return improved_solution
    
    def _apply_decomposition_improvement_with_status(self, solution: Solution, sync_count: int) -> tuple[Optional[Solution], bool]:
        """
        Apply decomposition improvement and return both result and success status.
        
        Returns
        -------
        tuple[Optional[Solution], bool]
            (improved_solution, decomposition_succeeded)
            decomposition_succeeded is True if clustering worked (regardless of final improvement)
        """
        try:
            # Find the best performing strategy worker
            best_worker = self._find_best_strategy_worker()
            if best_worker is None:
                print("No best worker found for decomposition improvement")
                return None, False
            
            print(f"Using {best_worker.strategy_config.strategy_name} strategy for decomposition improvement")
            
            # Apply safe decomposition without problematic LocalSearch
            improved_solution, clustering_succeeded = self._safe_decomposition_improve_with_status(
                solution=solution,
                best_worker=best_worker,
                sync_count=sync_count
            )
            
            return improved_solution, clustering_succeeded
            
        except Exception as e:
            print(f"Decomposition improvement failed: {e}")
            return None, False
    
    def _find_best_strategy_worker(self):
        """Find the strategy worker with the best current solution."""
        best_worker = None
        
        for worker in self.strategy_workers:
            if worker.current_best and worker.current_best.is_feasible():
                if best_worker is None:
                    best_worker = worker
                elif is_better_solution(worker.current_best, best_worker.current_best):
                    best_worker = worker
        
        return best_worker
    
    def _safe_decomposition_improve(self, solution: Solution, best_worker, sync_count: int) -> Optional[Solution]:
        """Apply decomposition improvement without risky LocalSearch operations."""
        improved_solution, _ = self._safe_decomposition_improve_with_status(solution, best_worker, sync_count)
        return improved_solution
    
    def _safe_decomposition_improve_with_status(self, solution: Solution, best_worker, sync_count: int) -> tuple[Optional[Solution], bool]:
        """
        Apply decomposition improvement and return both result and clustering success status.
        
        Returns
        -------
        tuple[Optional[Solution], bool]
            (improved_solution, clustering_succeeded)
            clustering_succeeded is True if the clustering step worked
        """
        try:
            from decomposition_improver import is_decomposition_supported
            
            if not is_decomposition_supported():
                return None, False
            
            print(f"Starting safe decomposition improvement with {self.params.num_subproblems} subproblems...")
            
            # Apply decomposition but WITHOUT LocalSearch, and get clustering status
            improved_solution, clustering_succeeded = self._decomposition_without_local_search_with_status(
                solution=solution,
                data=self.data,
                num_subproblems=self.params.num_subproblems,
                subproblem_iters=self.params.subproblem_iters,
                random_seed=sync_count * 1000
            )
            
            # If decomposition worked and best worker has persistent solver, use it to improve
            if improved_solution and improved_solution.is_feasible() and best_worker.persistent_solver:
                print("Using best worker's GeneticAlgorithm _improve_offspring for improvement")
                try:
                    # Use the best worker's genetic algorithm to improve the decomposed solution
                    final_solution = self._improve_using_best_worker_ga(
                        improved_solution, best_worker, solution
                    )
                    
                    if final_solution and is_better_solution(final_solution, improved_solution):
                        print("Best worker's _improve_offspring successfully improved decomposed solution")
                        return final_solution, clustering_succeeded
                    else:
                        print("_improve_offspring did not improve solution, using decomposed result")
                        return (improved_solution if is_better_solution(improved_solution, solution) else None), clustering_succeeded
                        
                except Exception as e:
                    print(f"_improve_offspring failed, using decomposed solution: {e}")
                    # Return decomposed solution if improvement fails
                    return (improved_solution if is_better_solution(improved_solution, solution) else None), clustering_succeeded
                
            # Return decomposed solution if no persistent solver available
            return (improved_solution if improved_solution and is_better_solution(improved_solution, solution) else None), clustering_succeeded
            
        except Exception as e:
            print(f"Safe decomposition improvement failed: {e}")
            return None, False
    
    def _decomposition_without_local_search(self, solution: Solution, data, num_subproblems: int, subproblem_iters: int, random_seed: int) -> Optional[Solution]:
        """Apply decomposition improvement without the problematic LocalSearch step."""
        improved_solution, _ = self._decomposition_without_local_search_with_status(solution, data, num_subproblems, subproblem_iters, random_seed)
        return improved_solution
    
    def _decomposition_without_local_search_with_status(self, solution: Solution, data, num_subproblems: int, subproblem_iters: int, random_seed: int) -> tuple[Optional[Solution], bool]:
        """
        Apply decomposition improvement and return both result and clustering success status.
        
        Returns
        -------
        tuple[Optional[Solution], bool]
            (improved_solution, clustering_succeeded)
            clustering_succeeded is True if clustering worked
        """
        try:
            from decomposition import barycenter_clustering_decomposition
            from decomposition_improver import solve_subproblem_worker, _merge_subproblem_solutions
            import concurrent.futures
            import time
            
            if not solution or not solution.is_feasible():
                return None, False
            
            start_time = time.time()

            print(f"Before decomposition: {solution.distance()/100:.2f}")

            # Step 1: Decompose the solution into subproblems with aggressive timeout and retry
            print("Attempting clustering decomposition with process-level timeout...")
            subproblems, subproblem_mappings = self._safe_clustering_decomposition(
                solution, 
                data, 
                num_subproblems,
                random_seed
            )
            
            if not subproblems:
                print("Clustering decomposition failed, abandoning this decomposition improvement")
                return None, False  # Clustering failed
            
            print(f"Successfully decomposed into {len(subproblems)} subproblems")
            
            # Step 2: Solve subproblems in parallel
            subproblem_solutions = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_subproblems) as executor:
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
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        idx, sol, mapping, is_feasible = future.result()
                        if is_feasible and sol is not None:
                            subproblem_solutions.append((sol, mapping))
                    except Exception as exc:
                        print(f"Subproblem worker generated an exception: {exc}")
            
            if not subproblem_solutions:
                return None, True  # Clustering succeeded, but subproblem solving failed
            
            print(f"Successfully solved {len(subproblem_solutions)} out of {len(subproblems)} subproblems")
            
            # Step 3: Merge subproblem solutions (WITHOUT LocalSearch)
            from pyvrp.PenaltyManager import PenaltyManager
            from solve import SolveParams
            params = SolveParams()
            pm = PenaltyManager.init_from(data, params.penalty)
            cost_evaluator = pm.cost_evaluator()
            
            merged_solution = _merge_subproblem_solutions(
                subproblem_solutions, data, cost_evaluator
            )
            
            if merged_solution is None:
                return None, True  # Clustering succeeded, but merging failed
            
            elapsed_time = time.time() - start_time
            original_cost = cost_evaluator.cost(solution)
            merged_cost = cost_evaluator.cost(merged_solution)
            
            print(f"Safe decomposition completed in {elapsed_time:.2f}s")
            print(f"Original cost: {original_cost}, Merged cost: {merged_cost}")
            
            # Return merged solution without risky LocalSearch
            if merged_solution.is_feasible() and merged_cost < original_cost:
                print("Decomposition found improvement without LocalSearch!")
                return merged_solution, True  # Clustering succeeded and found improvement
            else:
                print("Decomposition did not improve the solution")
                return None, True  # Clustering succeeded, but no improvement
            
        except Exception as e:
            print(f"Decomposition without LocalSearch failed: {e}")
            return None, False  # Complete failure
    
    def _improve_using_best_worker_ga(self, decomposed_solution: Solution, best_worker, original_solution: Solution) -> Optional[Solution]:
        """Use the best worker's GeneticAlgorithm _improve_offspring method to improve decomposed solution."""
        try:
            # Get the genetic algorithm instance from the persistent solver
            genetic_algo = best_worker.persistent_solver.algo
            
            print(f"Calling _improve_offspring on decomposed solution from {best_worker.strategy_config.strategy_name} strategy")
            
            decomposed_solution_cost = decomposed_solution.distance()/100
            
            # Call _improve_offspring with the decomposed solution
            genetic_algo._improve_offspring(decomposed_solution, from_decomposition=True)

            improved_solution_cost = decomposed_solution.distance()/100
            
            # Get the potentially improved solution
            improved_solution = genetic_algo._best
            
            # Restore the original best if the new one isn't better than our original
            if decomposed_solution_cost > improved_solution_cost:
                print(f"Decomposed solution is better than improved solution! {decomposed_solution_cost:.2f} > {improved_solution_cost:.2f}")
                return decomposed_solution
            else:
                print(f"Improved solution is better than decomposed solution! {improved_solution_cost:.2f} > {decomposed_solution_cost:.2f}")
                return improved_solution
        
        except Exception as e:
            print(f"Error using _improve_offspring: {e}")
            return decomposed_solution
    
    def _safe_clustering_decomposition(self, solution: Solution, data, num_subproblems: int, base_random_seed: int, timeout_seconds: float = 10.0, max_retries: int = 2):
        """
        Safely perform clustering decomposition with timeout and retry mechanisms.
        
        Parameters
        ----------
        solution : Solution
            Solution to decompose
        data : ProblemData
            Problem data
        num_subproblems : int
            Number of subproblems to create
        base_random_seed : int
            Base random seed for clustering
        timeout_seconds : float
            Timeout for each clustering attempt in seconds
        max_retries : int
            Maximum number of retry attempts
            
        Returns
        -------
        tuple or (None, None)
            (subproblems, mappings) if successful, (None, None) if failed
        """
        from decomposition import barycenter_clustering_decomposition
        import concurrent.futures
        import time
        
        for attempt in range(max_retries):
            try:
                print(f"Clustering attempt {attempt + 1}/{max_retries} (timeout: {timeout_seconds}s)...")
                
                start_time = time.time()
                
                # Try clustering with different random seeds on retry
                current_seed = base_random_seed + attempt * 1000
                max_customers_per_cluster = len(data.clients()) // num_subproblems
                
                # Add some variation to clustering parameters on retry
                if attempt > 0:
                    # Slightly adjust parameters to avoid same failure
                    max_customers_per_cluster = max(1, max_customers_per_cluster + attempt - 1)
                    print(f"Retry {attempt}: adjusting max_customers_per_cluster to {max_customers_per_cluster}")
                
                # Use ProcessPoolExecutor for true process-level timeout that can kill stuck processes
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._clustering_worker,
                        solution, 
                        data, 
                        num_subproblems,
                        max_customers_per_cluster,
                        current_seed
                    )
                    
                    try:
                        subproblems, subproblem_mappings = future.result(timeout=timeout_seconds)
                        
                        elapsed_time = time.time() - start_time
                        print(f"Clustering successful in {elapsed_time:.2f}s, generated {len(subproblems) if subproblems else 0} subproblems")
                        
                        if subproblems and len(subproblems) > 0:
                            return subproblems, subproblem_mappings
                        else:
                            print(f"Clustering attempt {attempt + 1} returned empty subproblems")
                            continue
                            
                    except concurrent.futures.TimeoutError:
                        elapsed_time = time.time() - start_time
                        print(f"Clustering attempt {attempt + 1} timed out after {elapsed_time:.2f}s")
                        
                        # Force cancel the future and terminate the process
                        print("Forcefully terminating stuck clustering process...")
                        future.cancel()
                        
                        # The ProcessPoolExecutor will automatically clean up the stuck process
                        
                        if attempt < max_retries - 1:
                            print(f"Retrying with different parameters...")
                            time.sleep(2)  # Longer pause to ensure process cleanup
                        else:
                            print("All clustering attempts timed out, abandoning decomposition")
                            
            except Exception as e:
                elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
                print(f"Clustering attempt {attempt + 1} failed after {elapsed_time:.2f}s with error: {e}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying with different parameters...")
                    time.sleep(1)  # Brief pause before retry
                else:
                    print("All clustering attempts failed, abandoning decomposition")
        
        print(f"Failed to decompose solution after {max_retries} attempts")
        return None, None
    
    @staticmethod
    def _clustering_worker(solution, data, num_subproblems, max_customers_per_cluster, random_seed):
        """
        Static worker function for clustering decomposition in separate process.
        This runs in a separate process and can be forcefully terminated.
        """
        try:
            from decomposition import barycenter_clustering_decomposition
            print(f"Worker process: Starting clustering with seed {random_seed}")
            
            result = barycenter_clustering_decomposition(
                solution, 
                data, 
                num_subproblems,
                max_customers_per_cluster=max_customers_per_cluster,
                random_state=random_seed
            )
            
            print(f"Worker process: Clustering completed successfully")
            return result
            
        except Exception as e:
            print(f"Worker process: Clustering failed with error: {e}")
            raise
    
    def _inject_solution_to_all(self, solution: Solution):
        """Inject elite solution to all strategy workers."""
        for worker in self.strategy_workers:
            worker.inject_solution(solution)
    
    def _get_best_result_from_workers(self) -> Optional[Result]:
        """Get the best result from all workers."""
        best_result = None
        best_cost = float('inf')
        
        for worker in self.strategy_workers:
            if worker.current_best and worker.best_cost < best_cost:
                best_cost = worker.best_cost
                from pyvrp.Statistics import Statistics
                # Ensure we have a valid solution
                solution = worker.current_best
                if solution is not None:
                    best_result = Result(
                        solution,
                        Statistics(),
                        worker.current_iteration,
                        worker.total_time
                    )
        
        return best_result
    
    def _print_strategy_summary(self):
        """Print summary of strategy configurations."""
        print(f"\n{'='*80}")
        print(f"PARALLEL MULTI-STRATEGY HGS CONFIGURATION")
        print(f"{'='*80}")
        print(f"Number of strategies: {self.params.num_strategies}")
        print(f"Synchronization frequency: {self.params.sync_frequency} iterations")
        print(f"Decomposition frequency: {self.params.decomposition_frequency} iterations")
        print(f"Subproblems for decomposition: {self.params.num_subproblems}")
        print(f"\nStrategy Details:")
        print("-" * 80)
        
        for i, config in enumerate(self.strategy_configs):
            print(f"Strategy {i+1}: {config.strategy_name}")
            print(f"  Repair Prob: {config.repair_probability}, "
                  f"No Improve: {config.nb_iter_no_improvement}, "
                  f"Pop Size: {config.min_population_size}")
        
        print(f"{'='*80}\n")
    
    def _print_final_summary(self, best_result: Result, total_time: float):
        """Print final summary of parallel execution."""
        print(f"\n{'='*80}")
        print(f"PARALLEL EXECUTION COMPLETED")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.2f} seconds")
        
        if best_result and best_result.best:
            solution = best_result.best
            print(f"Best solution found:")
            print(f"  Vehicles: {solution.num_routes()}")
            print(f"  Distance: {solution.distance() / 100:.2f}")
            print(f"  Duration: {solution.duration() / 100:.2f}")
            print(f"  Feasible: {solution.is_feasible()}")
        else:
            print("No feasible solution found.")
        
        # Print synchronization statistics
        sync_stats = self.synchronizer.get_statistics_summary()
        print(f"\nSynchronization Statistics:")
        print(f"  Total synchronizations: {sync_stats.get('total_synchronizations', 0)}")
        print(f"  Improvements found: {sync_stats.get('total_improvements', 0)}")
        print(f"  Decompositions triggered: {sync_stats.get('total_decompositions', 0)}")
        print(f"  Average sync time: {sync_stats.get('average_sync_time', 0):.3f}s")
        
        # Print decomposition statistics
        print(f"\\nDecomposition Statistics:")
        print(f"  Syncs without improvement at end: {self.syncs_without_improvement}")
        print(f"  Decomposition threshold: {self.decomposition_threshold}")
        
        print(f"{'='*80}\n")
    
    def _check_stopping_criterion(self, stop: StoppingCriterion) -> bool:
        """Check if stopping criterion is met."""
        try:
            if self.global_best_solution:
                cost = self.global_best_solution.distance() + self.global_best_solution.duration()
                return stop(cost)
            else:
                return stop(float('inf'))
        except Exception:
            return False