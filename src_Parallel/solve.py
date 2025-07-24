from __future__ import annotations

from typing import TYPE_CHECKING

import tomli

import pyvrp.search
from GeneticAlgorithm import GeneticAlgorithm, GeneticAlgorithmParams
from pyvrp.PenaltyManager import PenaltyManager, PenaltyParams
from pyvrp.Population import Population
from pyvrp._pyvrp import PopulationParams
from pyvrp._pyvrp import ProblemData, RandomNumberGenerator, Solution
from pyvrp.crossover import ordered_crossover as ox
from pyvrp.crossover import selective_route_exchange as srex
from pyvrp.diversity import broken_pairs_distance as bpd
from typing import Tuple, Optional
from pyvrp.search import (
    NODE_OPERATORS,
    ROUTE_OPERATORS,
    LocalSearch,
    NeighbourhoodParams,
    NodeOperator,
    RouteOperator,
    compute_neighbours,
)

# <<< 新增并行求解相关导入 >>>
from parameter_configs import HGSStrategyConfig

if TYPE_CHECKING:
    import pathlib

    from pyvrp.Result import Result
    from pyvrp.stop import StoppingCriterion


class SolveParams:
    """
    Solver parameters for PyVRP's hybrid genetic search algorithm.

    Parameters
    ----------
    genetic
        Genetic algorithm parameters.
    penalty
        Penalty parameters.
    population
        Population parameters.
    neighbourhood
        Neighbourhood parameters.
    node_ops
        Node operators to use in the search.
    route_ops
        Route operators to use in the search.
    """

    def __init__(
        self,
        genetic: GeneticAlgorithmParams = GeneticAlgorithmParams(),
        penalty: PenaltyParams = PenaltyParams(),
        population: PopulationParams = PopulationParams(),
        neighbourhood: NeighbourhoodParams = NeighbourhoodParams(),
        node_ops: list[type[NodeOperator]] = NODE_OPERATORS,
        route_ops: list[type[RouteOperator]] = ROUTE_OPERATORS,
    ):
        self._genetic = genetic
        self._penalty = penalty
        self._population = population
        self._neighbourhood = neighbourhood
        self._node_ops = node_ops
        self._route_ops = route_ops

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SolveParams)
            and self.genetic == other.genetic
            and self.penalty == other.penalty
            and self.population == other.population
            and self.neighbourhood == other.neighbourhood
            and self.node_ops == other.node_ops
            and self.route_ops == other.route_ops
        )

    @property
    def genetic(self):
        return self._genetic

    @property
    def penalty(self):
        return self._penalty

    @property
    def population(self):
        return self._population

    @property
    def neighbourhood(self):
        return self._neighbourhood

    @property
    def node_ops(self):
        return self._node_ops

    @property
    def route_ops(self):
        return self._route_ops

    @classmethod
    def from_file(cls, loc: str | pathlib.Path):
        """
        Loads the solver parameters from a TOML file.
        """
        with open(loc, "rb") as fh:
            data = tomli.load(fh)

        gen_params = GeneticAlgorithmParams(**data.get("genetic", {}))
        pen_params = PenaltyParams(**data.get("penalty", {}))
        pop_params = PopulationParams(**data.get("population", {}))
        nb_params = NeighbourhoodParams(**data.get("neighbourhood", {}))

        node_ops = NODE_OPERATORS
        if "node_ops" in data:
            node_ops = [getattr(pyvrp.search, op) for op in data["node_ops"]]

        route_ops = ROUTE_OPERATORS
        if "route_ops" in data:
            route_ops = [getattr(pyvrp.search, op) for op in data["route_ops"]]

        return cls(
            gen_params, pen_params, pop_params, nb_params, node_ops, route_ops
        )


def solve(
    data: ProblemData,
    stop: StoppingCriterion,
    seed: int = 0,
    collect_stats: bool = True,
    display: bool = False,
    params: SolveParams = SolveParams(),
) -> Result:
    """
    Solves the given problem data instance.

    Parameters
    ----------
    data
        Problem data instance to solve.
    stop
        Stopping criterion to use.
    seed
        Seed value to use for the random number stream. Default 0.
    collect_stats
        Whether to collect statistics about the solver's progress. Default
        ``True``.
    display
        Whether to display information about the solver progress. Default
        ``False``. Progress information is only available when
        ``collect_stats`` is also set, which it is by default.
    params
        Solver parameters to use. If not provided, a default will be used.

    Returns
    -------
    Result
        A Result object, containing statistics (if collected) and the best
        found solution.
    """
    rng = RandomNumberGenerator(seed=seed)
    neighbours = compute_neighbours(data, params.neighbourhood)
    ls = LocalSearch(data, rng, neighbours)

    for node_op in params.node_ops:
        ls.add_node_operator(node_op(data))

    for route_op in params.route_ops:
        ls.add_route_operator(route_op(data))

    pm = PenaltyManager.init_from(data, params.penalty)
    pop = Population(bpd, params.population)
    init = [
        Solution.make_random(data, rng)
        for _ in range(params.population.min_pop_size)
    ]

    # We use SREX when the instance is a proper VRP; else OX for TSP.
    crossover = srex if data.num_vehicles > 1 else ox

    gen_args = (data, pm, rng, pop, ls, crossover, init, params.genetic)
    algo = GeneticAlgorithm(*gen_args)  # type: ignore
    return algo.run(stop, collect_stats, display)


class PersistentSolver:
    """
    Persistent solver that maintains population state across multiple solve calls.
    This enables continuous evolution without re-initializing the population.
    """
    
    def __init__(self, 
                 data: ProblemData,
                 params: SolveParams = SolveParams(),
                 seed: int = 0):
        """
        Initialize the persistent solver with all necessary components.
        
        Parameters
        ----------
        data : ProblemData
            Problem data instance
        params : SolveParams
            Solver parameters  
        seed : int
            Random seed
        """
        self.data = data
        self.params = params
        self.seed = seed
        
        # Initialize all persistent components
        self.rng = RandomNumberGenerator(seed=seed)
        self.neighbours = compute_neighbours(data, params.neighbourhood)
        self.ls = LocalSearch(data, self.rng, self.neighbours)
        
        # Add operators to local search
        for node_op in params.node_ops:
            self.ls.add_node_operator(node_op(data))
        
        for route_op in params.route_ops:
            self.ls.add_route_operator(route_op(data))
        
        # Initialize penalty manager and population
        self.pm = PenaltyManager.init_from(data, params.penalty)
        self.pop = Population(bpd, params.population)
        
        # Create cost evaluator for population operations 
        # Use penalty manager to create proper cost evaluator
        from pyvrp._pyvrp import CostEvaluator
        # Get load penalties from penalty manager (simplified approach)
        load_penalties = [1.0] * data.num_vehicles  # Simple uniform penalties
        tw_penalty = 1.0  # Time window penalty
        dist_penalty = 1.0  # Distance penalty
        self.cost_evaluator = CostEvaluator(load_penalties, tw_penalty, dist_penalty)
        
        # Generate initial population  
        init = [
            Solution.make_random(data, self.rng)
            for _ in range(params.population.min_pop_size)
        ]
        
        # Choose crossover based on problem type
        crossover = srex if data.num_vehicles > 1 else ox
        
        # Create genetic algorithm with existing population
        gen_args = (data, self.pm, self.rng, self.pop, self.ls, crossover, init, params.genetic)
        self.algo = GeneticAlgorithm(*gen_args)
        
        # Track state
        self.total_iterations = 0
        self.best_solution = None
        self.is_initialized = True
    
    def solve_batch(self, 
                   stop: StoppingCriterion,
                   collect_stats: bool = True,
                   display: bool = False) -> Result:
        """
        Run a batch of iterations on the existing population.
        
        Parameters
        ----------
        stop : StoppingCriterion
            Stopping criterion for this batch
        collect_stats : bool
            Whether to collect statistics
        display : bool
            Whether to display progress
            
        Returns
        -------
        Result
            Result from this batch of iterations
        """
        if not self.is_initialized:
            raise RuntimeError("Solver not initialized")
        
        # Run the genetic algorithm for this batch
        result = self.algo.run(stop, collect_stats, display)
        
        # Update tracking
        self.total_iterations += getattr(result, 'iterations', 0)
        if result.best and (self.best_solution is None or 
                           self._is_better_solution(result.best, self.best_solution)):
            self.best_solution = result.best
        
        return result
    
    def inject_solution(self, solution: Solution):
        """
        Inject an elite solution into the population.
        
        Parameters
        ----------
        solution : Solution
            Solution to inject into the population
        """
        if self.is_initialized and solution.is_feasible():
            self.pop.add(solution, self.cost_evaluator)
            
            # Update best if this is better
            if (self.best_solution is None or 
                self._is_better_solution(solution, self.best_solution)):
                self.best_solution = solution
    
    def get_current_best(self) -> Optional[Solution]:
        """
        Get the current best solution from the population.
        
        Returns
        -------
        Solution
            Current best solution, or None if no feasible solution found
        """
        if not self.is_initialized:
            return None
            
        if self.best_solution:
            return self.best_solution
        elif self.is_initialized:
            # Use tournament selection to get best solution from population
            try:
                return self.pop.tournament(self.rng, self.cost_evaluator, k=1)
            except:
                return None
        else:
            return None
    
    def get_population_stats(self) -> dict:
        """
        Get statistics about the current population.
        
        Returns
        -------
        dict
            Population statistics
        """
        if not self.is_initialized:
            return {}
            
        return {
            'population_size': len(self.pop),
            'total_iterations': self.total_iterations,
            'best_cost': self._get_solution_cost(self.best_solution) if self.best_solution else float('inf')
        }
    
    def _is_better_solution(self, sol1: Solution, sol2: Solution) -> bool:
        """Compare two solutions to determine which is better."""
        if not sol2:
            return True
        if not sol1:
            return False
            
        # Both solutions exist, compare by feasibility first, then cost
        sol1_feasible = sol1.is_feasible()
        sol2_feasible = sol2.is_feasible()
        
        if sol1_feasible != sol2_feasible:
            return sol1_feasible  # Feasible is better than infeasible
            
        # Both have same feasibility, compare by cost
        return self._get_solution_cost(sol1) < self._get_solution_cost(sol2)
    
    def _get_solution_cost(self, solution: Solution) -> float:
        """Get cost of a solution."""
        if not solution:
            return float('inf')
        return solution.distance() + solution.duration()


# <<< 新增并行求解入口函数 >>>
def solve_parallel(
    data: ProblemData,
    stop: StoppingCriterion,
    seed: int = 0,
    collect_stats: bool = True,
    display: bool = True,
    params = None,
) -> Tuple[Result, 'SolutionSynchronizer']:
    """
    Solve a VRPTW problem using parallel multi-strategy HGS.
    
    Parameters
    ----------
    data : ProblemData
        Problem data instance to solve
    stop : StoppingCriterion
        Stopping criterion to use
    seed : int
        Seed value to use for the random number stream. Default 0.
    collect_stats : bool
        Whether to collect statistics about the solver's progress. Default True.
    display : bool
        Whether to display information about the solver progress. Default True.
    params : ParallelSolveParams, optional
        Parallel solver parameters to use. If not provided, a default will be used.
        
    Returns
    -------
    Result
        A Result object, containing statistics (if collected) and the best
        found solution across all parallel strategies.
    """
    from parallel_hgs_solver import ParallelMultiStrategyHGS, ParallelSolveParams
    from solution_synchronizer import SolutionSynchronizer
    
    if params is None:
        params = ParallelSolveParams()
    
    solver = ParallelMultiStrategyHGS(data, params)
    result, synchronizer = solver.solve(stop, seed, collect_stats, display)
    
    # Return both result and synchronizer
    return result, synchronizer


def create_solve_params_from_strategy_config(config: HGSStrategyConfig) -> SolveParams:
    """
    Create SolveParams from HGSStrategyConfig.
    
    Parameters
    ----------
    config : HGSStrategyConfig
        Strategy configuration to convert
        
    Returns
    -------
    SolveParams
        Corresponding SolveParams object
    """
    genetic_params = GeneticAlgorithmParams(
        repair_probability=config.repair_probability,
        nb_iter_no_improvement=config.nb_iter_no_improvement
    )
    
    population_params = PopulationParams(
        min_pop_size=config.min_population_size,
        generation_size=config.generation_size,
        lb_diversity=config.lb_diversity,
        ub_diversity=config.ub_diversity,
        nb_elite=config.nb_elite,
        nb_close=config.nb_close
    )
    
    neighbourhood_params = NeighbourhoodParams(
        weight_wait_time=config.weight_wait_time,
        weight_time_warp=config.weight_time_warp,
        nb_granular=config.nb_granular,
        symmetric_proximity=config.symmetric_proximity,
        symmetric_neighbours=config.symmetric_neighbours
    )
    
    penalty_params = PenaltyParams(
        penalty_increase=config.penalty_increase,
        penalty_decrease=config.penalty_decrease
    )
    
    return SolveParams(
        genetic=genetic_params,
        population=population_params,
        neighbourhood=neighbourhood_params,
        penalty=penalty_params
    )
