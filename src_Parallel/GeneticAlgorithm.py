from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Collection

from pyvrp.ProgressPrinter import ProgressPrinter
from pyvrp.Result import Result
from pyvrp.Statistics import Statistics

# 将Solution移到运行时导入，因为在247行需要使用它
from pyvrp._pyvrp import Solution

if TYPE_CHECKING:
    from pyvrp.PenaltyManager import PenaltyManager
    from pyvrp.Population import Population
    from pyvrp._pyvrp import (
        CostEvaluator,
        ProblemData,
        RandomNumberGenerator,
    )
    from pyvrp.search.SearchMethod import SearchMethod
    from pyvrp.stop.StoppingCriterion import StoppingCriterion


@dataclass
class GeneticAlgorithmParams:
    """
    Parameters for the genetic algorithm.

    Parameters
    ----------
    repair_probability
        Probability (in :math:`[0, 1]`) of repairing an infeasible solution.
        If the reparation makes the solution feasible, it is also added to
        the population in the same iteration.
    nb_iter_no_improvement
        Number of iterations without any improvement needed before a restart
        occurs.

    Attributes
    ----------
    repair_probability
        Probability of repairing an infeasible solution.
    nb_iter_no_improvement
        Number of iterations without improvement before a restart occurs.

    Raises
    ------
    ValueError
        When ``repair_probability`` is not in :math:`[0, 1]`, or
        ``nb_iter_no_improvement`` is negative.
    """

    repair_probability: float = 0.80
    nb_iter_no_improvement: int = 20_000

    def __post_init__(self):
        if not 0 <= self.repair_probability <= 1:
            raise ValueError("repair_probability must be in [0, 1].")

        if self.nb_iter_no_improvement < 0:
            raise ValueError("nb_iter_no_improvement < 0 not understood.")


class GeneticAlgorithm:
    """
    Creates a GeneticAlgorithm instance.

    Parameters
    ----------
    data
        Data object describing the problem to be solved.
    penalty_manager
        Penalty manager to use.
    rng
        Random number generator.
    population
        Population to use.
    search_method
        Search method to use.
    crossover_op
        Crossover operator to use for generating offspring.
    initial_solutions
        Initial solutions to use to initialise the population.
    params
        Genetic algorithm parameters. If not provided, a default will be used.

    Raises
    ------
    ValueError
        When the population is empty.
    """

    def __init__(
        self,
        data: ProblemData,
        penalty_manager: PenaltyManager,
        rng: RandomNumberGenerator,
        population: Population,
        search_method: SearchMethod,
        crossover_op: Callable[
            [
                tuple[Solution, Solution],
                ProblemData,
                CostEvaluator,
                RandomNumberGenerator,
            ],
            Solution,
        ],
        initial_solutions: Collection[Solution],
        params: GeneticAlgorithmParams = GeneticAlgorithmParams(),
    ):
        if len(initial_solutions) == 0:
            raise ValueError("Expected at least one initial solution.")

        self._data = data
        self._pm = penalty_manager
        self._rng = rng
        self._pop = population
        self._search = search_method
        self._crossover = crossover_op
        self._initial_solutions = initial_solutions
        self._params = params

        # Find best feasible initial solution if any exist, else set a random
        # infeasible solution (with infinite cost) as the initial best.
        self._best = min(initial_solutions, key=self._cost_evaluator.cost)
        
        # <<< 新增并行求解支持 >>>
        # 外部解注入队列
        self._injected_solutions: list[Solution] = []
        # 当前迭代计数器
        self._current_iteration = 0
        # 上次改进迭代
        self._last_improvement_iteration = 0

    @property
    def _cost_evaluator(self) -> CostEvaluator:
        return self._pm.cost_evaluator()

    def run(
        self,
        stop: StoppingCriterion,
        collect_stats: bool = True,
        display: bool = False,
    ):
        """
        Runs the genetic algorithm with the provided stopping criterion.

        Parameters
        ----------
        stop
            Stopping criterion to use. The algorithm runs until the first time
            the stopping criterion returns ``True``.
        collect_stats
            Whether to collect statistics about the solver's progress. Default
            ``True``.
        display
            Whether to display information about the solver progress. Default
            ``False``. Progress information is only available when
            ``collect_stats`` is also set.

        Returns
        -------
        Result
            A Result object, containing statistics (if collected) and the best
            found solution.
        """
        print_progress = ProgressPrinter(should_print=display)
        print_progress.start(self._data)

        start = time.perf_counter()
        stats = Statistics(collect_stats=collect_stats)
        iters = 0
        iters_no_improvement = 1

        for sol in self._initial_solutions:
            self._pop.add(sol, self._cost_evaluator)

        while not stop(self._cost_evaluator.cost(self._best)):
            iters += 1
            self._current_iteration = iters

            if iters_no_improvement == self._params.nb_iter_no_improvement:
                print_progress.restart()

                iters_no_improvement = 1
                self._pop.clear()

                for sol in self._initial_solutions:
                    self._pop.add(sol, self._cost_evaluator)
            
            # <<< 新增：处理注入的解决方案 >>>
            if self._injected_solutions:
                for injected_sol in self._injected_solutions:
                    self._pop.add(injected_sol, self._cost_evaluator)
                    self._pm.register(injected_sol)
                    
                    # 检查是否为新的最佳解
                    if self._cost_evaluator.cost(injected_sol) < self._cost_evaluator.cost(self._best):
                        self._best = injected_sol
                        self._last_improvement_iteration = iters
                        print(f"Injected solution became new best at iteration {iters}!")
                
                # 清空注入队列
                self._injected_solutions.clear()
            

            curr_best = self._cost_evaluator.cost(self._best)

            parents = self._pop.select(self._rng, self._cost_evaluator)
            offspring = self._crossover(
                parents, self._data, self._cost_evaluator, self._rng
            )
            self._improve_offspring(offspring)

            new_best = self._cost_evaluator.cost(self._best)

            if new_best < curr_best:
                iters_no_improvement = 1
                self._last_improvement_iteration = iters
            else:
                iters_no_improvement += 1

            stats.collect_from(self._pop, self._cost_evaluator)
            print_progress.iteration(stats)

        end = time.perf_counter() - start
        res = Result(self._best, stats, iters, end)

        print_progress.end(res)

        return res

    def _improve_offspring(self, sol: Solution, from_decomposition: bool = False):
        def is_new_best(sol):
            cost = self._cost_evaluator.cost(sol)
            best_cost = self._cost_evaluator.cost(self._best)
            return cost < best_cost

        sol = self._search(sol, self._cost_evaluator)
        self._pop.add(sol, self._cost_evaluator)
        self._pm.register(sol)

        if is_new_best(sol):
            self._best = sol
            if from_decomposition:
                print("Decomposition found a new best solution!")

        # Possibly repair if current solution is infeasible. In that case, we
        # penalise infeasibility more using a penalty booster.
        if (
            not sol.is_feasible()
            and self._rng.rand() < self._params.repair_probability
        ):
            sol = self._search(sol, self._pm.booster_cost_evaluator())

            if sol.is_feasible():
                self._pop.add(sol, self._cost_evaluator)
                self._pm.register(sol)

            if is_new_best(sol):
                self._best = sol
                self._last_improvement_iteration = self._current_iteration
                if from_decomposition:
                    print("Decomposition found a new best solution!")
    
    # <<< 新增：外部解注入接口 >>>
    def inject_elite_solution(self, solution: Solution):
        """
        向当前遗传算法实例注入一个精英解决方案。
        
        Parameters
        ----------
        solution : Solution
            要注入的精英解决方案
        """
        if solution is not None:
            self._injected_solutions.append(solution)
    
    def get_current_stats(self) -> dict[str, any]:
        """
        获取当前遗传算法的状态统计信息。
        
        Returns
        -------
        dict[str, any]
            包含当前统计信息的字典
        """
        return {
            'current_iteration': self._current_iteration,
            'last_improvement_iteration': self._last_improvement_iteration,
            'current_best_cost': self._cost_evaluator.cost(self._best),
            'current_best_feasible': self._best.is_feasible() if self._best else False,
            'population_size': len(self._pop) if hasattr(self._pop, '__len__') else 0,
        }
    
    def get_best_solution(self) -> Solution:
        """
        获取当前最佳解决方案。
        
        Returns
        -------
        Solution
            当前最佳解决方案
        """
        return self._best
