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

# <<< 新增导入 >>>
# 假设 decomposition.py 与 GeneticAlgorithm.py 在同一目录下或在Python路径中
from pyvrp.solve import solve as solve_subproblem
from pyvrp.stop import MaxIterations
try:
    from decomposition import barycenter_clustering_decomposition
    DECOMPOSITION_SUPPORTED = True
except ImportError:
    DECOMPOSITION_SUPPORTED = False
# <<< 结束新增导入 >>>

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
    # <<< 新增参数 >>>
    decomposition_frequency: int = 1_000  # 每 5000 次迭代执行一次分解
    num_subproblems: int = 1             # 分解成 1 个子问题
    subproblem_iters: int = 3_000         # 求解子问题时的迭代次数

    def __post_init__(self):
        if not 0 <= self.repair_probability <= 1:
            raise ValueError("repair_probability must be in [0, 1].")

        if self.nb_iter_no_improvement < 0:
            raise ValueError("nb_iter_no_improvement < 0 not understood.")
    
        # <<< 新增检查 >>>
        if self.decomposition_frequency < 0:
            raise ValueError("decomposition_frequency < 0 not understood.")

        if self.num_subproblems < 0:
            raise ValueError("num_subproblems < 0 not understood.")

        if self.subproblem_iters < 0:
            raise ValueError("subproblem_iters < 0 not understood.")


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

            if iters_no_improvement == self._params.nb_iter_no_improvement:
                print_progress.restart()

                iters_no_improvement = 1
                self._pop.clear()

                for sol in self._initial_solutions:
                    self._pop.add(sol, self._cost_evaluator)
            
            # --- START: DECOMPOSITION LOGIC (关键修改部分) ---
            # 检查是否支持分解并且达到了触发频率
            if DECOMPOSITION_SUPPORTED and self._params.num_subproblems >= 1 and iters % self._params.decomposition_frequency == 0:
                print_progress.restart()  # 可选：在日志中标记分解阶段

                print(f"Decomposition triggered at iteration {iters}.")

                # 1. 选择精英解 (论文中建议从前10%的解中随机选) [cite: 631]
                #    为简单起见，我们直接用当前最优解。
                
                elite_solution = self._best

                # 2. 调用您的分解函数
                # subproblems = barycenter_clustering_decomposition(
                #     elite_solution, 
                #     self._data, 
                #     self._params.num_subproblems,
                #     random_state=iters
                # )
                subproblems = [self._data]  # 简化为不分解，直接使用原数据
                
                if subproblems:
                    # 3. 求解子问题
                    subproblem_solutions = []
                    # 这里我们简化为串行求解，也可以修改为并行
                    for sub_data in subproblems:
                        # 论文中建议用固定的迭代次数来求解子问题 [cite: 633]
                        stop_sub = MaxIterations(self._params.subproblem_iters)
                        res = solve_subproblem(sub_data, stop=stop_sub, seed=iters)

                        if res.is_feasible():
                            subproblem_solutions.append(res.best)

                    # 4. 合并与整合
                    if subproblem_solutions:
                        new_routes = [
                            route
                            for sol in subproblem_solutions
                            for route in sol.routes()
                        ]
                        
                        print(f"Number of routes to merge: {len(new_routes)}")
                        
                        # 检查路由是否为空
                        if not new_routes:
                            print("Warning: No routes to merge, skipping decomposition step.")
                            continue
                        
                        try:
                            merged_solution = Solution(self._data, new_routes)
                            initial_cost = self._cost_evaluator.cost(merged_solution)
                            
                            print(f"Merged solution cost (before search): {initial_cost}")
                            print(f"Merged solution is feasible: {merged_solution.is_feasible()}")
                            
                            # 检查合并解决方案是否有效
                            if initial_cost == float('inf') or initial_cost > 1e15:
                                print("Warning: Merged solution has invalid cost, skipping search.")
                                continue
                            
                            # 只有当解决方案有效时才进行搜索
                            if merged_solution.is_feasible() or initial_cost < float('inf'):
                                # 使用 _improve_offspring 函数进行改进，并标记为来自分解
                                curr_best_cost = self._cost_evaluator.cost(self._best)
                                self._improve_offspring(merged_solution, from_decomposition=True)
                                new_best_cost = self._cost_evaluator.cost(self._best)
                                
                                # 如果找到了更好的解，重置无改进迭代计数器
                                if new_best_cost < curr_best_cost:
                                    iters_no_improvement = 1
                                    
                                print(f"Merged solution cost (after improvement): {new_best_cost}")
                            else:
                                print("Warning: Initial merged solution is infeasible, skipping improvement.")
                                
                        except Exception as e:
                            print(f"Error during solution merging: {e}")
                            continue

            # --- END: DECOMPOSITION LOGIC ---

            curr_best = self._cost_evaluator.cost(self._best)

            parents = self._pop.select(self._rng, self._cost_evaluator)
            offspring = self._crossover(
                parents, self._data, self._cost_evaluator, self._rng
            )
            self._improve_offspring(offspring)

            new_best = self._cost_evaluator.cost(self._best)

            if new_best < curr_best:
                iters_no_improvement = 1
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
                if from_decomposition:
                    print("Decomposition found a new best solution!")
