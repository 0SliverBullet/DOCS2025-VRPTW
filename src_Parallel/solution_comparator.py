"""
Solution comparison utilities for VRPTW problems.
Provides hierarchical comparison based on: vehicles count -> distance -> duration.
"""

from typing import Optional
from pyvrp._pyvrp import Solution
from pyvrp.Result import Result


def is_better_solution(sol1: Optional[Solution], sol2: Optional[Solution]) -> bool:
    """
    Compare two Solution objects and determine if sol1 is better than sol2.
    
    Comparison criteria (hierarchical):
    1. Number of vehicles (fewer is better)
    2. If same vehicle count, total distance (shorter is better)  
    3. If same distance, total duration (shorter is better)
    
    Parameters
    ----------
    sol1 : Solution or None
        First solution to compare
    sol2 : Solution or None  
        Second solution to compare
        
    Returns
    -------
    bool
        True if sol1 is better than sol2, False otherwise
    """
    # Handle None cases
    if sol1 is None and sol2 is None:
        return False
    if sol1 is None:
        return False
    if sol2 is None:
        return True
    
    # Compare vehicle count first
    vehicles1 = sol1.num_routes()
    vehicles2 = sol2.num_routes()
    if vehicles1 != vehicles2:
        return vehicles1 < vehicles2
    
    # Same vehicle count, compare distance
    distance1 = sol1.distance()
    distance2 = sol2.distance()
    if distance1 != distance2:
        return distance1 < distance2
    
    # Same distance, compare duration
    duration1 = sol1.duration()
    duration2 = sol2.duration()
    return duration1 < duration2


def is_better_result(result1: Optional[Result], result2: Optional[Result]) -> bool:
    """
    Compare two Result objects and determine if result1 is better than result2.
    
    Parameters
    ----------
    result1 : Result or None
        First result to compare
    result2 : Result or None
        Second result to compare
        
    Returns
    -------
    bool
        True if result1 is better than result2, False otherwise
    """
    # Handle None cases
    if result1 is None and result2 is None:
        return False
    if result1 is None:
        return False
    if result2 is None:
        return True
    
    # Extract solutions and compare
    sol1 = result1.best if hasattr(result1, 'best') else None
    sol2 = result2.best if hasattr(result2, 'best') else None
    
    return is_better_solution(sol1, sol2)


def get_solution_metrics(solution: Optional[Solution]) -> tuple[int, float, float]:
    """
    Get the metrics tuple for a solution for comparison purposes.
    
    Parameters
    ----------
    solution : Solution or None
        Solution to get metrics from
        
    Returns
    -------
    tuple[int, float, float]
        (num_vehicles, distance, duration) or (inf, inf, inf) for None
    """
    if solution is None:
        return (float('inf'), float('inf'), float('inf'))
    
    return (solution.num_routes(), solution.distance(), solution.duration())


def format_solution_summary(solution: Optional[Solution]) -> str:
    """
    Format a solution into a readable summary string.
    
    Parameters
    ----------
    solution : Solution or None
        Solution to format
        
    Returns
    -------
    str
        Formatted solution summary
    """
    if solution is None:
        return "No solution"
    
    vehicles, distance, duration = get_solution_metrics(solution)
    feasible = solution.is_feasible()
    
    return f"{vehicles}v, {distance/10:.1f}d, {duration/10:.1f}t, feasible: {feasible}"


class SolutionTracker:
    """
    Utility class to track and compare the best solution found.
    Uses hierarchical VRPTW comparison criteria.
    """
    
    def __init__(self):
        self.best_solution: Optional[Solution] = None
        self.best_metrics: tuple = (float('inf'), float('inf'), float('inf'))
    
    def update(self, solution: Optional[Solution]) -> bool:
        """
        Update the tracker with a new solution if it's better.
        
        Parameters
        ----------
        solution : Solution or None
            New solution to consider
            
        Returns
        -------
        bool
            True if the solution was accepted as new best, False otherwise
        """
        if solution is None:
            return False
        
        if is_better_solution(solution, self.best_solution):
            self.best_solution = solution
            self.best_metrics = get_solution_metrics(solution)
            return True
        
        return False
    
    def get_best(self) -> Optional[Solution]:
        """Get the current best solution."""
        return self.best_solution
    
    def get_best_metrics(self) -> tuple[int, float, float]:
        """Get metrics of the current best solution."""
        return self.best_metrics
    
    def has_solution(self) -> bool:
        """Check if a best solution exists."""
        return self.best_solution is not None
    
    def format_best(self) -> str:
        """Format the best solution as a string."""
        return format_solution_summary(self.best_solution)