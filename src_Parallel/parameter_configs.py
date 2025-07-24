"""
Parameter configurations for parallel multi-strategy HGS solver.
Provides diverse parameter combinations to maximize search space exploration.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import itertools


@dataclass
class HGSStrategyConfig:
    """Configuration for a single HGS strategy."""
    
    # Genetic Algorithm Parameters
    repair_probability: float = 0.80
    nb_iter_no_improvement: int = 20_000
    
    # Population Parameters
    min_population_size: int = 25
    generation_size: int = 40
    lb_diversity: float = 0.1
    ub_diversity: float = 0.5
    nb_elite: int = 4
    nb_close: int = 5
    
    # Neighbourhood Parameters
    weight_wait_time: float = 0.2
    weight_time_warp: float = 1.0
    nb_granular: int = 40
    symmetric_proximity: bool = True
    symmetric_neighbours: bool = False
    
    # Penalty Management
    penalty_increase: float = 1.34
    penalty_decrease: float = 0.32
    
    # Strategy identifier
    strategy_name: str = "default"
    
    def __post_init__(self):
        """Validate parameter ranges."""
        if not 0 <= self.repair_probability <= 1:
            raise ValueError("repair_probability must be in [0, 1]")
        if self.nb_iter_no_improvement < 0:
            raise ValueError("nb_iter_no_improvement must be >= 0")
        if self.min_population_size < 1:
            raise ValueError("min_population_size must be >= 1")
        if not 0 <= self.lb_diversity <= 1:
            raise ValueError("lb_diversity must be in [0, 1]")
        if not 0 <= self.ub_diversity <= 1:
            raise ValueError("ub_diversity must be in [0, 1]")
        if self.lb_diversity >= self.ub_diversity:
            raise ValueError("lb_diversity must be < ub_diversity")


def generate_diverse_strategies(num_strategies: int = 8) -> List[HGSStrategyConfig]:
    """
    Generate diverse HGS strategy configurations.
    
    Parameters
    ----------
    num_strategies : int
        Number of strategies to generate (default 8 for 8-core system)
        
    Returns
    -------
    List[HGSStrategyConfig]
        List of diverse strategy configurations
    """
    
    if num_strategies < 1:
        raise ValueError("num_strategies must be >= 1")
    
    # Define parameter variation ranges
    param_ranges = {
        'repair_probability': [0.6, 0.8, 1.0],
        'nb_iter_no_improvement': [15_000, 20_000, 25_000],
        'min_population_size': [20, 30, 45],
        'generation_size': [35, 50, 65],
        'lb_diversity': [0.08, 0.15, 0.20],
        'ub_diversity': [0.45, 0.60, 0.75],
        'nb_granular': [35, 45, 60],
        'penalty_increase': [1.2, 1.34, 1.5]
    }
    
    # Generate all possible combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    all_combinations = list(itertools.product(*param_values))
    
    # Select diverse configurations
    strategies = []
    
    if num_strategies <= len(all_combinations):
        # Evenly distribute selections across all combinations
        step = len(all_combinations) // num_strategies
        selected_indices = [i * step for i in range(num_strategies)]
        
        for i, idx in enumerate(selected_indices):
            combination = all_combinations[idx]
            config_dict = dict(zip(param_names, combination))
            config_dict['strategy_name'] = f"strategy_{i+1}"
            
            strategies.append(HGSStrategyConfig(**config_dict))
    else:
        # If we need more strategies than combinations, use cycling
        for i in range(num_strategies):
            combination = all_combinations[i % len(all_combinations)]
            config_dict = dict(zip(param_names, combination))
            config_dict['strategy_name'] = f"strategy_{i+1}"
            
            # Add some variation for repeated configurations
            if i >= len(all_combinations):
                # Slight variation in penalty parameters
                config_dict['penalty_start'] = 100 + (i % 3) * 50
                config_dict['penalty_decrease'] = 0.99 - (i % 3) * 0.01
            
            strategies.append(HGSStrategyConfig(**config_dict))
    
    return strategies


def get_predefined_strategies() -> List[HGSStrategyConfig]:
    """
    Get 8 predefined diverse strategies optimized for different problem characteristics.
    
    Returns
    -------
    List[HGSStrategyConfig]
        List of 8 predefined diverse strategies
    """
    
    strategies = [
        # Strategy 1: Conservative - High diversity, stable parameters
        HGSStrategyConfig(
            repair_probability=0.8,
            nb_iter_no_improvement=20_000,
            min_population_size=25,
            generation_size=40,
            lb_diversity=0.1,
            ub_diversity=0.5,
            nb_close=4,
            nb_elite=5,
            weight_wait_time=0.2,
            weight_time_warp=1.0,
            nb_granular=40,
            strategy_name="original"
        ),
        
        # Strategy 2: Aggressive - Fast convergence, high penalties
        HGSStrategyConfig(
            repair_probability=0.9,
            nb_iter_no_improvement=20_000,
            min_population_size=15,
            generation_size=20,
            lb_diversity=0.05,
            ub_diversity=0.4,
            nb_close=4,
            nb_elite=3,
            weight_wait_time=0.2,
            weight_time_warp=1.0,
            nb_granular=30,
            strategy_name="aggressive"
        ),
        
        # Strategy 3: Large Population - Maximum diversity focus
        HGSStrategyConfig(
            repair_probability=0.8,
            nb_iter_no_improvement=20_000,
            min_population_size=35,
            generation_size=60,
            lb_diversity=0.15,
            ub_diversity=0.65,
            nb_close=8,
            nb_elite=6,
            nb_granular=40,
            strategy_name="large_population"
        ),
        
        # Strategy 4: Small Population - Intensive search
        HGSStrategyConfig(
            repair_probability=0.8,
            nb_iter_no_improvement=20_000,
            min_population_size=15,
            generation_size=20,
            lb_diversity=0.1,
            ub_diversity=0.4,
            nb_close=5,
            nb_elite=4,
            nb_granular=40,
            strategy_name="small_population"
        ),
        
        # Strategy 5: Granular Focus - High granular neighbourhood
        HGSStrategyConfig(
            nb_close=7,
            nb_elite=4,
            nb_granular=65,
            strategy_name="granular_focus"
        ),
        
        # Strategy 6: Patient - Long convergence with balanced parameters
        HGSStrategyConfig(
            repair_probability=0.9,
            nb_iter_no_improvement=30_000,
            min_population_size=40,
            generation_size=70,
            lb_diversity=0.2,
            ub_diversity=0.7,
            nb_close=9,
            nb_elite=6,
            nb_granular=50,
            strategy_name="patient"
        ),
        
        # Strategy 7: Balanced - Well-balanced across all dimensions
        HGSStrategyConfig(
            repair_probability=0.8,
            nb_iter_no_improvement=20_000,
            min_population_size=10,
            generation_size=20,
            lb_diversity=0.15,
            nb_close=3,
            nb_elite=2,
            nb_granular=25,
            strategy_name="balanced"
        ),
        
        # Strategy 8: Dynamic - Variable penalty with high weights
        HGSStrategyConfig(
            repair_probability=0.75,
            nb_iter_no_improvement=22_000,
            min_population_size=28,
            generation_size=42,
            lb_diversity=0.10,
            ub_diversity=0.50,
            nb_close=6,
            nb_elite=4,
            weight_wait_time=0.28,
            weight_time_warp=1.15,
            nb_granular=52,
            penalty_increase=1.48,
            penalty_decrease=0.95,
            strategy_name="dynamic"
        )
    ]
    
    return strategies


def get_strategy_by_name(strategy_name: str) -> HGSStrategyConfig:
    """
    Get a specific strategy configuration by name.
    
    Parameters
    ----------
    strategy_name : str
        Name of the strategy to retrieve
        
    Returns
    -------
    HGSStrategyConfig
        Strategy configuration
        
    Raises
    ------
    ValueError
        If strategy name is not found
    """
    strategies = get_predefined_strategies()
    
    for strategy in strategies:
        if strategy.strategy_name == strategy_name:
            return strategy
    
    raise ValueError(f"Strategy '{strategy_name}' not found")


def print_strategy_summary(strategies: List[HGSStrategyConfig]) -> None:
    """Print a summary of strategy configurations."""
    
    print(f"Strategy Configuration Summary ({len(strategies)} strategies):")
    print("=" * 80)
    
    for i, strategy in enumerate(strategies):
        print(f"Strategy {i+1}: {strategy.strategy_name}")
        print(f"  Repair Probability: {strategy.repair_probability}")
        print(f"  No Improvement Iters: {strategy.nb_iter_no_improvement}")
        print(f"  Population Size: {strategy.min_population_size}")
        print(f"  Generation Size: {strategy.generation_size}")
        print(f"  Diversity Range: [{strategy.lb_diversity}, {strategy.ub_diversity}]")
        print(f"  Granular Neighbours: {strategy.nb_granular}")
        print(f"  Weight Time Warp: {strategy.weight_time_warp}")
        print(f"  Penalty Increase: {strategy.penalty_increase}")
        print("-" * 40)


if __name__ == "__main__":
    # Test strategy generation
    print("Testing parameter configuration generation...")
    
    # Test predefined strategies
    predefined = get_predefined_strategies()
    print_strategy_summary(predefined)
    
    # Test diverse generation
    print("\nTesting diverse strategy generation...")
    diverse = generate_diverse_strategies(8)
    print_strategy_summary(diverse)