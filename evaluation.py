from main import run

from src.costs import B1HomogeneityCost
from src.data import Simulation
from src.utils import evaluate_coil_config
from src.optimizers.dummy import DummyOptimizer
from src.optimizers.gradient_descent import GradientDescentOptimizer
from src.optimizers.grid_search import GridSearchOptimizer
from src.optimizers.pso import PSOOptimizer

import numpy as np
import json
import os

"""
Available optimizers (configured in main.py):
1. GradientDescentOptimizer - Uses numerical gradients to optimize coil configurations
2. GridSearchOptimizer (commented out) - Systematically explores parameter space on a grid
3. PSOOptimizer (commented out) - Uses particle swarm optimization for global search
"""

if __name__ == "__main__":
    # Load simulation data
    simulation_file = "data/simulations/children_1_tubes_2_id_23848.h5"
    coil_file = "data/antenna/antenna.h5"
    
    # Create simulation object with correct paths
    simulation = Simulation(
        path=simulation_file,
        coil_path=coil_file
    )
    
    # Define cost function
    cost_function = B1HomogeneityCost()
    
    # Run optimization (optimizer selection is in main.py)
    best_coil_config = run(simulation=simulation, cost_function=cost_function)
    
    # Evaluate best coil configuration
    result = evaluate_coil_config(best_coil_config, simulation, cost_function)

    # Save results to JSON file
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
