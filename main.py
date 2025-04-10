from src.costs.base import BaseCost
from src.optimizers.dummy import DummyOptimizer
from src.optimizers.gradient_descent import GradientDescentOptimizer
from src.optimizers.grid_search import GridSearchOptimizer
from src.optimizers.pso import PSOOptimizer
from src.optimizers.adam_torch import AdamTorchOptimizer
from src.optimizers.test import ConjugateGradientWongOptimizer
from src.data import Simulation, CoilConfig

import numpy as np

def run(simulation: Simulation, 
        cost_function: BaseCost,
        timeout: int = 100) -> CoilConfig:
    """
        Main function to run the optimization, returns the best coil configuration

        Args:
            simulation: Simulation object
            cost_function: Cost function object
            timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    # Choose one optimizer by uncommenting it:
    
    # Option 1: Gradient Descent Optimizer
    # optimizer = GradientDescentOptimizer(cost_function=cost_function, max_iter=1, learning_rate=0.5, epsilon=1e-2)
    
    # Option 2: Grid Search Optimizer (commented out)
    # optimizer = GridSearchOptimizer(cost_function=cost_function, n_phase_points=8, n_amp_points=5)
    
    # Option 3: Particle Swarm Optimizer (commented out)
    # optimizer = PSOOptimizer(cost_function=cost_function, n_particles=50, max_iter=50)
    
    # Option 4: Adam Optimizer (commented out)
    optimizer = AdamTorchOptimizer(cost_function=cost_function, max_iter=15, learning_rate=0.3,
                                   timeout=290, num_inits=75)

    # Option 5: test optimizer (commented out)
    # optimizer = ConjugateGradientWongOptimizer(cost_function=cost_function)


    best_coil_config = optimizer.optimize(simulation)
    return best_coil_config