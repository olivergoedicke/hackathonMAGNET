from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

from typing import Callable
import numpy as np
from itertools import product

from tqdm import tqdm


class GridSearchOptimizer(BaseOptimizer):
    """
    GridSearchOptimizer systematically explores the parameter space by evaluating points on a grid.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 n_phase_points: int = 8,    # Number of points to sample for each phase
                 n_amp_points: int = 5) -> None:  # Number of points to sample for each amplitude
        super().__init__(cost_function)
        self.n_phase_points = n_phase_points
        self.n_amp_points = n_amp_points
        
    def optimize(self, simulation: Simulation):
        # Create grid points
        phase_points = np.linspace(0, 2*np.pi, self.n_phase_points)
        amp_points = np.linspace(0, 1, self.n_amp_points)
        
        # Calculate total number of combinations for one coil
        points_per_coil = self.n_phase_points * self.n_amp_points
        
        # To avoid combinatorial explosion, we'll optimize coils sequentially
        best_phases = np.zeros(8)
        best_amplitudes = np.ones(8)
        best_cost = self.cost_function(simulation(CoilConfig(phase=best_phases, amplitude=best_amplitudes)))
        
        for coil_idx in range(8):
            current_best_cost = -np.inf if self.direction == "maximize" else np.inf
            current_best_phase = 0
            current_best_amp = 1
            
            # Try all combinations for this coil
            iterator = product(phase_points, amp_points)
            pbar = tqdm(iterator, total=points_per_coil,
                       desc=f"Optimizing coil {coil_idx+1}/8")
            
            for phase, amp in pbar:
                # Create configuration with current grid point
                test_phases = best_phases.copy()
                test_amplitudes = best_amplitudes.copy()
                test_phases[coil_idx] = phase
                test_amplitudes[coil_idx] = amp
                
                coil_config = CoilConfig(phase=test_phases, amplitude=test_amplitudes)
                current_cost = self.cost_function(simulation(coil_config))
                
                # Update best if better
                if ((self.direction == "maximize" and current_cost > current_best_cost) or 
                    (self.direction == "minimize" and current_cost < current_best_cost)):
                    current_best_cost = current_cost
                    current_best_phase = phase
                    current_best_amp = amp
                    pbar.set_postfix_str(f"Best cost {current_best_cost:.2f}")
            
            # Update best configuration for this coil
            best_phases[coil_idx] = current_best_phase
            best_amplitudes[coil_idx] = current_best_amp
            best_cost = current_best_cost
            
        return CoilConfig(phase=best_phases, amplitude=best_amplitudes) 