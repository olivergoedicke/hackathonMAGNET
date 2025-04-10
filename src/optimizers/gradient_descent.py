from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

from typing import Callable, Optional, Tuple
import numpy as np
import time

from tqdm import trange


class GradientDescentOptimizer(BaseOptimizer):
    """
    GradientDescentOptimizer uses numerical gradients to optimize coil configurations.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 100,
                 learning_rate: float = 0.01,
                 epsilon: float = 1e-6,
                 momentum: float = 0.9,
                 timeout: float = 300,  # 300 seconds = 5 minutes
                 num_inits: int = 5) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # For numerical gradient computation
        self.momentum = momentum  # Add momentum for faster convergence
        self.timeout = timeout  # Timeout in seconds
        self.num_inits = num_inits # Number of random initializations
        
    def _compute_gradient(self, simulation: Simulation, coil_config: CoilConfig, start_time: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute numerical gradients for phase and amplitude using central differences.
        Checks against the provided start_time for timeout.
        """
        phase_grad = np.zeros_like(coil_config.phase)
        amp_grad = np.zeros_like(coil_config.amplitude)
        
        # Check timeout at the beginning of gradient computation for this step
        if time.time() - start_time > self.timeout:
             raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded before computing gradient.")
        
        # Base cost calculation (does not require gradient-specific timeout check)
        # base_cost = self.cost_function(simulation(coil_config)) # Base cost not needed for central diff
        
        # Compute phase gradients
        for i in range(len(coil_config.phase)):
            # Check timeout before each simulation call within gradient computation
            if time.time() - start_time > self.timeout:
                 raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during phase gradient computation.")
            
            # Forward difference
            config_plus = CoilConfig(
                phase=coil_config.phase.copy(),
                amplitude=coil_config.amplitude.copy()
            )
            config_plus.phase[i] += self.epsilon
            
            # Backward difference
            config_minus = CoilConfig(
                phase=coil_config.phase.copy(),
                amplitude=coil_config.amplitude.copy()
            )
            config_minus.phase[i] -= self.epsilon
            
            cost_plus = self.cost_function(simulation(config_plus))
            # Check timeout again after potentially long simulation call
            if time.time() - start_time > self.timeout:
                 raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during phase gradient computation.")
            
            cost_minus = self.cost_function(simulation(config_minus))
            # Check timeout again
            if time.time() - start_time > self.timeout:
                 raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during phase gradient computation.")
            
            phase_grad[i] = (cost_plus - cost_minus) / (2 * self.epsilon)
        
        # Compute amplitude gradients
        for i in range(len(coil_config.amplitude)):
            # Check timeout before each simulation call
            if time.time() - start_time > self.timeout:
                 raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during amplitude gradient computation.")
            
            # Forward difference
            config_plus = CoilConfig(
                phase=coil_config.phase.copy(),
                amplitude=coil_config.amplitude.copy()
            )
            config_plus.amplitude[i] += self.epsilon
            
            # Backward difference
            config_minus = CoilConfig(
                phase=coil_config.phase.copy(),
                amplitude=coil_config.amplitude.copy()
            )
            config_minus.amplitude[i] -= self.epsilon
            
            cost_plus = self.cost_function(simulation(config_plus))
            # Check timeout
            if time.time() - start_time > self.timeout:
                 raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during amplitude gradient computation.")
            
            cost_minus = self.cost_function(simulation(config_minus))
             # Check timeout
            if time.time() - start_time > self.timeout:
                 raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during amplitude gradient computation.")
            
            amp_grad[i] = (cost_plus - cost_minus) / (2 * self.epsilon)
        
        return phase_grad, amp_grad
        
    def optimize(self, simulation: Simulation) -> CoilConfig:
        overall_start_time = time.time()
        best_coil_config_overall = None
        best_cost_overall = -np.inf if self.direction == "maximize" else np.inf
        initial_cost_better = (lambda current, best: current > best) if self.direction == "maximize" else (lambda current, best: current < best)
        
        print(f"Starting Gradient Descent optimization with {self.num_inits} initializations...")
        
        overall_timeout_reached = False
        for init_run in range(self.num_inits):
            if overall_timeout_reached:
                break
            
            print(f"--- Initialization Run {init_run + 1}/{self.num_inits} ---")
            
            # Check overall timeout before starting a new run
            if time.time() - overall_start_time > self.timeout:
                print(f"\nOverall optimization stopped due to timeout ({self.timeout} seconds) before starting run {init_run + 1}.")
                overall_timeout_reached = True
                break
            
            current_best_coil_config_run = None
            current_best_cost_run = -np.inf if self.direction == "maximize" else np.inf
            run_timed_out = False
            
            try:
                # Initialize with random configuration for this run
                coil_config = CoilConfig(
                    phase=np.random.uniform(low=0, high=2*np.pi, size=(8,)),
                    amplitude=np.random.uniform(low=0, high=1, size=(8,))
                )
                # Check timeout after simulation call
                if time.time() - overall_start_time > self.timeout:
                    raise TimeoutError("Overall timeout reached during initialization")
                
                current_cost_run = self.cost_function(simulation(coil_config))
                # Check timeout after simulation call
                if time.time() - overall_start_time > self.timeout:
                    raise TimeoutError("Overall timeout reached during initial cost evaluation")
                
                current_best_coil_config_run = coil_config
                current_best_cost_run = current_cost_run
                print(f"  Initial cost for run {init_run + 1}: {current_best_cost_run:.4f}")
                
                # Initialize momentum terms
                phase_velocity = np.zeros_like(coil_config.phase)
                amp_velocity = np.zeros_like(coil_config.amplitude)
                
                pbar = trange(self.max_iter, desc=f"Run {init_run + 1}", leave=False)
                for i in pbar:
                    # Check overall timeout at the start of each iteration
                    if time.time() - overall_start_time > self.timeout:
                        pbar.close()
                        print(f"\nOptimization stopped during run {init_run + 1} iteration {i} due to overall timeout ({self.timeout} seconds)")
                        run_timed_out = True # Mark run as timed out
                        break # Break inner loop
                    
                    # Compute gradients, passing the overall start time for timeout checks within _compute_gradient
                    phase_grad, amp_grad = self._compute_gradient(simulation, coil_config, overall_start_time)
                    
                    # Update velocities with momentum
                    update_direction = 1 if self.direction == "maximize" else -1
                    phase_velocity = self.momentum * phase_velocity + update_direction * self.learning_rate * phase_grad
                    amp_velocity = self.momentum * amp_velocity + update_direction * self.learning_rate * amp_grad
                    
                    # Update parameters
                    coil_config.phase += phase_velocity
                    coil_config.amplitude += amp_velocity
                    
                    # Ensure constraints
                    coil_config.phase = np.mod(coil_config.phase, 2*np.pi)
                    coil_config.amplitude = np.clip(coil_config.amplitude, 0, 1)
                    
                    # Evaluate new configuration
                    # Check timeout before simulation call
                    if time.time() - overall_start_time > self.timeout:
                         pbar.close()
                         print(f"\nOptimization stopped during run {init_run + 1} iteration {i} due to overall timeout ({self.timeout} seconds) before cost evaluation")
                         run_timed_out = True
                         break # Break inner loop
                    
                    current_cost = self.cost_function(simulation(coil_config))
                    # Check timeout after simulation call
                    if time.time() - overall_start_time > self.timeout:
                         pbar.close()
                         print(f"\nOptimization stopped during run {init_run + 1} iteration {i} due to overall timeout ({self.timeout} seconds) after cost evaluation")
                         run_timed_out = True
                         break # Break inner loop
                    
                    if initial_cost_better(current_cost, current_best_cost_run):
                        current_best_cost_run = current_cost
                        current_best_coil_config_run = CoilConfig(
                            phase=coil_config.phase.copy(),
                            amplitude=coil_config.amplitude.copy()
                        )
                    
                    pbar.set_postfix_str(f"Best cost {current_best_cost_run:.4f}")
                
                # End of inner loop (max_iter or timeout break)
                if not pbar.disable: # Only close if it wasn't already closed by timeout break
                     pbar.close()
                if not run_timed_out:
                    print(f"  Run {init_run + 1} finished after {self.max_iter} iterations. Best cost for this run: {current_best_cost_run:.4f}")
                else:
                    overall_timeout_reached = True # Signal to stop outer loop
            
            except TimeoutError as e:
                 # Handle timeout from initialization or _compute_gradient
                 print(f"\nTimeout occurred during run {init_run + 1}: {str(e)}")
                 overall_timeout_reached = True # Signal to stop outer loop
                 # Ensure pbar is closed if it was opened
                 if 'pbar' in locals() and hasattr(pbar, 'close') and not pbar.disable:
                     pbar.close()
            
            # Compare the best result of this run (if any) with the overall best
            if current_best_coil_config_run is not None:
                if best_coil_config_overall is None or initial_cost_better(current_best_cost_run, best_cost_overall):
                    print(f"  *** New overall best cost found: {current_best_cost_run:.4f} (previous: {best_cost_overall if best_cost_overall != -np.inf and best_cost_overall != np.inf else 'N/A'}) ***")
                    best_cost_overall = current_best_cost_run
                    best_coil_config_overall = current_best_coil_config_run
            else:
                 print(f"  Run {init_run + 1} did not produce a valid result (likely immediate timeout or error).")
        
        # After all runs (or timeout)
        print(f"--- Optimization finished. Overall best cost: {best_cost_overall if best_cost_overall != -np.inf and best_cost_overall != np.inf else 'N/A'} ---")
        if best_coil_config_overall is None:
            print("\nWarning: Optimization failed to find any valid configuration across all runs.")
            # Return a default or raise an error? Returning default for now.
            return CoilConfig(phase=np.zeros((8,)), amplitude=np.zeros((8,)))
        
        return best_coil_config_overall 