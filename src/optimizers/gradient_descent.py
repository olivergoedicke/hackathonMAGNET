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
                 timeout: float = 300) -> None:  # 300 seconds = 5 minutes
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # For numerical gradient computation
        self.momentum = momentum  # Add momentum for faster convergence
        self.timeout = timeout  # Timeout in seconds
        
    def _compute_gradient(self, simulation: Simulation, coil_config: CoilConfig, start_time: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute numerical gradients for phase and amplitude using central differences.
        Takes advantage of the masked field computation in simulation._shift_field.
        """
        phase_grad = np.zeros_like(coil_config.phase)
        amp_grad = np.zeros_like(coil_config.amplitude)
        
        # Base cost for current configuration
        base_cost = self.cost_function(simulation(coil_config))
        
        # Compute phase gradients
        for i in range(len(coil_config.phase)):
            # Check timeout
            if time.time() - start_time > self.timeout:
                raise TimeoutError("Optimization exceeded time limit of 5 minutes")
                
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
            
            # Compute costs using masked field computation
            cost_plus = self.cost_function(simulation(config_plus))
            cost_minus = self.cost_function(simulation(config_minus))
            phase_grad[i] = (cost_plus - cost_minus) / (2 * self.epsilon)
        
        # Compute amplitude gradients
        for i in range(len(coil_config.amplitude)):
            # Check timeout
            if time.time() - start_time > self.timeout:
                raise TimeoutError("Optimization exceeded time limit of 5 minutes")
                
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
            
            # Compute costs using masked field computation
            cost_plus = self.cost_function(simulation(config_plus))
            cost_minus = self.cost_function(simulation(config_minus))
            amp_grad[i] = (cost_plus - cost_minus) / (2 * self.epsilon)
            
        return phase_grad, amp_grad
        
    def optimize(self, simulation: Simulation):
        start_time = time.time()
        
        best_overall_config = None
        best_overall_cost = -np.inf if self.direction == "maximize" else np.inf
        
        num_initializations = 300  # Number of random initializations
        
        print(f"Running {num_initializations} random initializations...")
        init_pbar = trange(num_initializations, desc="Initializations")
        for _ in init_pbar:
            # Check timeout during initialization phase
            if time.time() - start_time > self.timeout:
                init_pbar.close()
                print("Timeout during initialization phase.")
                # If timeout happens during init, return None or raise error?
                # For now, return None or the best found so far if any.
                return best_overall_config # Might be None if no init finished

            # Initialize with random configuration
            initial_coil_config = CoilConfig(
                phase=np.random.uniform(low=0, high=2*np.pi, size=(8,)),
                amplitude=np.random.uniform(low=0, high=1, size=(8,))
            )
            
            initial_cost = self.cost_function(simulation(initial_coil_config))
            
            # Update best overall if this initialization is better
            if ((self.direction == "maximize" and initial_cost > best_overall_cost) or
                (self.direction == "minimize" and initial_cost < best_overall_cost)):
                best_overall_cost = initial_cost
                best_overall_config = initial_coil_config
                init_pbar.set_postfix_str(f"Best initial cost {best_overall_cost:.2f}")

        if best_overall_config is None:
            print("No successful initialization completed within the time limit.")
            return None # Or raise an exception

        print(f"Starting optimization from best initial cost: {best_overall_cost:.2f}")
        # Start optimization from the best initial configuration
        coil_config = best_overall_config 
        best_cost = best_overall_cost
        best_coil_config = best_overall_config

        try:
            # Initialize momentum terms
            phase_velocity = np.zeros_like(coil_config.phase)
            amp_velocity = np.zeros_like(coil_config.amplitude)
            
            pbar = trange(self.max_iter)
            for i in pbar:
                # Check timeout
                if time.time() - start_time > self.timeout:
                    pbar.close()
                    print("\nOptimization stopped due to timeout (5 minutes)")
                    break
                
                # Compute gradients
                phase_grad, amp_grad = self._compute_gradient(simulation, coil_config, start_time)
                
                # Update velocities with momentum
                if self.direction == "maximize":
                    phase_velocity = self.momentum * phase_velocity + self.learning_rate * phase_grad
                    amp_velocity = self.momentum * amp_velocity + self.learning_rate * amp_grad
                else:
                    phase_velocity = self.momentum * phase_velocity - self.learning_rate * phase_grad
                    amp_velocity = self.momentum * amp_velocity - self.learning_rate * amp_grad
                
                # Update parameters
                coil_config.phase += phase_velocity
                coil_config.amplitude += amp_velocity
                
                # Ensure constraints
                coil_config.phase = np.mod(coil_config.phase, 2*np.pi)  # Keep phases in [0, 2π]
                coil_config.amplitude = np.clip(coil_config.amplitude, 0, 1)  # Keep amplitudes in [0, 1]
                
                # Evaluate new configuration
                current_cost = self.cost_function(simulation(coil_config))
                if ((self.direction == "maximize" and current_cost > best_cost) or 
                    (self.direction == "minimize" and current_cost < best_cost)):
                    best_cost = current_cost
                    best_coil_config = CoilConfig(
                        phase=coil_config.phase.copy(),
                        amplitude=coil_config.amplitude.copy()
                    )
                
                pbar.set_postfix_str(f"Best cost {best_cost:.2f}")
        
        except TimeoutError as e:
            print(f"\n{str(e)}")
        
        finally:
            # Return the best configuration found so far during optimization
            return best_coil_config 