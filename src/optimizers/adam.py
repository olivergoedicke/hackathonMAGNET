from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

from typing import Callable, Optional, Tuple
import numpy as np
import time

from tqdm import trange


class AdamOptimizer(BaseOptimizer):
    """
    AdamOptimizer uses the Adam optimization algorithm with numerical gradients
    to optimize coil configurations.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 100,
                 learning_rate: float = 0.001,  # Adam typically uses smaller learning rates
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,       # Epsilon for Adam stability
                 num_epsilon: float = 1e-6,   # Epsilon for numerical gradient calculation
                 timeout: float = 300,
                 num_inits: int = 5) -> None:  # 300 seconds = 5 minutes
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.num_epsilon = num_epsilon # Epsilon for numerical gradient
        self.timeout = timeout  # Timeout in seconds
        self.num_inits = num_inits # Number of random initializations

    def _compute_gradient(self, simulation: Simulation, coil_config: CoilConfig, start_time: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute numerical gradients for phase and amplitude using central differences.
        Checks against the provided overall start_time for timeout.
        """
        phase_grad = np.zeros_like(coil_config.phase)
        amp_grad = np.zeros_like(coil_config.amplitude)

        # Check timeout at the beginning of gradient computation
        if time.time() - start_time > self.timeout:
            raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded before computing gradient.")

        # base_cost = self.cost_function(simulation(coil_config)) # Base cost not needed for central diff

        # Compute phase gradients
        for i in range(len(coil_config.phase)):
            # Check timeout before simulation calls
            if time.time() - start_time > self.timeout:
                 raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during phase gradient computation.")

            # Forward difference
            config_plus = CoilConfig(
                phase=coil_config.phase.copy(),
                amplitude=coil_config.amplitude.copy()
            )
            config_plus.phase[i] += self.num_epsilon

            # Backward difference
            config_minus = CoilConfig(
                phase=coil_config.phase.copy(),
                amplitude=coil_config.amplitude.copy()
            )
            config_minus.phase[i] -= self.num_epsilon

            cost_plus = self.cost_function(simulation(config_plus))
            # Check timeout after simulation call
            if time.time() - start_time > self.timeout:
                 raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during phase gradient computation.")

            cost_minus = self.cost_function(simulation(config_minus))
             # Check timeout after simulation call
            if time.time() - start_time > self.timeout:
                 raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during phase gradient computation.")

            phase_grad[i] = (cost_plus - cost_minus) / (2 * self.num_epsilon)

        # Compute amplitude gradients
        for i in range(len(coil_config.amplitude)):
            # Check timeout before simulation calls
            if time.time() - start_time > self.timeout:
                  raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during amplitude gradient computation.")

            # Forward difference
            config_plus = CoilConfig(
                phase=coil_config.phase.copy(),
                amplitude=coil_config.amplitude.copy()
            )
            config_plus.amplitude[i] += self.num_epsilon

            # Backward difference
            config_minus = CoilConfig(
                phase=coil_config.phase.copy(),
                amplitude=coil_config.amplitude.copy()
            )
            config_minus.amplitude[i] -= self.num_epsilon

            cost_plus = self.cost_function(simulation(config_plus))
            # Check timeout after simulation call
            if time.time() - start_time > self.timeout:
                  raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during amplitude gradient computation.")

            cost_minus = self.cost_function(simulation(config_minus))
             # Check timeout after simulation call
            if time.time() - start_time > self.timeout:
                  raise TimeoutError(f"Overall optimization timeout ({self.timeout}s) exceeded during amplitude gradient computation.")

            amp_grad[i] = (cost_plus - cost_minus) / (2 * self.num_epsilon)

        return phase_grad, amp_grad

    def optimize(self, simulation: Simulation) -> CoilConfig:
        overall_start_time = time.time()
        best_coil_config_overall = None
        best_cost_overall = -np.inf if self.direction == "maximize" else np.inf
        initial_cost_better = (lambda current, best: current > best) if self.direction == "maximize" else (lambda current, best: current < best)

        print(f"Starting Adam optimization with {self.num_inits} initializations...")

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
                # Check timeout after initialization
                if time.time() - overall_start_time > self.timeout:
                    raise TimeoutError("Overall timeout reached during initialization")

                current_cost_run = self.cost_function(simulation(coil_config))
                # Check timeout after initial cost evaluation
                if time.time() - overall_start_time > self.timeout:
                    raise TimeoutError("Overall timeout reached during initial cost evaluation")

                current_best_coil_config_run = coil_config
                current_best_cost_run = current_cost_run
                print(f"  Initial cost for run {init_run + 1}: {current_best_cost_run:.4f}")

                # Initialize Adam moment estimates for this run
                m_phase = np.zeros_like(coil_config.phase)
                v_phase = np.zeros_like(coil_config.phase)
                m_amp = np.zeros_like(coil_config.amplitude)
                v_amp = np.zeros_like(coil_config.amplitude)
                t = 0 # Timestep for Adam bias correction

                pbar = trange(self.max_iter, desc=f"Run {init_run + 1}", leave=False)
                for i in pbar:
                    # Check overall timeout at the start of each iteration
                    if time.time() - overall_start_time > self.timeout:
                        pbar.close()
                        print(f"\nOptimization stopped during run {init_run + 1} iteration {i} due to overall timeout ({self.timeout} seconds)")
                        run_timed_out = True
                        break # Break inner loop

                    t += 1 # Increment timestep

                    # Compute gradients, passing overall start time for timeout checks
                    phase_grad, amp_grad = self._compute_gradient(simulation, coil_config, overall_start_time)

                    # Update biased first moment estimate
                    m_phase = self.beta1 * m_phase + (1 - self.beta1) * phase_grad
                    m_amp = self.beta1 * m_amp + (1 - self.beta1) * amp_grad

                    # Update biased second raw moment estimate
                    v_phase = self.beta2 * v_phase + (1 - self.beta2) * (phase_grad**2)
                    v_amp = self.beta2 * v_amp + (1 - self.beta2) * (amp_grad**2)

                    # Compute bias-corrected first moment estimate
                    m_hat_phase = m_phase / (1 - self.beta1**t)
                    m_hat_amp = m_amp / (1 - self.beta1**t)

                    # Compute bias-corrected second raw moment estimate
                    v_hat_phase = v_phase / (1 - self.beta2**t)
                    v_hat_amp = v_amp / (1 - self.beta2**t)

                    # Update parameters
                    update_direction = 1 if self.direction == "maximize" else -1
                    phase_update = self.learning_rate * m_hat_phase / (np.sqrt(v_hat_phase) + self.epsilon)
                    amp_update = self.learning_rate * m_hat_amp / (np.sqrt(v_hat_amp) + self.epsilon)

                    coil_config.phase += update_direction * phase_update
                    coil_config.amplitude += update_direction * amp_update

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
                if not pbar.disable:
                    pbar.close()
                if not run_timed_out:
                     print(f"  Run {init_run + 1} finished after {self.max_iter} iterations. Best cost for this run: {current_best_cost_run:.4f}")
                else:
                    overall_timeout_reached = True # Signal to stop outer loop

            except TimeoutError as e:
                # Handle timeout from initialization or _compute_gradient
                print(f"\nTimeout occurred during run {init_run + 1}: {str(e)}")
                overall_timeout_reached = True # Signal to stop outer loop
                if 'pbar' in locals() and hasattr(pbar, 'close') and not pbar.disable:
                    pbar.close()

            # Compare the best result of this run with the overall best
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
            return CoilConfig(phase=np.zeros((8,)), amplitude=np.zeros((8,))) # Return default

        return best_coil_config_overall 