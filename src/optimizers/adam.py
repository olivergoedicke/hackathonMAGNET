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
                 timeout: float = 300) -> None:  # 300 seconds = 5 minutes
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.num_epsilon = num_epsilon # Epsilon for numerical gradient
        self.timeout = timeout  # Timeout in seconds

    def _compute_gradient(self, simulation: Simulation, coil_config: CoilConfig, start_time: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute numerical gradients for phase and amplitude using central differences.
        """
        phase_grad = np.zeros_like(coil_config.phase)
        amp_grad = np.zeros_like(coil_config.amplitude)

        # Base cost for current configuration
        base_cost = self.cost_function(simulation(coil_config))

        # Compute phase gradients
        for i in range(len(coil_config.phase)):
            # Check timeout
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Optimization exceeded time limit of {self.timeout} seconds during gradient computation")

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
            cost_minus = self.cost_function(simulation(config_minus))
            phase_grad[i] = (cost_plus - cost_minus) / (2 * self.num_epsilon)

        # Compute amplitude gradients
        for i in range(len(coil_config.amplitude)):
            # Check timeout
            if time.time() - start_time > self.timeout:
                 raise TimeoutError(f"Optimization exceeded time limit of {self.timeout} seconds during gradient computation")

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
            cost_minus = self.cost_function(simulation(config_minus))
            amp_grad[i] = (cost_plus - cost_minus) / (2 * self.num_epsilon)

        return phase_grad, amp_grad

    def optimize(self, simulation: Simulation) -> CoilConfig:
        start_time = time.time()
        best_coil_config = None
        best_cost = -np.inf if self.direction == "maximize" else np.inf

        try:
            # Initialize with random configuration
            coil_config = CoilConfig(
                phase=np.random.uniform(low=0, high=2*np.pi, size=(8,)),
                amplitude=np.random.uniform(low=0, high=1, size=(8,))
            )

            best_coil_config = coil_config # Start with the initial config as best
            best_cost = self.cost_function(simulation(coil_config))

            # Initialize Adam moment estimates
            m_phase = np.zeros_like(coil_config.phase)
            v_phase = np.zeros_like(coil_config.phase)
            m_amp = np.zeros_like(coil_config.amplitude)
            v_amp = np.zeros_like(coil_config.amplitude)
            t = 0 # Timestep

            pbar = trange(self.max_iter)
            for i in pbar:
                # Check timeout
                if time.time() - start_time > self.timeout:
                    pbar.close()
                    print(f"\nOptimization stopped due to timeout ({self.timeout} seconds)")
                    break

                t += 1 # Increment timestep

                # Compute gradients
                phase_grad, amp_grad = self._compute_gradient(simulation, coil_config, start_time)

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

                pbar.set_postfix_str(f"Best cost {best_cost:.4f}")

        except TimeoutError as e:
            print(f"\n{str(e)}")

        finally:
            if best_coil_config is None:
                # If timeout happened before the first evaluation or initialization failed
                print("\nWarning: Optimization did not complete a single iteration or failed to initialize.")
                # Return a default or initial configuration might be better here depending on requirements
                return CoilConfig(phase=np.zeros((8,)), amplitude=np.zeros((8,)))
            # Return the best configuration found so far
            return best_coil_config 