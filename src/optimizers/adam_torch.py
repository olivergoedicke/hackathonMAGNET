# File: src/optimizers/adam_torch.py

import torch
import numpy as np
import time
from tqdm import trange

from ..data.simulation import Simulation
from ..data.dataclasses import CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

class AdamTorchOptimizer(BaseOptimizer):
    """
    Optimizes coil configurations using the Adam optimizer from PyTorch.

    Uses NUMERICAL GRADIENTS because the underlying simulation and cost
    functions are not implemented in PyTorch. The Adam update rule is
    applied using these numerical gradients.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 100,
                 learning_rate: float = 0.01,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,        # Adam epsilon
                 grad_epsilon: float = 1e-6, # Epsilon for numerical gradient calculation
                 timeout: float = 300,     # 5 minutes
                 device: str | None = None) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.grad_epsilon = grad_epsilon # Epsilon for numerical gradient
        self.timeout = timeout

        # --- Device Selection ---
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AdamTorchOptimizer using device: {self.device}")

    def _objective_function(self, simulation: Simulation, phase_np: np.ndarray, amplitude_np: np.ndarray) -> float:
        """ Evaluates the cost function for a given numpy configuration. """
        try:
            # Apply constraints before evaluation - important!
            phase_np_constrained = np.mod(phase_np, 2 * np.pi)
            amplitude_np_constrained = np.clip(amplitude_np, 0, 1)

            config = CoilConfig(phase=phase_np_constrained, amplitude=amplitude_np_constrained)
            cost = self.cost_function(simulation(config))

            # Handle maximization vs minimization for internal objective
            cost_value = float(cost) # Ensure standard float
            if self.direction == "maximize":
                cost_value = -cost_value

            # Return infinity if cost is NaN or Inf (indicates bad parameters)
            if np.isnan(cost_value) or np.isinf(cost_value):
                return float('inf')

            return cost_value

        except Exception as e:
            # print(f"Warning: Error during cost evaluation: {e}")
            return float('inf') # Return infinity on error

    def _compute_numerical_gradient(self, simulation: Simulation, phase_np: np.ndarray, amplitude_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Computes numerical gradients for phase and amplitude using central differences. """
        phase_grad = np.zeros_like(phase_np)
        amp_grad = np.zeros_like(amplitude_np)

        # Combine parameters for easier iteration
        params = np.concatenate((phase_np, amplitude_np))
        grad = np.zeros_like(params)
        
        # Calculate base cost once if needed for forward/backward differences
        base_cost = None 

        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()

            params_plus[i] += self.grad_epsilon
            params_minus[i] -= self.grad_epsilon

            cost_plus = self._objective_function(simulation, params_plus[:8], params_plus[8:])
            cost_minus = self._objective_function(simulation, params_minus[:8], params_minus[8:])

            # Handle cases where cost evaluation returns infinity (use one-sided diff if possible)
            if np.isinf(cost_plus) or np.isinf(cost_minus):
                 if base_cost is None: # Calculate base cost only if needed
                      base_cost = self._objective_function(simulation, params[:8], params[8:])
                 
                 if not np.isinf(cost_plus) and not np.isinf(base_cost):
                     grad[i] = (cost_plus - base_cost) / self.grad_epsilon # Forward difference
                 elif not np.isinf(cost_minus) and not np.isinf(base_cost):
                     grad[i] = (base_cost - cost_minus) / self.grad_epsilon # Backward difference
                 else:
                     grad[i] = 0 # Cannot compute gradient
                     # print(f"Warning: Could not compute gradient for param {i} due to Inf cost.")
            else:
                 # Central difference
                 grad[i] = (cost_plus - cost_minus) / (2 * self.grad_epsilon)

        phase_grad = grad[:8]
        amp_grad = grad[8:]

        # Ensure gradients are finite, replace NaN/Inf with 0
        phase_grad = np.nan_to_num(phase_grad, nan=0.0, posinf=0.0, neginf=0.0)
        amp_grad = np.nan_to_num(amp_grad, nan=0.0, posinf=0.0, neginf=0.0)

        return phase_grad, amp_grad

    def optimize(self, simulation: Simulation) -> CoilConfig:
        start_time = time.time()

        # --- Initialization ---
        initial_phase = np.random.uniform(0, 2 * np.pi, size=(8,))
        initial_amplitude = np.ones((8,)) # Start amplitude at 1

        # Convert initial parameters to torch tensors
        phase = torch.tensor(initial_phase, dtype=torch.float32, device=self.device)
        amplitude = torch.tensor(initial_amplitude, dtype=torch.float32, device=self.device)
        # Ensure they require grad for the optimizer state, even if we set .grad manually
        phase.requires_grad_(True)
        amplitude.requires_grad_(True)


        # Setup Adam optimizer
        optimizer = torch.optim.Adam([phase, amplitude],
                                     lr=self.learning_rate,
                                     betas=self.betas,
                                     eps=self.eps)

        # --- Initial Evaluation ---
        best_phase_np = phase.detach().cpu().numpy()
        best_amplitude_np = amplitude.detach().cpu().numpy()
        # Use the objective function which handles min/max direction internally
        best_internal_cost = self._objective_function(simulation, best_phase_np, best_amplitude_np)

        if np.isinf(best_internal_cost):
             print("Error: Initial configuration yields invalid cost. Returning initial guess.")
             # Return the initial numpy arrays in a CoilConfig
             return CoilConfig(phase=initial_phase, amplitude=initial_amplitude)

        # Store the display cost (actual cost according to min/max direction)
        best_display_cost = -best_internal_cost if self.direction == "maximize" else best_internal_cost
        print(f"Initial cost: {best_display_cost:.4f}")


        # --- Optimization Loop ---
        pbar = trange(self.max_iter)
        for i in pbar:
            if time.time() - start_time > self.timeout:
                print(f"\nOptimization stopped due to timeout ({self.timeout}s)")
                break

            # Get current parameters as numpy arrays for gradient calculation
            current_phase_np = phase.detach().cpu().numpy()
            current_amplitude_np = amplitude.detach().cpu().numpy()

            # --- Calculate Numerical Gradients ---
            phase_grad_np, amp_grad_np = self._compute_numerical_gradient(
                simulation, current_phase_np, current_amplitude_np
            )

            # --- Manually Assign Gradients to Tensors ---
            with torch.no_grad(): # Ensure this operation isn't tracked
                phase.grad = torch.tensor(phase_grad_np, dtype=torch.float32, device=self.device)
                amplitude.grad = torch.tensor(amp_grad_np, dtype=torch.float32, device=self.device)

            # --- Optimizer Step ---
            # Checks if gradients exist before stepping
            if phase.grad is not None and amplitude.grad is not None:
                optimizer.step() # Apply Adam update using the manually assigned gradients
            else:
                print(f"\nWarning: Gradients are None at iter {i}. Skipping optimizer step.")

            # Zero gradients *after* the step for the next iteration
            optimizer.zero_grad()

            # --- Apply Constraints to Tensors ---
            # Apply constraints directly to the tensors after the optimizer step
            with torch.no_grad():
                phase.data = torch.remainder(phase.data, 2 * np.pi)
                amplitude.data = torch.clamp(amplitude.data, 0.0, 1.0)

            # --- Evaluate Current Cost and Update Best ---
            # Evaluate with the parameters *after* step and constraints
            eval_phase_np = phase.detach().cpu().numpy()
            eval_amplitude_np = amplitude.detach().cpu().numpy()
            current_internal_cost = self._objective_function(simulation, eval_phase_np, eval_amplitude_np)

            if current_internal_cost < best_internal_cost:
                best_internal_cost = current_internal_cost
                best_phase_np = eval_phase_np.copy()      # Store the best numpy arrays
                best_amplitude_np = eval_amplitude_np.copy()

                # Update display cost
                best_display_cost = -best_internal_cost if self.direction == "maximize" else best_internal_cost
                # Optionally calculate grad norm for display
                current_grad_norm = np.linalg.norm(np.concatenate((phase_grad_np, amp_grad_np)))
                pbar.set_postfix_str(f"Best cost {best_display_cost:.4f}, Grad norm {current_grad_norm:.2e}")


        # --- Return Best Found Configuration ---
        print(f"\nOptimization finished. Best cost found: {best_display_cost:.4f}")

        # Final check for NaNs in results (should be less likely with constraints)
        if np.isnan(best_phase_np).any() or np.isnan(best_amplitude_np).any():
             print("Warning: Best parameters contain NaN. Returning initial guess.")
             return CoilConfig(phase=initial_phase, amplitude=initial_amplitude)

        return CoilConfig(phase=best_phase_np, amplitude=best_amplitude_np)
