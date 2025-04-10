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
                 num_inits: int = 300,       # Number of random initializations
                 device: str | None = None) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.grad_epsilon = grad_epsilon # Epsilon for numerical gradient
        self.timeout = timeout
        self.num_inits = num_inits

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
        overall_start_time = time.time()
        overall_best_config = None
        overall_best_internal_cost = float('inf')
        initial_cost_better = (lambda current, best: current < best) # Using internal cost (minimization)
        outer_run_count = 0

        print(f"Starting repeated optimization cycles (batch_size={self.num_inits}, steps_per_batch={self.max_iter})")

        while time.time() - overall_start_time < self.timeout:
            outer_run_count += 1
            print(f"--- Outer Optimization Cycle {outer_run_count} ---")
            cycle_start_time = time.time()

            # --- Initial Configuration Evaluation (for this cycle) ---+
            best_initial_config_np_cycle = None
            best_initial_internal_cost_cycle = float('inf')

            print(f"  Evaluating {self.num_inits} initial random configurations for cycle {outer_run_count}...")
            pbar_init = trange(self.num_inits, desc=f"Cycle {outer_run_count} Init Eval", leave=False)
            initial_eval_timed_out_cycle = False
            for _ in pbar_init:
                # Check overall timeout *before* evaluation
                if time.time() - overall_start_time >= self.timeout:
                    pbar_init.close()
                    print(f"\nTimeout ({self.timeout}s) occurred during initial evaluation of cycle {outer_run_count}.")
                    initial_eval_timed_out_cycle = True
                    break

                # Generate random configuration (NumPy)
                initial_phase_np = np.random.uniform(0, 2 * np.pi, size=(8,))
                initial_amplitude_np = np.random.uniform(0, 1, size=(8,))

                # Evaluate its cost
                try:
                    current_initial_internal_cost = self._objective_function(simulation, initial_phase_np, initial_amplitude_np)

                    if initial_cost_better(current_initial_internal_cost, best_initial_internal_cost_cycle):
                        best_initial_internal_cost_cycle = current_initial_internal_cost
                        best_initial_config_np_cycle = (initial_phase_np, initial_amplitude_np)
                        # Display actual cost based on direction for this cycle's best init
                        best_initial_display_cost_cycle = -best_initial_internal_cost_cycle if self.direction == "maximize" else best_initial_internal_cost_cycle
                        pbar_init.set_postfix_str(f"Best init cost {best_initial_display_cost_cycle:.4f}")

                except Exception as e:
                    # print(f"Warning: Error evaluating initial config in cycle {outer_run_count}: {e}")
                    pass # Continue to next initial config

            if not pbar_init.disable:
                 pbar_init.close()

            # If timeout occurred during initial eval, break the outer loop
            if initial_eval_timed_out_cycle:
                break

            if best_initial_config_np_cycle is None:
                print(f"\nWarning: Failed to find any valid initial configuration in cycle {outer_run_count}. Skipping to next cycle.")
                # Check timeout before continuing to potentially avoid infinite loop if objective always fails
                if time.time() - overall_start_time >= self.timeout:
                    print("Timeout reached after failed initial evaluation phase.")
                    break
                continue # Skip optimization phase for this cycle

            # --- Start Optimization from Best Initial Configuration (for this cycle) ---+
            best_initial_phase_np_cycle, best_initial_amplitude_np_cycle = best_initial_config_np_cycle
            best_initial_display_cost_cycle = -best_initial_internal_cost_cycle if self.direction == "maximize" else best_initial_internal_cost_cycle
            print(f"  Starting optimization for cycle {outer_run_count} from initial cost: {best_initial_display_cost_cycle:.4f}")

            # Convert the cycle's best initial parameters to torch tensors
            phase = torch.tensor(best_initial_phase_np_cycle, dtype=torch.float32, device=self.device, requires_grad=True)
            amplitude = torch.tensor(best_initial_amplitude_np_cycle, dtype=torch.float32, device=self.device, requires_grad=True)

            # Setup Adam optimizer for this cycle
        optimizer = torch.optim.Adam([phase, amplitude],
                                     lr=self.learning_rate,
                                     betas=self.betas,
                                     eps=self.eps)

            # Track best cost found *during this cycle's optimization run*
            # Initialize with the cost of the starting point for this cycle
            run_best_internal_cost_cycle = best_initial_internal_cost_cycle
            run_best_config_cycle = CoilConfig(phase=best_initial_phase_np_cycle.copy(), amplitude=best_initial_amplitude_np_cycle.copy())

            # --- Optimization Loop (for this cycle) ---
            optimization_timed_out_cycle = False
            try:
                pbar = trange(self.max_iter, desc=f"Cycle {outer_run_count} Opt Run", leave=False)
        for i in pbar:
                    # Check overall timeout at the start of each iteration
                    if time.time() - overall_start_time >= self.timeout:
                        pbar.close()
                        print(f"\nTimeout ({self.timeout}s) occurred during optimization step {i} of cycle {outer_run_count}.")
                        optimization_timed_out_cycle = True
                break

                    # Get current parameters as numpy arrays
            current_phase_np = phase.detach().cpu().numpy()
            current_amplitude_np = amplitude.detach().cpu().numpy()

            # --- Calculate Numerical Gradients ---
            phase_grad_np, amp_grad_np = self._compute_numerical_gradient(
                simulation, current_phase_np, current_amplitude_np
            )
                    # Check timeout *after* gradient calculation
                    if time.time() - overall_start_time >= self.timeout:
                        pbar.close()
                        print(f"\nTimeout ({self.timeout}s) occurred after gradient calc in step {i} of cycle {outer_run_count}.")
                        optimization_timed_out_cycle = True
                        break

            # --- Manually Assign Gradients to Tensors ---
                    with torch.no_grad():
                phase.grad = torch.tensor(phase_grad_np, dtype=torch.float32, device=self.device)
                amplitude.grad = torch.tensor(amp_grad_np, dtype=torch.float32, device=self.device)

            # --- Optimizer Step ---
            if phase.grad is not None and amplitude.grad is not None:
                        optimizer.step() # Apply Adam update
            else:
                        print(f"\nWarning: Gradients are None at iter {i} in cycle {outer_run_count}. Skipping optimizer step.")

                    # Zero gradients *after* the step
            optimizer.zero_grad()

            # --- Apply Constraints to Tensors ---
            with torch.no_grad():
                phase.data = torch.remainder(phase.data, 2 * np.pi)
                amplitude.data = torch.clamp(amplitude.data, 0.0, 1.0)

                    # --- Evaluate Current Cost and Update Cycle Best ---
            eval_phase_np = phase.detach().cpu().numpy()
            eval_amplitude_np = amplitude.detach().cpu().numpy()
            current_internal_cost = self._objective_function(simulation, eval_phase_np, eval_amplitude_np)
                    # Check timeout *after* cost evaluation
                    if time.time() - overall_start_time >= self.timeout:
                        pbar.close()
                        print(f"\nTimeout ({self.timeout}s) occurred after cost eval in step {i} of cycle {outer_run_count}.")
                        optimization_timed_out_cycle = True
                        break

                    # Update the best cost found *during this cycle's run*
                    if initial_cost_better(current_internal_cost, run_best_internal_cost_cycle):
                        run_best_internal_cost_cycle = current_internal_cost
                        run_best_config_cycle = CoilConfig(phase=eval_phase_np.copy(), amplitude=eval_amplitude_np.copy())

                        # Update display cost for progress bar for this cycle
                        run_best_display_cost_cycle = -run_best_internal_cost_cycle if self.direction == "maximize" else run_best_internal_cost_cycle
                        current_grad_norm = np.linalg.norm(np.concatenate((phase_grad_np, amp_grad_np))) if phase_grad_np is not None and amp_grad_np is not None else float('nan')
                        pbar.set_postfix_str(f"Cycle best cost {run_best_display_cost_cycle:.4f}, Grad norm {current_grad_norm:.2e}")

                # End of optimization loop (max_iter or timeout break)
                if not pbar.disable:
                     pbar.close()

            except TimeoutError as e: # Should be caught by inner checks, but as safety
                print(f"\nTimeoutError during optimization run of cycle {outer_run_count}: {str(e)}")
                optimization_timed_out_cycle = True # Ensure we check overall best below
                if 'pbar' in locals() and hasattr(pbar, 'close') and not pbar.disable:
                    pbar.close()
            except Exception as e:
                print(f"\nError occurred during optimization run of cycle {outer_run_count}: {e}")
                if 'pbar' in locals() and hasattr(pbar, 'close') and not pbar.disable:
                    pbar.close()
                # run_best_config_cycle might still hold the best state before the error

            # --- Update Overall Best After Cycle ---+
            if run_best_config_cycle is not None:
                # Compare this cycle's best internal cost with the overall best
                if initial_cost_better(run_best_internal_cost_cycle, overall_best_internal_cost):
                    overall_best_internal_cost = run_best_internal_cost_cycle
                    overall_best_config = run_best_config_cycle # Store the actual best CoilConfig object from the cycle
                    overall_best_display_cost = -overall_best_internal_cost if self.direction == "maximize" else overall_best_internal_cost
                    print(f"  *** New overall best cost found: {overall_best_display_cost:.4f} in cycle {outer_run_count} ***")

            # Ensure the outer loop terminates if a timeout occurred anywhere in the cycle
            if initial_eval_timed_out_cycle or optimization_timed_out_cycle:
                 print(f"Terminating outer loop due to timeout during cycle {outer_run_count}.")
                 break

            # Optional: Print time taken for cycle
            cycle_duration = time.time() - cycle_start_time
            print(f"  Cycle {outer_run_count} duration: {cycle_duration:.2f}s")

        # --- End of Outer Loop ---+

        # --- Return Overall Best Found Configuration ---+
        print("\n--- Repeated optimization finished. --- ")
        if overall_best_config is not None:
            overall_final_display_cost = -overall_best_internal_cost if self.direction == "maximize" else overall_best_internal_cost
            print(f"Overall best cost found across {outer_run_count} cycles: {overall_final_display_cost:.4f}")
            # Final check for NaNs in the absolute best result
            if np.isnan(overall_best_config.phase).any() or np.isnan(overall_best_config.amplitude).any():
                print("\nWarning: Overall best parameters contain NaN. This might indicate issues during optimization.")
                # Decide recovery strategy: return default? or the possibly NaN result?
                # Returning default for safety:
                print("\nReturning default zero configuration due to NaNs in overall best result.")
                return CoilConfig(phase=np.zeros((8,)), amplitude=np.zeros((8,)))
            # If no NaNs, return the best config found
            return overall_best_config
        else:
            # If no valid config was ever found across all cycles
            print("\nWarning: Optimization failed to find any valid configuration across all cycles.")
            return CoilConfig(phase=np.zeros((8,)), amplitude=np.zeros((8,)))
