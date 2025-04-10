# File: src/optimizers/conjugate_gradient_wong.py

import numpy as np
import time
from tqdm import trange
import math # For isnan

from ..data.simulation import Simulation
from ..data.dataclasses import CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

class ConjugateGradientWongOptimizer(BaseOptimizer):
    """
    Optimizes coil configurations using Conjugate Gradient Descent based on
    the method described by Wong et al. (Magn Reson Med 21, 39-48, 1991)[cite: 1].

    Uses numerical gradients as analytical derivatives from Biot-Savart [cite: 61]
    are not directly available from the Simulation class.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 100,
                 gtol: float = 1e-5, # Gradient tolerance for convergence
                 epsilon: float = 1e-6, # Step for numerical gradient
                 line_search_max_iter: int = 10, # Max iterations for line search
                 line_search_tol: float = 1e-4, # Tolerance for line search
                 timeout: float = 300) -> None: # 5 minutes
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.gtol = gtol
        self.epsilon = epsilon # For numerical gradient computation
        self.line_search_max_iter = line_search_max_iter
        self.line_search_tol = line_search_tol
        self.timeout = timeout

    def _objective_function(self, simulation: Simulation, phase_np: np.ndarray, amplitude_np: np.ndarray) -> float:
        """ Helper function to evaluate the cost, handling potential errors. """
        try:
            # Apply constraints before evaluation
            phase_np = np.mod(phase_np, 2 * np.pi)
            amplitude_np = np.clip(amplitude_np, 0, 1)
            
            config = CoilConfig(phase=phase_np, amplitude=amplitude_np)
            cost = self.cost_function(simulation(config))
            
            # Handle maximization vs minimization
            if self.direction == "maximize":
                cost = -cost
                
            # Return infinity if cost is NaN or Inf (bad parameters)
            if np.isnan(cost) or np.isinf(cost):
                return float('inf')
                
            return float(cost) # Ensure standard float
            
        except Exception as e:
            # print(f"Warning: Error during cost evaluation: {e}")
            return float('inf') # Return infinity on error

    def _compute_gradient(self, simulation: Simulation, phase_np: np.ndarray, amplitude_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Computes numerical gradients for phase and amplitude using central differences. """
        phase_grad = np.zeros_like(phase_np)
        amp_grad = np.zeros_like(amplitude_np)
        
        # Combine parameters for easier iteration
        params = np.concatenate((phase_np, amplitude_np))
        grad = np.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            
            params_plus[i] += self.epsilon
            params_minus[i] -= self.epsilon

            cost_plus = self._objective_function(simulation, params_plus[:8], params_plus[8:])
            cost_minus = self._objective_function(simulation, params_minus[:8], params_minus[8:])

            # Handle cases where cost evaluation returns infinity
            if np.isinf(cost_plus) or np.isinf(cost_minus):
                 # Attempt forward/backward difference if central fails
                 if not np.isinf(cost_plus):
                     base_cost = self._objective_function(simulation, params[:8], params[8:])
                     if not np.isinf(base_cost):
                         grad[i] = (cost_plus - base_cost) / self.epsilon
                     else: grad[i] = 0 # Cannot compute gradient
                 elif not np.isinf(cost_minus):
                      base_cost = self._objective_function(simulation, params[:8], params[8:])
                      if not np.isinf(base_cost):
                           grad[i] = (base_cost - cost_minus) / self.epsilon
                      else: grad[i] = 0
                 else:
                     grad[i] = 0 # Both failed, cannot compute gradient
            else:
                 grad[i] = (cost_plus - cost_minus) / (2 * self.epsilon)


        phase_grad = grad[:8]
        amp_grad = grad[8:]

        return phase_grad, amp_grad

    def _line_search(self, simulation: Simulation, phase_np: np.ndarray, amplitude_np: np.ndarray,
                     search_dir_phase: np.ndarray, search_dir_amp: np.ndarray,
                     start_time: float) -> float:
        """ Finds step size alpha that minimizes cost along search_dir using basic bracketing. """
        alpha = 1.0 # Initial guess for step size
        alpha_high = None
        alpha_low = 0.0

        # Cost at the starting point (alpha=0)
        cost_start = self._objective_function(simulation, phase_np, amplitude_np)

        for _ in range(self.line_search_max_iter):
            if time.time() - start_time > self.timeout:
                print("\nTimeout during line search.")
                return 0.0 # Return zero step on timeout

            # Calculate cost at current alpha
            phase_new = phase_np + alpha * search_dir_phase
            amp_new = amplitude_np + alpha * search_dir_amp
            cost_current = self._objective_function(simulation, phase_new, amp_new)

            # Basic check: if cost increases significantly, backtrack
            if cost_current > cost_start + self.line_search_tol * alpha: # Simplified Armijo-like condition
                alpha_high = alpha
            else:
                # If cost decreased or didn't increase much, try larger step
                alpha_low = alpha
                if alpha_high is None:
                    alpha *= 2.0 # Increase step size if we haven't bracketed yet
                else:
                    # If bracketed, move towards the middle
                     pass # Stay within bracket

            # If we have a bracket [alpha_low, alpha_high]
            if alpha_high is not None:
                # Check if bracket is small enough
                if abs(alpha_high - alpha_low) < self.line_search_tol:
                    break
                # Move alpha towards middle or low end of bracket
                alpha = (alpha_low + alpha_high) / 2.0


            # Prevent alpha from becoming too small
            if alpha < 1e-9:
                 break


        # Final check on the chosen alpha
        phase_final = phase_np + alpha * search_dir_phase
        amp_final = amplitude_np + alpha * search_dir_amp
        cost_final = self._objective_function(simulation, phase_final, amp_final)

        # If the final cost is worse than start, return very small step
        if cost_final > cost_start:
            return 1e-6 # Return small alpha if no improvement found

        return alpha

    def optimize(self, simulation: Simulation) -> CoilConfig:
        start_time = time.time()

        # Initialize parameters
        phase_np = np.random.uniform(0, 2 * np.pi, size=(8,))
        amplitude_np = np.ones((8,)) # Start amplitudes at 1

        best_phase = phase_np.copy()
        best_amplitude = amplitude_np.copy()
        best_cost = self._objective_function(simulation, best_phase, best_amplitude)
        
        if np.isinf(best_cost):
             print("Error: Initial configuration yields invalid cost. Aborting.")
             return CoilConfig(phase=best_phase, amplitude=best_amplitude) # Return initial guess


        # Initial gradient calculation
        g_phase_old, g_amp_old = self._compute_gradient(simulation, phase_np, amplitude_np)
        g_old_combined = np.concatenate((g_phase_old, g_amp_old))

        # Initial search direction D_0 = -G_0 [cite: 57]
        d_phase = -g_phase_old
        d_amp = -g_amp_old
        d_combined = -g_old_combined

        pbar = trange(self.max_iter)
        for i in pbar:
            if time.time() - start_time > self.timeout:
                print(f"\nOptimization stopped due to timeout ({self.timeout}s)")
                break

            # --- Line Search for alpha --- [cite: 58, 59, 60]
            alpha = self._line_search(simulation, phase_np, amplitude_np, d_phase, d_amp, start_time)
            
            if alpha <= 0: # If line search fails or times out
                print(f"\nWarning: Line search failed or returned non-positive alpha at iter {i}. Stopping.")
                break


            # --- Update Parameters --- P_i+1 = P_i + alpha * D_i [cite: 57]
            phase_np += alpha * d_phase
            amplitude_np += alpha * d_amp

            # Apply constraints
            phase_np = np.mod(phase_np, 2 * np.pi)
            amplitude_np = np.clip(amplitude_np, 0, 1)

            # --- Calculate New Gradient ---
            g_phase_new, g_amp_new = self._compute_gradient(simulation, phase_np, amplitude_np)
            g_new_combined = np.concatenate((g_phase_new, g_amp_new))

            # Check for convergence based on gradient norm
            grad_norm = np.linalg.norm(g_new_combined)
            if grad_norm < self.gtol:
                print(f"\nConverged at iteration {i} with gradient norm {grad_norm:.2e}")
                break

            # --- Calculate Beta (Polak-Ribière is often preferred over Fletcher-Reeves) ---
            # beta = ||g_new||^2 / ||g_old||^2  (Fletcher-Reeves)
            # beta = dot(g_new, g_new - g_old) / ||g_old||^2 (Polak-Ribiere) - safer
            # beta = |G_i+1| / |G_i| (From paper [cite: 57] - uses norm ratio directly)

            g_old_sq_norm = np.dot(g_old_combined, g_old_combined)
            if g_old_sq_norm < 1e-12: # Avoid division by zero if gradient was already tiny
                beta = 0.0
            else:
                 # Using Polak-Ribiere variant for potentially better stability
                 beta = np.dot(g_new_combined, g_new_combined - g_old_combined) / g_old_sq_norm
                 beta = max(0, beta) # Reset beta if negative (common practice)


            # --- Update Search Direction --- D_i+1 = -G_i+1 + beta * D_i [cite: 57]
            d_combined = -g_new_combined + beta * d_combined
            d_phase = d_combined[:8]
            d_amp = d_combined[8:]

            # Update old gradient for next iteration
            g_old_combined = g_new_combined.copy()

            # --- Update Best Found Configuration ---
            current_cost = self._objective_function(simulation, phase_np, amplitude_np)
            if current_cost < best_cost:
                 best_cost = current_cost
                 best_phase = phase_np.copy()
                 best_amplitude = amplitude_np.copy()
                 
                 # Display cost considering original direction (min/max)
                 display_cost = -best_cost if self.direction == "maximize" else best_cost
                 pbar.set_postfix_str(f"Best cost {display_cost:.4f}, Grad norm {grad_norm:.2e}")


        # Return the best configuration found
        final_display_cost = -best_cost if self.direction == "maximize" else best_cost
        print(f"\nOptimization finished. Best cost found: {final_display_cost:.4f}")
        
        # Final check for NaNs in results
        if np.isnan(best_phase).any() or np.isnan(best_amplitude).any():
             print("Warning: Best parameters contain NaN. Returning last valid parameters or initial guess if none valid.")
             # Fallback logic needed here - returning initial guess for now
             return CoilConfig(phase=np.random.uniform(0, 2 * np.pi, size=(8,)), amplitude=np.ones((8,)))


        return CoilConfig(phase=best_phase, amplitude=best_amplitude)