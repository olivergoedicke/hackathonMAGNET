import torch
import torch.optim as optim
import numpy as np
import time
from tqdm import trange
from typing import Optional, Tuple

# Assuming these imports work and are compatible/adaptable for PyTorch tensors
# You might need to adjust these relative imports based on your exact structure
try:
    from ..data.simulation import Simulation, CoilConfig # Adjust path if needed
    from ..costs.base import BaseCost                 # Adjust path if needed
    from .base import BaseOptimizer                   # Adjust path if needed
except ImportError:
    # Fallback for potential execution context issues, adjust as necessary
    from src.data.simulation import Simulation, CoilConfig
    from src.costs.base import BaseCost
    from src.optimizers.base import BaseOptimizer


class AdamOptimizerTorch(BaseOptimizer):
    """
    AdamOptimizerTorch uses PyTorch's Adam optimizer with multiple initializations.

    It first runs several random initializations, evaluates their initial cost,
    and then starts the Adam optimization process from the best-performing
    initial configuration.

    **Crucial Assumption:** This implementation assumes that the `simulation` object's
    call method (`simulation(...)`) and the `cost_function` object's call method
    can handle PyTorch tensors for `phase` and `amplitude` within a `CoilConfig`
    (or be adapted to work with them directly) and are differentiable using
    PyTorch's autograd mechanism. The `cost_function` must return a scalar tensor loss.
    If this is not the case, you would need to compute gradients numerically
    (like in GradientDescentOptimizer) and manually apply the Adam update rule,
    or adapt your simulation/cost function code.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 100,
                 learning_rate: float = 0.001, # Default Adam LR is often smaller
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 timeout: float = 300, # 5 minutes
                 num_initializations: int = 10,
                 device: str = 'cpu') -> None:
        """
        Initializes the AdamOptimizerTorch.

        Args:
            cost_function: The cost function to optimize. Must have a 'direction'
                           attribute ('minimize' or 'maximize') and be callable.
            max_iter: Maximum number of optimization iterations.
            learning_rate: Learning rate for the Adam optimizer.
            betas: Coefficients used for computing running averages of gradient
                   and its square in Adam.
            eps: Term added to the denominator to improve numerical stability in Adam.
            timeout: Maximum time allowed for the entire optimization process (seconds).
            num_initializations: Number of random configurations to try before
                                 starting the main optimization.
            device: The torch device to run computations on ('cpu' or 'cuda').
        """
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.timeout = timeout
        self.num_initializations = num_initializations
        self.device = torch.device(device)

        cost_direction = getattr(self.cost_function, 'direction', 'minimize')
        if not isinstance(cost_direction, str) or cost_direction not in ['minimize', 'maximize']:
             raise ValueError("Cost function must have a 'direction' attribute ('minimize' or 'maximize').")
        self.direction = cost_direction
        print(f"AdamOptimizerTorch initialized. Optimizing to {self.direction} cost.")
        print(f"Device: {self.device}")


    def optimize(self, simulation: Simulation) -> Optional[CoilConfig]:
        """
        Optimizes the coil configuration using Adam after evaluating multiple initializations.

        Args:
            simulation: The simulation environment. Must have a `coil_system.num_coils`
                        attribute and its call method must be compatible with PyTorch tensors.

        Returns:
            The best CoilConfig found (with NumPy arrays), or None if optimization
            failed, timed out before finding a valid start, or encountered errors.
        """
        start_time = time.time()
        best_initial_config_tensors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        best_initial_cost: float = -np.inf if self.direction == "maximize" else np.inf

        try:
            num_coils = simulation.coil_system.num_coils # Get number of coils
        except AttributeError:
             print("Error: Simulation object must have 'coil_system.num_coils' attribute.")
             return None

        # --- Initialization Phase ---
        print(f"Running {self.num_initializations} random initializations...")
        init_pbar = trange(self.num_initializations, desc="Initializations", leave=False)
        initialization_timeout = False
        successful_inits = 0
        for _ in init_pbar:
            if time.time() - start_time > self.timeout:
                init_pbar.close()
                print("\nTimeout occurred during initialization phase.")
                initialization_timeout = True
                break

            # Initialize parameters as PyTorch tensors
            # Use torch.pi which is more standard than np.pi in torch contexts
            phase_init = torch.rand(num_coils, device=self.device, dtype=torch.float32) * 2 * torch.pi
            amp_init = torch.rand(num_coils, device=self.device, dtype=torch.float32)

            # --- Evaluate Initial Cost ---
            # This requires simulation & cost_function to work even without gradients.
            # We use no_grad context as gradients are not needed for selection.
            try:
                 with torch.no_grad():
                     # Pass tensors directly if CoilConfig/Simulation supports it.
                     # Otherwise, conversion might be needed (less ideal).
                     # Assuming direct tensor support here for clarity:
                     temp_config = CoilConfig(phase=phase_init, amplitude=amp_init)
                     sim_data = simulation(temp_config) # Must return structure cost_fn expects
                     initial_cost_tensor = self.cost_function(sim_data) # Must return scalar tensor

                     if not isinstance(initial_cost_tensor, torch.Tensor) or not initial_cost_tensor.ndim == 0:
                         print(f"\nWarning: Cost function did not return a scalar tensor for initialization. Skipping.")
                         continue

                     initial_cost = initial_cost_tensor.item() # Get float value
                     successful_inits += 1

            except Exception as e:
                 print(f"\nError during cost evaluation for an initialization: {e}. Skipping.")
                 # Consider logging the full traceback for debugging
                 # import traceback; traceback.print_exc()
                 continue # Skip this initialization

            # Update best initial if this initialization is better
            if ((self.direction == "maximize" and initial_cost > best_initial_cost) or
                (self.direction == "minimize" and initial_cost < best_initial_cost)):
                best_initial_cost = initial_cost
                # Store clones of the tensors that gave the best initial cost
                best_initial_config_tensors = (phase_init.detach().clone(), amp_init.detach().clone())
                init_pbar.set_postfix_str(f"Best initial cost {best_initial_cost:.4f}")
        # --- End Initialization Loop ---
        init_pbar.close() # Ensure pbar is closed

        if best_initial_config_tensors is None:
            if not initialization_timeout:
                print(f"\nNo successful initializations completed out of {self.num_initializations} attempts.")
            # If timeout happened or no init worked, return None
            return None

        print(f"\nFound best initial cost: {best_initial_cost:.4f} after {successful_inits} successful evaluations.")
        print(f"Starting optimization from this configuration...")

        # --- Optimization Phase ---
        # Start optimization from the best initial configuration tensors
        phase = best_initial_config_tensors[0].clone().requires_grad_(True)
        amplitude = best_initial_config_tensors[1].clone().requires_grad_(True)

        # Setup Adam optimizer
        optimizer = optim.Adam([phase, amplitude], lr=self.learning_rate, betas=self.betas, eps=self.eps)

        best_cost_opt: float = best_initial_cost # Tracks best cost found during optimization steps
        # Keep track of the best *tensors* found during optimization
        best_phase_tensor = phase.detach().clone()
        best_amplitude_tensor = amplitude.detach().clone()

        optimization_timeout = False
        try:
            pbar = trange(self.max_iter, desc="Optimization")
            for i in pbar:
                if time.time() - start_time > self.timeout:
                    pbar.close()
                    print("\nOptimization stopped due to timeout.")
                    optimization_timeout = True
                    break

                optimizer.zero_grad()

                # --- Forward pass: Requires simulation & cost_function Autograd compatibility ---
                # Create a config with the current tensors needing gradients
                current_config = CoilConfig(phase=phase, amplitude=amplitude)
                sim_data = simulation(current_config) # Must handle tensors & build graph
                cost = self.cost_function(sim_data)   # Must return scalar tensor & build graph

                if not isinstance(cost, torch.Tensor) or not cost.ndim == 0:
                    pbar.close()
                    print(f"\nError: Cost function did not return a scalar tensor during optimization step {i}. Stopping.")
                    # Return the best configuration found *before* the error
                    # Or you could raise an error: raise TypeError(...)
                    break # Exit optimization loop

                # --- Backward pass ---
                # If maximizing, negate the cost because optimizers minimize by default
                loss = -cost if self.direction == "maximize" else cost
                loss.backward() # Compute gradients

                # --- Optimizer step ---
                optimizer.step() # Update parameters (phase, amplitude) based on gradients

                # --- Apply constraints AFTER optimizer step ---
                # Use torch.no_grad() to avoid tracking these constraint operations
                with torch.no_grad():
                    # phase % (2 * torch.pi)
                    phase.data = torch.fmod(phase.data, 2 * torch.pi)
                    # Ensure phase remains positive [0, 2pi)
                    phase.data[phase.data < 0] += 2 * torch.pi
                    # Clamp amplitude [0, 1]
                    amplitude.data = torch.clamp(amplitude.data, 0, 1)

                # --- Evaluate cost with updated tensors (for tracking best) ---
                # Re-evaluate cost with the constrained tensors for accurate tracking.
                # Use no_grad as this evaluation is just for monitoring/selecting the best.
                with torch.no_grad():
                    # Create a temporary config with detached tensors
                    eval_config = CoilConfig(phase=phase.detach().clone(), amplitude=amplitude.detach().clone())
                    current_sim_data = simulation(eval_config)
                    current_cost_tensor = self.cost_function(current_sim_data)
                    current_cost_val = current_cost_tensor.item() # Get float value

                # Update best cost and configuration found during optimization
                if ((self.direction == "maximize" and current_cost_val > best_cost_opt) or
                    (self.direction == "minimize" and current_cost_val < best_cost_opt)):
                    best_cost_opt = current_cost_val
                    best_phase_tensor = phase.detach().clone()
                    best_amplitude_tensor = amplitude.detach().clone()
                    # Postfix shows current cost and best cost found *during* optimization
                    pbar.set_postfix_str(f"Cost {current_cost_val:.4f} | Best {best_cost_opt:.4f} (*)")
                else:
                    pbar.set_postfix_str(f"Cost {current_cost_val:.4f} | Best {best_cost_opt:.4f}")
                # --- End Evaluation ---

            # --- End Optimization Loop ---
            if not pbar.disable: # Ensure pbar is closed if loop finishes naturally or breaks
                pbar.close()

        except Exception as e:
            # Ensure pbar is closed in case of exception within the loop
            if 'pbar' in locals() and not pbar.disable:
                pbar.close()
            print(f"\nAn error occurred during optimization: {e}")
            import traceback # Optional: Print stack trace for debugging
            traceback.print_exc()
            # Fall through to finally block to return the best config found *before* the error

        finally:
            # --- Finalization ---
            elapsed_time = time.time() - start_time
            if optimization_timeout:
                 print(f"\nTimeout reached after {elapsed_time:.2f}s. Returning best configuration found.")
            elif 'pbar' in locals() and not pbar.disable and pbar.n < pbar.total:
                # Loop exited early, not due to timeout (could be error or future convergence check)
                print(f"\nOptimization loop stopped early at iteration {pbar.n} after {elapsed_time:.2f}s.")
            else: # Optimization finished max_iter or completed normally
                 print(f"\nOptimization finished {self.max_iter} iterations in {elapsed_time:.2f}s.")

            print(f"Best cost found during optimization: {best_cost_opt:.4f}")

            # Convert the best tensors back to a CoilConfig with NumPy arrays
            # (Assuming the rest of the system expects NumPy for the final result)
            final_config = CoilConfig(
                phase=best_phase_tensor.cpu().numpy(),
                amplitude=best_amplitude_tensor.cpu().numpy()
            )
            return final_config
