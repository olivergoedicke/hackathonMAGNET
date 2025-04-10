from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

from typing import Callable, NamedTuple
import numpy as np
import time

from tqdm import trange


class Particle(NamedTuple):
    """
    Represents a particle in the swarm, containing position (phase and amplitude)
    and velocity information
    """
    phase: np.ndarray
    amplitude: np.ndarray
    phase_velocity: np.ndarray
    amp_velocity: np.ndarray
    best_phase: np.ndarray
    best_amplitude: np.ndarray
    best_cost: float


class PSOOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) optimizer that uses swarm intelligence
    to find optimal coil configurations.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 n_particles: int = 20,
                 max_iter: int = 100,
                 inertia: float = 0.7,
                 cognitive_param: float = 1.5,
                 social_param: float = 1.5,
                 timeout: float = 300, # Overall timeout in seconds
                 num_runs: int = 3) -> None:
        super().__init__(cost_function)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.inertia = inertia  # Particle's trust in its current velocity
        self.cognitive_param = cognitive_param  # Particle's trust in its own past
        self.social_param = social_param  # Particle's trust in the swarm
        self.timeout = timeout
        self.num_runs = num_runs
        
    def _initialize_particles(self) -> list[Particle]:
        """
        Initialize particles with random positions and velocities
        """
        particles = []
        for _ in range(self.n_particles):
            # Random initial position
            phase = np.random.uniform(low=0, high=2*np.pi, size=(8,))
            amplitude = np.random.uniform(low=0, high=1, size=(8,))
            
            # Random initial velocities
            phase_velocity = np.random.uniform(low=-np.pi, high=np.pi, size=(8,))
            amp_velocity = np.random.uniform(low=-0.5, high=0.5, size=(8,))
            
            particles.append(Particle(
                phase=phase,
                amplitude=amplitude,
                phase_velocity=phase_velocity,
                amp_velocity=amp_velocity,
                best_phase=phase.copy(),
                best_amplitude=amplitude.copy(),
                best_cost=-np.inf if self.direction == "maximize" else np.inf
            ))
        
        return particles
        
    def optimize(self, simulation: Simulation) -> CoilConfig:
        overall_start_time = time.time()
        best_coil_config_overall = None
        best_cost_overall = -np.inf if self.direction == "maximize" else np.inf
        cost_is_better = (lambda current, best: current > best) if self.direction == "maximize" else (lambda current, best: current < best)

        print(f"Starting Particle Swarm Optimization with {self.num_runs} runs...")

        overall_timeout_reached = False
        for run_num in range(self.num_runs):
            if overall_timeout_reached:
                break

            print(f"--- PSO Run {run_num + 1}/{self.num_runs} ---")

            # Check overall timeout before starting a new run
            if time.time() - overall_start_time > self.timeout:
                print(f"\nOverall optimization stopped due to timeout ({self.timeout} seconds) before starting run {run_num + 1}.")
                overall_timeout_reached = True
                break

            # Initialize for this run
            particles = self._initialize_particles()
            global_best_phase_run = np.zeros(8)
            global_best_amplitude_run = np.ones(8)
            global_best_cost_run = -np.inf if self.direction == "maximize" else np.inf
            run_timed_out = False

            try:
                # First evaluation to set initial bests
                for i in range(self.n_particles):
                    # Check timeout before simulation call
                    if time.time() - overall_start_time > self.timeout:
                        raise TimeoutError(f"Overall timeout ({self.timeout}s) reached during initial particle evaluation for run {run_num + 1}.")

                    coil_config = CoilConfig(phase=particles[i].phase, amplitude=particles[i].amplitude)
                    current_cost = self.cost_function(simulation(coil_config))

                    # Check timeout after simulation call
                    if time.time() - overall_start_time > self.timeout:
                         raise TimeoutError(f"Overall timeout ({self.timeout}s) reached during initial particle evaluation for run {run_num + 1}.")

                    # Update particle's best based on initial evaluation
                    particles[i] = particles[i]._replace(best_cost=current_cost)

                    # Update this run's global best
                    if cost_is_better(current_cost, global_best_cost_run):
                        global_best_cost_run = current_cost
                        global_best_phase_run = particles[i].phase.copy()
                        global_best_amplitude_run = particles[i].amplitude.copy()
                print(f"  Initial best cost for run {run_num + 1}: {global_best_cost_run:.4f}")

                # Main optimization loop for this run
                pbar = trange(self.max_iter, desc=f"Run {run_num + 1}", leave=False)
                for iter_num in pbar:
                    # Check overall timeout at the start of each iteration
                    if time.time() - overall_start_time > self.timeout:
                        pbar.close()
                        print(f"\nOptimization stopped during run {run_num + 1} iteration {iter_num} due to overall timeout ({self.timeout} seconds)")
                        run_timed_out = True
                        break # Break inner loop (iterations)

                    for i in range(self.n_particles):
                        # Check timeout before simulation call
                        if time.time() - overall_start_time > self.timeout:
                            raise TimeoutError(f"Overall timeout ({self.timeout}s) reached during run {run_num + 1} iteration {iter_num}.")

                        # Create coil config from particle position
                        coil_config = CoilConfig(
                            phase=particles[i].phase,
                            amplitude=particles[i].amplitude
                        )

                        # Evaluate particle's position
                        current_cost = self.cost_function(simulation(coil_config))

                         # Check timeout after simulation call
                        if time.time() - overall_start_time > self.timeout:
                            raise TimeoutError(f"Overall timeout ({self.timeout}s) reached during run {run_num + 1} iteration {iter_num}.")

                        # Update particle's best position
                        if cost_is_better(current_cost, particles[i].best_cost):
                            particles[i] = particles[i]._replace(
                                best_phase=particles[i].phase.copy(),
                                best_amplitude=particles[i].amplitude.copy(),
                                best_cost=current_cost
                            )

                        # Update this run's global best position
                        if cost_is_better(current_cost, global_best_cost_run):
                            global_best_cost_run = current_cost
                            global_best_phase_run = particles[i].phase.copy()
                            global_best_amplitude_run = particles[i].amplitude.copy()
                            pbar.set_postfix_str(f"Best cost {global_best_cost_run:.4f}")

                    # Update particle velocities and positions
                    for i in range(self.n_particles):
                        # Random coefficients
                        r1, r2 = np.random.rand(2)

                        # Update velocities
                        new_phase_velocity = (
                            self.inertia * particles[i].phase_velocity +
                            self.cognitive_param * r1 * (particles[i].best_phase - particles[i].phase) +
                            self.social_param * r2 * (global_best_phase_run - particles[i].phase)
                        )

                        new_amp_velocity = (
                            self.inertia * particles[i].amp_velocity +
                            self.cognitive_param * r1 * (particles[i].best_amplitude - particles[i].amplitude) +
                            self.social_param * r2 * (global_best_amplitude_run - particles[i].amplitude)
                        )

                        # Update positions
                        new_phase = particles[i].phase + new_phase_velocity
                        new_amplitude = particles[i].amplitude + new_amp_velocity

                        # Apply constraints
                        new_phase = np.mod(new_phase, 2*np.pi)
                        new_amplitude = np.clip(new_amplitude, 0, 1)

                        # Update particle
                        particles[i] = particles[i]._replace(
                            phase=new_phase,
                            amplitude=new_amplitude,
                            phase_velocity=new_phase_velocity,
                            amp_velocity=new_amp_velocity
                        )
                # End of inner loop (max_iter or timeout break)
                if not pbar.disable:
                    pbar.close()
                if not run_timed_out:
                     print(f"  Run {run_num + 1} finished after {self.max_iter} iterations. Best cost for this run: {global_best_cost_run:.4f}")
                else:
                    overall_timeout_reached = True # Signal to stop outer loop

            except TimeoutError as e:
                 # Handle timeout from initialization or inner loop
                 print(f"\nTimeout occurred during run {run_num + 1}: {str(e)}")
                 overall_timeout_reached = True # Signal to stop outer loop
                 # Ensure pbar is closed if it was opened
                 if 'pbar' in locals() and hasattr(pbar, 'close') and not pbar.disable:
                     pbar.close()

            # Compare the best result of this run with the overall best
            # Check if global_best_cost_run was ever updated from its initial +/- inf value
            initial_inf_val = -np.inf if self.direction == "maximize" else np.inf
            if global_best_cost_run != initial_inf_val:
                if best_coil_config_overall is None or cost_is_better(global_best_cost_run, best_cost_overall):
                    print(f"  *** New overall best cost found: {global_best_cost_run:.4f} (previous: {best_cost_overall if best_cost_overall != initial_inf_val else 'N/A'}) ***")
                    best_cost_overall = global_best_cost_run
                    # Store the config itself, not just phase/amplitude separately
                    best_coil_config_overall = CoilConfig(phase=global_best_phase_run.copy(), amplitude=global_best_amplitude_run.copy())
            else:
                 print(f"  Run {run_num + 1} did not produce a valid result (likely immediate timeout or error before first evaluation).")


        # After all runs (or timeout)
        print(f"--- Optimization finished. Overall best cost: {best_cost_overall if best_cost_overall != -np.inf and best_cost_overall != np.inf else 'N/A'} ---")
        if best_coil_config_overall is None:
            print("\nWarning: Optimization failed to find any valid configuration across all runs.")
            # Return a default or raise an error? Returning default.
            return CoilConfig(phase=np.zeros((8,)), amplitude=np.zeros((8,)))

        return best_coil_config_overall 