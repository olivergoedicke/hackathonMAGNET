from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

from typing import Callable, NamedTuple
import numpy as np

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
                 social_param: float = 1.5) -> None:
        super().__init__(cost_function)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.inertia = inertia  # Particle's trust in its current velocity
        self.cognitive_param = cognitive_param  # Particle's trust in its own past
        self.social_param = social_param  # Particle's trust in the swarm
        
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
        
    def optimize(self, simulation: Simulation):
        # Initialize particles
        particles = self._initialize_particles()
        
        # Initialize global best
        global_best_phase = np.zeros(8)
        global_best_amplitude = np.ones(8)
        global_best_cost = -np.inf if self.direction == "maximize" else np.inf
        
        # Main optimization loop
        pbar = trange(self.max_iter)
        for _ in pbar:
            for i in range(self.n_particles):
                # Create coil config from particle position
                coil_config = CoilConfig(
                    phase=particles[i].phase,
                    amplitude=particles[i].amplitude
                )
                
                # Evaluate particle's position
                current_cost = self.cost_function(simulation(coil_config))
                
                # Update particle's best position
                if ((self.direction == "maximize" and current_cost > particles[i].best_cost) or
                    (self.direction == "minimize" and current_cost < particles[i].best_cost)):
                    particles[i] = particles[i]._replace(
                        best_phase=particles[i].phase.copy(),
                        best_amplitude=particles[i].amplitude.copy(),
                        best_cost=current_cost
                    )
                
                # Update global best position
                if ((self.direction == "maximize" and current_cost > global_best_cost) or
                    (self.direction == "minimize" and current_cost < global_best_cost)):
                    global_best_cost = current_cost
                    global_best_phase = particles[i].phase.copy()
                    global_best_amplitude = particles[i].amplitude.copy()
                    pbar.set_postfix_str(f"Best cost {global_best_cost:.2f}")
            
            # Update particle velocities and positions
            for i in range(self.n_particles):
                # Random coefficients
                r1, r2 = np.random.rand(2)
                
                # Update velocities
                new_phase_velocity = (
                    self.inertia * particles[i].phase_velocity +
                    self.cognitive_param * r1 * (particles[i].best_phase - particles[i].phase) +
                    self.social_param * r2 * (global_best_phase - particles[i].phase)
                )
                
                new_amp_velocity = (
                    self.inertia * particles[i].amp_velocity +
                    self.cognitive_param * r1 * (particles[i].best_amplitude - particles[i].amplitude) +
                    self.social_param * r2 * (global_best_amplitude - particles[i].amplitude)
                )
                
                # Update positions
                new_phase = particles[i].phase + new_phase_velocity
                new_amplitude = particles[i].amplitude + new_amp_velocity
                
                # Apply constraints
                new_phase = np.mod(new_phase, 2*np.pi)  # Keep phases in [0, 2π]
                new_amplitude = np.clip(new_amplitude, 0, 1)  # Keep amplitudes in [0, 1]
                
                # Update particle
                particles[i] = particles[i]._replace(
                    phase=new_phase,
                    amplitude=new_amplitude,
                    phase_velocity=new_phase_velocity,
                    amp_velocity=new_amp_velocity
                )
        
        return CoilConfig(phase=global_best_phase, amplitude=global_best_amplitude) 