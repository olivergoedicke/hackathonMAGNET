from .data import CoilConfig, Simulation
from .costs.base import BaseCost

from typing import Dict, Any

def evaluate_coil_config(coil_config: CoilConfig,
                         simulation: Simulation,
                         cost_function: BaseCost) -> Dict[str, Any]:
    """
    Evaluates the coil configuration using the cost function.

    Args:
        coil_config: Coil configuration to evaluate.
        simulation: Simulation object.
        cost_function: Cost function object.

    Returns:
        A dictionary containing the best coil configuration, cost, and cost improvement,
        with numeric types converted for JSON serialization.
    """
    default_coil_config = CoilConfig()

    # It's more efficient to calculate simulation data once if needed multiple times
    simulation_data: SimulationData = simulation(coil_config)
    simulation_data_default: SimulationData = simulation(default_coil_config)

    # Calculate cost for both configurations
    default_coil_config_cost = cost_function(simulation_data_default)
    best_coil_config_cost = cost_function(simulation_data)

    # Calculate cost improvement
    # Explicitly cast potential numpy floats to standard Python floats
    cost_improvement_absolute = float(default_coil_config_cost - best_coil_config_cost)

    # Handle potential division by zero if default cost is zero
    if default_coil_config_cost != 0:
        cost_improvement_relative = float((best_coil_config_cost - default_coil_config_cost) / default_coil_config_cost)
    else:
        cost_improvement_relative = float('inf') if best_coil_config_cost > 0 else float('-inf') if best_coil_config_cost < 0 else 0.0


    # Create a dictionary to store the results, converting numpy types
    result = {
        # Convert numpy arrays to lists
        "best_coil_phase": list(coil_config.phase),
        "best_coil_amplitude": list(coil_config.amplitude),
        # Explicitly convert cost values to standard Python floats
        "best_coil_config_cost": float(best_coil_config_cost),
        "default_coil_config_cost": float(default_coil_config_cost),
        "cost_improvement_absolute": cost_improvement_absolute, # Already converted
        "cost_improvement_relative": cost_improvement_relative, # Already converted
        "cost_function_name": cost_function.__class__.__name__, # String
        "cost_function_direction": cost_function.direction, # String
        "simulation_data": simulation_data.simulation_name, # String
    }
    return result