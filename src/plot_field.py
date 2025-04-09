import h5py
import numpy as np
import matplotlib
# Set the backend to 'Agg' which is more stable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from src.data.utils import B1Calculator
from src.data.dataclasses import SimulationData, CoilConfig

# Get the absolute path to the workspace root
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to the H5 file
h5_file_path = os.path.join(workspace_root, "src", "data", "simulations", "children_0_tubes_2_id_19969.h5")

def convert_to_complex(field_data):
    """
    Convert structured array with 're' and 'im' components to complex array
    """
    return field_data['re'] + 1j * field_data['im']

def plot_b1_slice(b1_field, slice_idx=3, title="B1+ Field", save_path=None):
    """
    Plot a 2D slice of the B1+ field magnitude
    b1_field: The B1+ field array (3D)
    slice_idx: which z-slice to plot
    save_path: if provided, save the plot to this path
    """
    try:
        # Calculate magnitude of B1+ field
        magnitude = np.abs(b1_field)
        
        # Check dimensions
        print(f"B1+ field shape: {magnitude.shape}")
        
        # Ensure slice_idx is within bounds
        if slice_idx >= magnitude.shape[2]:
            slice_idx = magnitude.shape[2] // 2
            print(f"Adjusted slice_idx to {slice_idx}")
        
        # Plot the slice
        plt.figure(figsize=(10, 8))
        plt.imshow(magnitude[:, :, slice_idx], cmap='viridis')
        plt.colorbar(label='B1+ Field Magnitude')
        plt.title(f'{title} - Slice {slice_idx}')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
        
    except Exception as e:
        print(f"Error in plot_b1_slice: {str(e)}")

def plot_b1_3d(b1_field, threshold=0.1, save_path=None):
    """
    Create a 3D visualization of the B1+ field magnitude
    b1_field: The B1+ field array (3D)
    threshold: minimum magnitude to plot (to reduce noise)
    save_path: if provided, save the plot to this path
    """
    try:
        # Calculate magnitude of B1+ field
        magnitude = np.abs(b1_field)
        
        # Check dimensions
        print(f"B1+ field shape: {magnitude.shape}")
        
        # Create 3D grid with correct dimensions
        x, y, z = np.meshgrid(np.arange(magnitude.shape[0]),
                             np.arange(magnitude.shape[1]),
                             np.arange(magnitude.shape[2]),
                             indexing='ij')  # Use 'ij' indexing to match array dimensions
        
        # Plot points where magnitude exceeds threshold
        mask = magnitude > threshold
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot only a subset of points to avoid memory issues
        stride = 2  # Plot every other point
        scatter = ax.scatter(x[::stride, ::stride, ::stride][mask[::stride, ::stride, ::stride]],
                           y[::stride, ::stride, ::stride][mask[::stride, ::stride, ::stride]],
                           z[::stride, ::stride, ::stride][mask[::stride, ::stride, ::stride]],
                           c=magnitude[::stride, ::stride, ::stride][mask[::stride, ::stride, ::stride]],
                           cmap='viridis', alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D B1+ Field Magnitude')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
        
    except Exception as e:
        print(f"Error in plot_b1_3d: {str(e)}")

# Create output directory if it doesn't exist
output_dir = os.path.join(workspace_root, "figures")
os.makedirs(output_dir, exist_ok=True)

# Open the H5 file and plot
try:
    with h5py.File(h5_file_path, 'r') as f:
        # Convert field data to complex numbers
        efield_complex = convert_to_complex(f['efield'][:])
        hfield_complex = convert_to_complex(f['hfield'][:])
        
        # Print field shapes for debugging
        print(f"E-field shape: {efield_complex.shape}")
        print(f"H-field shape: {hfield_complex.shape}")
        
        # Create default coil configuration
        coil_config = CoilConfig()
        
        # Create simulation data object with all required parameters
        simulation_data = SimulationData(
            simulation_name="children_0_tubes_2_id_19969",
            field=[efield_complex, hfield_complex],  # [E-field, H-field] as complex numbers
            subject=f['subject'][:],
            properties=np.zeros((3, 121, 111, 126)),  # Placeholder for properties
            coil_config=coil_config
        )
        
        # Calculate B1+ field
        b1_calculator = B1Calculator()
        b1_field = b1_calculator(simulation_data)
        
        # Print B1+ field shape for debugging
        print(f"B1+ field shape: {b1_field.shape}")
        
        # Plot 2D slice
        plot_b1_slice(b1_field, slice_idx=3, 
                     title="B1+ Field Magnitude",
                     save_path=os.path.join(output_dir, "b1_slice.png"))
        
        # Plot 3D visualization
        plot_b1_3d(b1_field, threshold=0.1,
                  save_path=os.path.join(output_dir, "b1_3d.png"))
        
except Exception as e:
    print(f"Error processing H5 file: {str(e)}") 