import h5py
import numpy as np
import os

# Get the absolute path to the workspace root
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to the existing H5 file
h5_file_path = os.path.join(workspace_root, "src", "data", "simulations", "children_0_tubes_2_id_19969.h5")

# --- Reading from the H5 file ---
with h5py.File(h5_file_path, 'r') as f:
    print("\nDetailed Dataset Information:")
    
    # Function to print dataset details
    def print_dataset_details(name, dataset):
        print(f"\nDataset: {name}")
        print(f"Shape: {dataset.shape}")
        print(f"Dtype: {dataset.dtype}")
        
        # Print dimension information
        if name == 'efield' or name == 'hfield':
            print("Dimensions: [components (x,y,z), x-dim, y-dim, z-dim, frequency points]")
            print("Data structure: Complex numbers stored as (real, imaginary) pairs")
            # Print sample values from first component, first frequency
            sample = dataset[0, 0, 0, 0, 0]
            print(f"Sample value (first point): real={sample['re']}, imaginary={sample['im']}")
        
        elif name == 'input':
            print("Dimensions: [components (x,y,z), x-dim, y-dim, z-dim]")
            print("Data type: Real-valued float32")
            # Print sample value
            print(f"Sample value (first point): {dataset[0, 0, 0, 0]}")
        
        elif name == 'subject':
            print("Dimensions: [x-dim, y-dim, z-dim, components]")
            print("Data type: Boolean mask")
            # Print sample value
            print(f"Sample value (first point): {dataset[0, 0, 0, 0]}")
        
        # Print memory size
        size_mb = dataset.size * dataset.dtype.itemsize / (1024 * 1024)
        print(f"Approximate memory size: {size_mb:.2f} MB")
    
    # Print details for each dataset
    for name, dataset in f.items():
        print_dataset_details(name, dataset)

print("\nFinished exploring the H5 file structure.")