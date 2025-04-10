
import numpy.typing as npt
import numpy as np
import h5py
import os
import einops

from typing import Tuple
from .dataclasses import SimulationRawData, SimulationData, CoilConfig



class Simulation:
    def __init__(self, 
                 path: str,
                 coil_path: str = "data/antenna/antenna.h5"):
        self.path = path
        self.coil_path = coil_path
        
        self.simulation_raw_data = self._load_raw_simulation_data()
        
    def _load_raw_simulation_data(self) -> SimulationRawData:
        # Load raw simulation data from path
        
        def read_field() -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            with h5py.File(self.path) as f:
                re_efield, im_efield = f["efield"]["re"][:], f["efield"]["im"][:]
                re_hfield, im_hfield = f["hfield"]["re"][:], f["hfield"]["im"][:]
                field = np.stack([np.stack([re_efield, im_efield], axis=0), np.stack([re_hfield, im_hfield], axis=0)], axis=0)
            return field

        def read_physical_properties() -> npt.NDArray[np.float32]:
            with h5py.File(self.path) as f:
                physical_properties = f["input"][:]
            return physical_properties
        
        def read_subject_mask() -> npt.NDArray[np.bool_]:
            with h5py.File(self.path) as f:
                subject = f["subject"][:]
            subject = np.max(subject, axis=-1)
            return subject
        
        def read_coil_mask() -> npt.NDArray[np.float32]:
            with h5py.File(self.coil_path) as f:
                coil = f["masks"][:]
            return coil
        
        def read_simulation_name() -> str:
            return os.path.basename(self.path)[:-3]

        simulation_raw_data = SimulationRawData(
            simulation_name=read_simulation_name(),
            properties=read_physical_properties(),
            field=read_field(),
            subject=read_subject_mask(),
            coil=read_coil_mask()
        )
        
        return simulation_raw_data
    
    
    def _shift_field(self,
                     field: npt.NDArray[np.float32],
                     phase: npt.NDArray[np.float32],
                     amplitude: npt.NDArray[np.float32],
                     subject_mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.float32]:
        """
        Shift the field calculating field_shifted = field * amplitude (e ^ (phase * 1j))
        and summing over all coils, optimized for the subject area.

        Args:
            field: The raw field data (hf, reim, comp, x, y, z, coils).
                   Shape: (2, 2, 3, 121, 111, 126, 8)
            phase: Coil phases (coils,). Shape: (8,)
            amplitude: Coil amplitudes (coils,). Shape: (8,)
            subject_mask: Boolean mask indicating the subject voxels (x, y, z).
                          Shape: (121, 111, 126)

        Returns:
            The phase-shifted field summed over coils, with calculations focused on the subject mask.
            Shape: (hf=2, reimout=2, comp=3, x=121, y=111, z=126)
        """
        # Calculate complex coefficients for the shift
        re_phase = np.cos(phase) * amplitude
        im_phase = np.sin(phase) * amplitude
        # Correct coefficients for einsum:
        # Real output = re_field * re_coeff - im_field * im_coeff
        # Imag output = re_field * im_coeff + im_field * re_coeff
        coeffs_real = np.stack((re_phase, im_phase), axis=0)  # (reim=2, coils=8) driving real output
        coeffs_im = np.stack((im_phase, -re_phase), axis=0) # (reim=2, coils=8) driving imag output
        coeffs = np.stack((coeffs_real, coeffs_im), axis=0) # (reimout=2, reim=2, coils=8)

        # Repeat coefficients for hf dimension (E and H fields)
        # Shape: (hf=2, reimout=2, reim=2, coils=8)
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> hf reimout reim coils', hf=field.shape[0])

        # --- Optimization: Mask field before einsum ---
        # Get spatial dimensions shape
        spatial_shape = field.shape[3:-1] # (121, 111, 126)
        n_subject_voxels = np.sum(subject_mask)

        # Flatten spatial dimensions (x, y, z) and select only subject voxels
        # Input field shape: (hf, reim, comp, x, y, z, coils)
        # Rearrange to put spatial dimensions together for flattening: (hf, reim, comp, coils, x, y, z)
        field_permuted = einops.rearrange(field, 'hf reim comp x y z coils -> hf reim comp coils x y z')
        # Flatten spatial dimensions: (hf, reim, comp, coils, n_spatial_voxels)
        field_flat_spatial = einops.rearrange(field_permuted, 'hf reim comp coils x y z -> hf reim comp coils (x y z)')

        # Create boolean mask compatible with flattened field (flattened spatial dimensions)
        flat_mask = subject_mask.flatten() # Shape: (n_spatial_voxels,)

        # Select only the subject voxels from the flattened field
        # Shape becomes: (hf, reim, comp, coils, n_subject_voxels)
        field_subject_voxels_flat = field_flat_spatial[:, :, :, :, flat_mask]

        # Perform einsum only on subject voxels. Need to align dimensions correctly.
        # field_subject: (hf, reim, comp, coils, voxels)
        # coeffs:      (hf, reimout, reim, coils)
        # Output:      (hf, reimout, comp, voxels)
        shifted_subject_voxels = einops.einsum(field_subject_voxels_flat, coeffs,
                                               'hf reim comp coils voxels, hf reimout reim coils -> hf reimout comp voxels')

        # Create an output array of zeros with the desired output spatial shape
        # Shape: (hf, reimout, comp, x, y, z)
        output_shape = (field.shape[0], coeffs.shape[1], field.shape[2]) + spatial_shape
        field_shift_masked = np.zeros(output_shape, dtype=field.dtype)

        # Flatten the output array's spatial dimensions (x, y, z) for easier indexing
        # Shape: (hf, reimout, comp, n_spatial_voxels)
        field_shift_masked_flat = einops.rearrange(field_shift_masked, 'hf reimout comp x y z -> hf reimout comp (x y z)')

        # Place the calculated shifted values back into the correct positions using the flat mask
        field_shift_masked_flat[:, :, :, flat_mask] = shifted_subject_voxels

        # Reshape back to the original spatial dimensions (comp, x, y, z)
        # Shape: (hf, reimout, comp, x, y, z)
        field_shift_final = einops.rearrange(field_shift_masked_flat, 'hf reimout comp (x y z) -> hf reimout comp x y z',
                                            x=spatial_shape[0], y=spatial_shape[1], z=spatial_shape[2])

        return field_shift_final

    def phase_shift(self, coil_config: CoilConfig) -> SimulationData:
        # Pass the subject mask to _shift_field
        # Ensure self.simulation_raw_data.field has shape (2, 2, 3, 121, 111, 126, 8)
        # Ensure self.simulation_raw_data.subject has shape (121, 111, 126)
        field_shifted = self._shift_field(self.simulation_raw_data.field,
                                          coil_config.phase,
                                          coil_config.amplitude,
                                          self.simulation_raw_data.subject) # Pass the mask

        simulation_data = SimulationData(
            simulation_name=self.simulation_raw_data.simulation_name,
            properties=self.simulation_raw_data.properties,
            field=field_shifted, # Use the optimized shifted field
            subject=self.simulation_raw_data.subject,
            coil_config=coil_config
        )
        return simulation_data
    
    def __call__(self, coil_config: CoilConfig) -> SimulationData:
        return self.phase_shift(coil_config)