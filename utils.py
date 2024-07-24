import os
import subprocess
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

def convert_dcm_to_nii(dcm_folder, output_folder):
    """Convert DICOM files to NIfTI format if not already converted."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if NIfTI files already exist
    nii_files = [f for f in os.listdir(output_folder) if f.endswith(('.nii', '.nii.gz'))]
    if nii_files:
        print(f"NIfTI files already exist in {output_folder}. Skipping conversion.")
        return
    
    command = f'dcm2niix -o {output_folder} {dcm_folder}'
    subprocess.run(command, shell=True, check=True)

def load_nifti_file(file_path):
    """Load a NIfTI file and return the image data and affine."""
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

def save_nifti_file(data, affine, output_path):
    """Save the image data to a NIfTI file."""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_path)

def compute_magnitude(real_data, imaginary_data):
    """Compute the magnitude from real and imaginary data."""
    magnitude_data = np.sqrt(np.square(real_data) + np.square(imaginary_data))
    return magnitude_data

def preprocess_image(image, target_shape, original_affine):
    """Preprocess the image data: normalize and downsample using PyTorch."""
    # Normalize the image
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Convert the image to a PyTorch tensor and add batch and channel dimensions
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    # Get the original shape
    original_shape = image_tensor.shape[2:]
    
    # Use trilinear interpolation to resize the image
    image_resized = F.interpolate(image_tensor, size=target_shape, mode='trilinear', align_corners=False)
    resized_image = image_resized.squeeze().numpy()
    
    # Update voxel spacing in the affine matrix
    new_affine = original_affine.copy()
    scale_factors = [original_shape[i] / target_shape[i] for i in range(len(original_shape))]
    new_affine[:3, :3] *= np.diag(scale_factors)
    
    # Get the shape of the resized image
    resized_shape = resized_image.shape
    
    return resized_image, new_affine, resized_shape


def process_dicom_to_nii(dcm_folder, output_folder, target_shape):
    """Convert DICOM files to preprocessed NIfTI files."""
    convert_dcm_to_nii(dcm_folder, output_folder)
    
    real_file = None
    imaginary_file = None
    for file in os.listdir(output_folder):
        if 'real' in file and file.endswith(('.nii', '.nii.gz')):
            real_file = os.path.join(output_folder, file)
        elif 'imaginary' in file and file.endswith(('.nii', '.nii.gz')):
            imaginary_file = os.path.join(output_folder, file)
    
    if not real_file or not imaginary_file:
        raise ValueError("Real or imaginary NIfTI files not found.")
    
    real_data, real_affine = load_nifti_file(real_file)
    imaginary_data, imaginary_affine = load_nifti_file(imaginary_file)
    
    if not np.array_equal(real_affine, imaginary_affine):
        raise ValueError("Real and imaginary images have different affine matrices.")
    
    magnitude_data = compute_magnitude(real_data, imaginary_data)
    magnitude_file = os.path.join(output_folder, 'magnitude.nii')
    save_nifti_file(magnitude_data, real_affine, magnitude_file)

def process_nii_to_3d_array(nii_folder, target_shape, return_original_shape=False):
    """Process NIfTI files to preprocessed 3D array."""
    magnitude_file = os.path.join(nii_folder, 'magnitude.nii')
    if not os.path.exists(magnitude_file):
        raise ValueError(f"Magnitude NIfTI file not found in {nii_folder}.")
    
    magnitude_data, original_affine = load_nifti_file(magnitude_file)
    original_shape = magnitude_data.shape
    preprocessed_image, new_affine, resized_shape = preprocess_image(magnitude_data, target_shape, original_affine)
    
    if return_original_shape:
        return preprocessed_image, original_shape, resized_shape, new_affine
    return preprocessed_image
