import os
import subprocess
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import ants


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
        if 'imaginary' in file and file.endswith(('.nii', '.nii.gz')):
            imaginary_file = os.path.join(output_folder, file)
        elif 'real' in file and file.endswith(('.nii', '.nii.gz')):
            real_file = os.path.join(output_folder, file)
        
    
    if not real_file or not imaginary_file:
        raise ValueError("Real or imaginary NIfTI files not found.")
    
    real_data, real_affine = load_nifti_file(real_file)
    imaginary_data, imaginary_affine = load_nifti_file(imaginary_file)
    
    if not np.array_equal(real_affine, imaginary_affine):
        raise ValueError("Real and imaginary images have different affine matrices.")
    
    magnitude_data = compute_magnitude(real_data, imaginary_data)
    magnitude_file = os.path.join(output_folder, 'magnitude.nii')
    save_nifti_file(magnitude_data, real_affine, magnitude_file)

def process_nii_to_3d_array(nii_path, target_shape, return_original_shape=False):
    """Process NIfTI files to preprocessed 3D array."""
    # magnitude_file = os.path.join(nii_path, 'magnitude.nii')
    if not os.path.exists(nii_path):
        raise ValueError(f"Magnitude NIfTI file not found in {nii_path}.")
    
    magnitude_data, original_affine = load_nifti_file(nii_path)
    original_shape = magnitude_data.shape
    preprocessed_image, new_affine, resized_shape = preprocess_image(magnitude_data, target_shape, original_affine)
    
    if return_original_shape:
        return preprocessed_image, original_shape, resized_shape, new_affine
    return preprocessed_image


def skull_stripping(nii_file, output_file):
    """Perform skull stripping using FSL's bet2."""
    command = f'bet2 {nii_file} {output_file} -m'
    subprocess.run(command, shell=True, check=True)
    print(f"Skull stripping completed for {nii_file}. Output saved to {output_file}.")
    
    
def register_image(fixed_image_path, moving_image_path, output_path):
    """Perform image registration using ANTs."""
    fixed_img = ants.image_read(fixed_image_path)
    moving_img = ants.image_read(moving_image_path)
    
    outs = ants.registration(fixed_img, moving_img, type_of_transform='Affine')
    registered_img = outs['warpedmovout']
    
    ants.image_write(registered_img, output_path)
    print(f"Registration completed for {moving_image_path}. Output saved to {output_path}.")
    
    
def find_bounding_box(image_data):
    # 找到各轴上非零体素的最小和最大索引
    coords = np.argwhere(image_data)
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)
    
    # 返回bounding box的起始和结束坐标
    return (x_min, x_max), (y_min, y_max), (z_min, z_max)

def crop_image(image_data, bbox):
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bbox
    cropped_image = image_data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    return cropped_image

def get_max_bounding_box(bbox_list):
    # 将bbox_list中的每个bbox分别解包成6个独立的值
    x_mins = [bbox[0][0] for bbox in bbox_list]
    x_maxs = [bbox[0][1] for bbox in bbox_list]
    y_mins = [bbox[1][0] for bbox in bbox_list]
    y_maxs = [bbox[1][1] for bbox in bbox_list]
    z_mins = [bbox[2][0] for bbox in bbox_list]
    z_maxs = [bbox[2][1] for bbox in bbox_list]
    
    max_bbox = (
        (min(x_mins), max(x_maxs)),
        (min(y_mins), max(y_maxs)),
        (min(z_mins), max(z_maxs))
    )
    return max_bbox


