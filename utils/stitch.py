from lightglue import LightGlue, SuperPoint, viz2d, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, match_pair
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms.functional as TF
import os
import cv2
import pandas as pd  # Import pandas
import glob
import math
from tqdm.notebook import tqdm


# Utility function to calculate image corners in homogeneous coordinates
def get_homogeneous_corners(width, height):
    return np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]]).T

def warp_perspective_padded(src, dst, transf):
    src_h, src_w = src.shape[:2]
    dst_h, dst_w = dst.shape[:2]

    # Define the corners of the src image
    src_corners = np.array([
        [0, 0],
        [src_w, 0],
        [src_w, src_h],
        [0, src_h]
    ], dtype=np.float32)

    # Transform the src corners using the homography matrix (transf)
    src_corners_transformed = cv2.perspectiveTransform(src_corners[None, :, :], transf)[0]

    # Define the corners of the dst image in its own coordinate space
    dst_corners = np.array([
        [0, 0],
        [dst_w, 0],
        [dst_w, dst_h],
        [0, dst_h]
    ], dtype=np.float32)

    # Combine all corners to find the overall bounding box
    all_corners = np.vstack((src_corners_transformed, dst_corners))

    # Compute the bounding box of all corners
    x_min, y_min = np.int32(all_corners.min(axis=0))
    x_max, y_max = np.int32(all_corners.max(axis=0))

    # Calculate the translation needed to shift images to positive coordinates
    shift_x = -x_min
    shift_y = -y_min

    # Compute the size of the output canvas
    output_width = x_max - x_min
    output_height = y_max - y_min

    # Compute the 3x3 translation matrix to shift the images
    translation_matrix = np.array([
        [1, 0, shift_x],
        [0, 1, shift_y],
        [0, 0,      1]
    ], dtype=np.float32)

    # Update the transformation matrix to include the translation
    new_transf = translation_matrix @ transf

    # Warp the src image using the updated transformation matrix
    warped = cv2.warpPerspective(src, new_transf, (output_width, output_height))
    
    # Warp the dst image using only the translation matrix (affine)
    dst_pad = cv2.warpAffine(dst, translation_matrix[:2], (output_width, output_height))

    # Determine the anchor points
    anchorX = int(shift_x)
    anchorY = int(shift_y)

    return dst_pad, warped, anchorX, anchorY

# Rotate image tensor by specified angle
def rotate_image(image, angle):
    return TF.rotate(image, angle)

def rotate_image1(image, angle):
    """
    Rotate a PyTorch image tensor without cropping and add necessary padding to preserve all content.
    Keeps the tensor on its original device (CPU or CUDA).
    Args:
        image: A PyTorch tensor in CHW format with values in [0, 1].
        angle: Rotation angle in degrees (counterclockwise).
    Returns:
        Rotated image as a PyTorch tensor on the same device.
    """
    device = image.device  # Store original device (CPU or CUDA)

    # Convert tensor to NumPy array (HWC format)
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC on CPU
        if image_np.max() <= 1:  # Scale up to [0, 255] if needed
            image_np = (image_np * 255).astype(np.uint8)
    else:
        raise TypeError("Input must be a PyTorch tensor in CHW format.")

    # Get the height and width of the image
    h, w = image_np.shape[:2]
    center = (w // 2, h // 2)

    # Compute the rotation matrix and new dimensions
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Perform rotation with padding
    rotated_image_np = cv.warpAffine(
        image_np, rotation_matrix, (new_w, new_h),
        flags=cv.INTER_CUBIC,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # Convert back to PyTorch tensor (CHW format)
    rotated_image_tensor = torch.from_numpy(rotated_image_np).permute(2, 0, 1).float()
    if rotated_image_tensor.max() > 1:  # Normalize back to [0, 1]
        rotated_image_tensor /= 255.0

    # Move the tensor back to the original device
    return rotated_image_tensor.to(device)

def compute_similarity_score(image1, image2, extractor, matcher):
    """
    Compute a similarity score between two images by comparing feature matches.
    The score is defined as the number of matching keypoints.
    """
    with torch.no_grad():
        feats0 = extractor.extract(image1)
        feats1 = extractor.extract(image2)
        feats0, feats1, matches01 = match_pair(extractor, matcher, image1, image2)
    points0 = feats0['keypoints'][matches01['matches'][..., 0]]
    score = points0.shape[0]
    del feats0, feats1, matches01, points0
    torch.cuda.empty_cache()
    return score

def split_image_paths(image_paths):
    """
    Splits the image_paths list into two halves:
    - First half gets an extra frame if the total number of images is odd.
    - Second half is reversed to start from the last image and go to the middle.

    Args:
        image_paths (list): List of image paths.

    Returns:
        tuple: (first_half, second_half) - two lists of image paths.
    """
    # Compute the midpoint
    mid = (len(image_paths) + 1) // 2

    # Split the list
    first_half = image_paths[:mid]  # First half
    second_half = image_paths[mid:][::-1]  # Second half, reversed

    return first_half, second_half
