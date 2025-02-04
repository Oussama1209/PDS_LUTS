import cv2
import numpy as np
import torchvision.transforms.functional as TF
import torch

def get_homogeneous_corners(width, height):
    """
    Calculate image corners in homogeneous coordinates.
    Returns a 3x4 numpy array.
    """
    return np.array([[0, 0, 1],
                     [width, 0, 1],
                     [width, height, 1],
                     [0, height, 1]]).T

def warp_perspective_padded(src, dst, transf):
    """
    Warp the source image (src) into the destination image (dst) space
    using a homography matrix (transf) and add padding so that no parts are lost.

    Returns:
        dst_pad: The padded destination image.
        warped: The warped source image.
        anchorX, anchorY: The translation offsets.
        sign: Boolean flag (True if any offset is positive).
    """
    src_h, src_w = src.shape[:2]
    dst_h, dst_w = dst.shape[:2]

    # Define corners for src and dst images.
    src_corners = np.array([[0, 0],
                            [src_w, 0],
                            [src_w, src_h],
                            [0, src_h]], dtype=np.float32)
    src_corners_transformed = cv2.perspectiveTransform(src_corners[None, :, :], transf)[0]

    dst_corners = np.array([[0, 0],
                            [dst_w, 0],
                            [dst_w, dst_h],
                            [0, dst_h]], dtype=np.float32)

    # Combine corners and compute bounding box.
    all_corners = np.vstack((src_corners_transformed, dst_corners))
    x_min, y_min = np.int32(all_corners.min(axis=0))
    x_max, y_max = np.int32(all_corners.max(axis=0))
    shift_x, shift_y = -x_min, -y_min
    output_width, output_height = x_max - x_min, y_max - y_min

    # Build the translation matrix.
    translation_matrix = np.array([[1, 0, shift_x],
                                   [0, 1, shift_y],
                                   [0, 0, 1]], dtype=np.float32)
    new_transf = translation_matrix @ transf

    # Warp the source image and the destination image.
    warped = cv2.warpPerspective(src, new_transf, (output_width, output_height))
    dst_pad = cv2.warpAffine(dst, translation_matrix[:2], (output_width, output_height))
    anchorX, anchorY = int(shift_x), int(shift_y)
    sign = (anchorX > 0) or (anchorY > 0)
    return dst_pad, warped, anchorX, anchorY, sign

def rotate_image(image, angle):
    """
    Rotate a PyTorch image tensor by the specified angle.
    """
    return TF.rotate(image, angle)

def rotate_image1(image, angle):
    """
    Rotate a PyTorch image tensor without cropping and add padding.
    The tensor remains on its original device.
    """
    device = image.device
    if isinstance(image, torch.Tensor):
        # Convert from CHW to HWC and move to CPU.
        image_np = image.permute(1, 2, 0).cpu().numpy()
        if image_np.max() <= 1:
            image_np = (image_np * 255).astype(np.uint8)
    else:
        raise TypeError("Input must be a PyTorch tensor in CHW format.")

    h, w = image_np.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated_image_np = cv2.warpAffine(
        image_np, rotation_matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    rotated_image_tensor = torch.from_numpy(rotated_image_np).permute(2, 0, 1).float()
    if rotated_image_tensor.max() > 1:
        rotated_image_tensor /= 255.0
    return rotated_image_tensor.to(device)

def split_image_paths(image_paths):
    """
    Split a list of image paths into two halves:
      - The first half (with one extra element if the total count is odd)
      - The second half reversed.
    """
    mid = (len(image_paths) + 1) // 2
    first_half = image_paths[:mid]
    second_half = image_paths[mid:][::-1]
    return first_half, second_half
