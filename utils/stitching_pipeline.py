import os
import cv2 as cv
import cv2
import glob
import math
import pandas as pd
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm

from utils.image_utils import warp_perspective_padded, rotate_image
from utils.stitch import *

from lightglue import DoGHardNet, LightGlue
from lightglue.utils import load_image, match_pair
from torchvision.transforms.functional import to_pil_image


def run_panorama_pipeline(image_folder, start_index=0):
    """
    Run the full panorama stitching pipeline.
    
    Parameters:
        general_folder_path (str): Base folder for outputs.
        image_folder (str): Folder containing input images.
        start_index (int): Starting index for the image list.
    
    Returns:
        combined_image: The final stitched panorama (as a cv2 image).
        image_corners_df: DataFrame containing corner and metadata information.
    """
    # Set up device and output paths.
    general_folder_path = 'results/'
    os.makedirs(general_folder_path, exist_ok=True)
    torch.cuda.empty_cache()
    
    # Initialize feature extractor and matcher.
    extractor = DoGHardNet(max_num_keypoints=None).eval().cuda()
    matcher = LightGlue(features='doghardnet').eval().cuda()
    
    # Gather and sort image paths.
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))
    image_paths.sort()
    image_paths = image_paths[start_index:] + image_paths[:start_index]
    
    # Initialize DataFrame for corners and metadata.
    image_corners_df = pd.DataFrame(columns=['image_path', 'corners', 'frame_number', 'rotation_angle'])
    
    # Process the first image.
    image_path0 = image_paths[0]
    img0_cv = cv2.imread(image_path0)
    cv2.imwrite(os.path.join(general_folder_path, 'warped_image.jpg'), img0_cv)
    image0 = load_image(image_path0)
    
    # Assume image0 is in CHW; get height and width.
    h = image0.shape[1]
    w = image0.shape[2]
    first_image_corners = np.array([[0, 0],
                                    [w, 0],
                                    [w, h],
                                    [0, h]], dtype=np.int32)
    frame_number = os.path.splitext(os.path.basename(image_path0))[0]
    new_row = pd.DataFrame({
        'image_path': [image_path0],
        'corners': [np.array(first_image_corners, dtype=np.int32)],
        'frame_number': [frame_number],
        'rotation_angle': [0],
    })
    image_corners_df = pd.concat([image_corners_df, new_row], ignore_index=True)
    
    # Split image paths into two halves.
    first_half, second_half = split_image_paths(image_paths)
    
    ### Process the first half ###
    start = True
    for idx in tqdm(range(1, len(first_half))):
        if not start:
            image_path0  = general_folder_path + 'warped_image.jpg'

        image0 = load_image(image_path0)

        image_path1  = first_half[idx]

        image1 = load_image(image_path1)

        # Determine best rotation angle
        rotation_angles = range(0, 360, 45)
        best_score, best_angle = -1, 0

        for angle in rotation_angles:
            rotated_image1 = rotate_image(image1, angle)
            score = compute_similarity_score(image0, rotated_image1, extractor, matcher)
            if score > best_score:
                best_score, best_angle = score, angle
        best_rotated_image = rotate_image(image1, best_angle)

        # Prepare images for final stitching
        imocv0 = cv.imread(image_path0)
        imocv1 = cv.cvtColor(np.array(TF.to_pil_image(best_rotated_image.cpu()).convert("RGB")), cv.COLOR_RGB2BGR)

        # Extract keypoints and compute homography matrix
        feats0, feats1, matches01 = match_pair(extractor, matcher, image0, best_rotated_image)
        points0 = feats0['keypoints'][matches01['matches'][..., 0]].cpu().numpy()
        points1 = feats1['keypoints'][matches01['matches'][..., 1]].cpu().numpy()

        if points0.shape[0] >= 4:
            M, _ = cv.findHomography(points1, points0, cv.RANSAC, 5.0)
            
            M_normalized = M / M[2, 2]

            # Extract the rotation and translation components
            M = np.array([
                [M_normalized[0, 0], M_normalized[0, 1], M_normalized[0, 2]],
                [M_normalized[1, 0], M_normalized[1, 1], M_normalized[1, 2]],
                [0,                 0,                 1]
            ])

            dst_padded, warped_image, anchorX1, anchorY1 = warp_perspective_padded(imocv1, imocv0, M)

            before_last_key = image_corners_df['image_path'].iloc[-1]
            warped_image_corners = image_corners_df.loc[image_corners_df['image_path'] == before_last_key, 'corners'].values[0]
            x_coords, y_coords = warped_image_corners[:, 0], warped_image_corners[:, 1]

            # Define source corners as a numpy array of four points
            corners = np.array([[x_coords[0], y_coords[0]],
                                [x_coords[1], y_coords[1]],
                                [x_coords[2], y_coords[2]],
                                [x_coords[3], y_coords[3]]], dtype=np.float32)

            b_x_min, b_y_min = np.min(corners, axis=0).astype(int)

            # Define source corners as a numpy array and convert to float32
            new_image_corners = np.array([[0, 0], [imocv1.shape[1], 0],
                                        [imocv1.shape[1], imocv1.shape[0]],
                                        [0, imocv1.shape[0]]], dtype=np.float32)

            # Perform perspective transform with correctly typed data
            transformed_corners = cv.perspectiveTransform(np.array([new_image_corners], dtype=np.float32), M)[0]

            # Update image_corners_df
            adjusted_corners = transformed_corners + [anchorX1, anchorY1]
            new_row = pd.DataFrame({
                'image_path': [image_path1],
                'corners': [np.array(adjusted_corners, dtype=np.int32)],
                'frame_number': [os.path.splitext(os.path.basename(image_path1))[0]],
                'rotation_angle': [best_angle],
            })
            image_corners_df = pd.concat([image_corners_df, new_row], ignore_index=True)

            if start:
                idx0 = image_corners_df[image_corners_df['image_path'] == image_path0].index[0]
                image_corners_df.at[idx0, 'corners'] += [anchorX1, anchorY1]
                (anchorX, anchorY) = (0, 0)

            # Overlay warped image onto padded destination
            non_zero_mask = (warped_image > 0).astype(np.uint8)
            dst_padded[non_zero_mask == 1] = warped_image[non_zero_mask == 1]

            # Get last key
            last_key = image_corners_df['image_path'].iloc[-1]
            warped_image_corners = image_corners_df.loc[image_corners_df['image_path'] == last_key, 'corners'].values[0]
            x_coords, y_coords = warped_image_corners[:, 0], warped_image_corners[:, 1]

            # Define source corners as a numpy array of four points
            corners = np.array([[x_coords[0], y_coords[0]],
                                [x_coords[1], y_coords[1]],
                                [x_coords[2], y_coords[2]],
                                [x_coords[3], y_coords[3]]], dtype=np.float32)

            x_min, y_min = np.min(corners, axis=0).astype(int)
            x_max, y_max = np.max(corners, axis=0).astype(int)

            # Crop the region of interest
            warped_image = warped_image[y_min:y_max, x_min:x_max]
            cv.imwrite(general_folder_path + 'warped_image.jpg', warped_image)

            # Save final images
            cv.imwrite(general_folder_path + 'panorama.jpg', dst_padded)
        else:
            print("Not enough points to compute homography.")

        if start:
            cv.imwrite(general_folder_path+'aligned_image.jpg', cv.imread(general_folder_path+'panorama.jpg'))

        # Load the images
        current_panorama = cv.imread(general_folder_path + 'panorama.jpg')
        new_image = cv.imread(general_folder_path + 'aligned_image.jpg')

        # Create a 3x3 translation matrix
        translation_matrix = np.float32([
            [1, 0, b_x_min - anchorX1],
            [0, 1, b_y_min - anchorY1],
            [0, 0, 1]
        ])

        if start:
            # Create a 3x3 translation matrix Identity matrix
            translation_matrix = np.float32([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])

        # Apply the translation to the images
        dst_padded, warped_image, anchorX, anchorY = warp_perspective_padded(current_panorama, new_image, translation_matrix)
        idx = image_corners_df[image_corners_df['image_path'] == last_key].index[0]
        if not start:
            image_corners_df.at[idx, 'corners'] += [b_x_min - anchorX1 + anchorX, b_y_min -anchorY1 + anchorY]
        for img_path in image_corners_df['image_path']:
            if img_path != image_path1:
                idx = image_corners_df[image_corners_df['image_path'] == img_path].index[0]
                image_corners_df.at[idx, 'corners'] += [anchorX, anchorY]
        start = False

        # Create a mask where dst_padded has zero pixels
        mask_dst = cv.cvtColor(dst_padded, cv.COLOR_BGR2GRAY)
        mask_dst = (mask_dst == 0).astype(np.uint8)

        # Ensure mask has three channels
        mask_dst_3ch = cv.merge([mask_dst, mask_dst, mask_dst])

        # Combine images by filling zeros in dst_padded with pixels from warped_image
        combined_image = dst_padded.copy()
        combined_image[mask_dst_3ch == 1] = warped_image[mask_dst_3ch == 1]

        # Save the combined image
        cv.imwrite(general_folder_path + 'aligned_image.jpg', combined_image)
    
    ### Process the second half ###
    start = True

    for idx in tqdm(range(1, len(second_half))):
        # image_path0 = general_folder_path + 'anchor_test.jpg'
        if not start:
            image_path0  = general_folder_path + 'warped_image.jpg'
        else:
            image_path0  = image_paths[0]

        image0 = load_image(image_path0)
        image_path1  = second_half[idx]
        image1 = load_image(image_path1)

        # Determine best rotation angle
        rotation_angles = range(0, 360, 45)
        best_score, best_angle = -1, 0

        for angle in rotation_angles:
            rotated_image1 = rotate_image(image1, angle)
            score = compute_similarity_score(image0, rotated_image1, extractor, matcher)
            if score > best_score:
                best_score, best_angle = score, angle
                best_rotated_image = rotated_image1

        # Prepare images for final stitching
        imocv0 = cv.imread(image_path0)
        imocv1 = cv.cvtColor(np.array(TF.to_pil_image(best_rotated_image.cpu()).convert("RGB")), cv.COLOR_RGB2BGR)

        # Extract keypoints and compute homography matrix
        feats0, feats1, matches01 = match_pair(extractor, matcher, image0, best_rotated_image)
        points0 = feats0['keypoints'][matches01['matches'][..., 0]].cpu().numpy()
        points1 = feats1['keypoints'][matches01['matches'][..., 1]].cpu().numpy()

        if points0.shape[0] >= 4:
            M, _ = cv.findHomography(points1, points0, cv.RANSAC, 5.0)
            
            M_normalized = M / M[2, 2]

            # Extract the rotation and translation components
            M = np.array([
                [M_normalized[0, 0], M_normalized[0, 1], M_normalized[0, 2]],
                [M_normalized[1, 0], M_normalized[1, 1], M_normalized[1, 2]],
                [0,                 0,                 1]
            ])

            dst_padded, warped_image, anchorX1, anchorY1 = warp_perspective_padded(imocv1, imocv0, M)

            if start:
                before_last_key = image_corners_df['image_path'].iloc[0]
                warped_image_corners = image_corners_df.loc[image_corners_df['image_path'] == before_last_key, 'corners'].values[0]
                x_coords, y_coords = warped_image_corners[:, 0], warped_image_corners[:, 1]
            else : 
                before_last_key = image_corners_df['image_path'].iloc[-1]
                warped_image_corners = image_corners_df.loc[image_corners_df['image_path'] == before_last_key, 'corners'].values[0]
                x_coords, y_coords = warped_image_corners[:, 0], warped_image_corners[:, 1]

            # Define source corners as a numpy array of four points
            corners = np.array([[x_coords[0], y_coords[0]],
                                [x_coords[1], y_coords[1]],
                                [x_coords[2], y_coords[2]],
                                [x_coords[3], y_coords[3]]], dtype=np.float32)

            b_x_min, b_y_min = np.min(corners, axis=0).astype(int)

            # Define source corners as a numpy array and convert to float32
            new_image_corners = np.array([[0, 0], [imocv1.shape[1], 0],
                                        [imocv1.shape[1], imocv1.shape[0]],
                                        [0, imocv1.shape[0]]], dtype=np.float32)

            # Perform perspective transform with correctly typed data
            transformed_corners = cv.perspectiveTransform(np.array([new_image_corners], dtype=np.float32), M)[0]

            # Update image_corners_df
            adjusted_corners = transformed_corners + [anchorX1, anchorY1]
            new_row = pd.DataFrame({
                'image_path': [image_path1],
                'corners': [np.array(adjusted_corners, dtype=np.int32)],
                'frame_number': [os.path.splitext(os.path.basename(image_path1))[0]],
                'rotation_angle': [best_angle],
            })
            image_corners_df = pd.concat([image_corners_df, new_row], ignore_index=True)
            
            # Overlay warped image onto padded destination
            non_zero_mask = (warped_image > 0).astype(np.uint8)
            dst_padded[non_zero_mask == 1] = warped_image[non_zero_mask == 1]

            # Get last and before last keys
            last_key = image_corners_df['image_path'].iloc[-1]
            warped_image_corners = image_corners_df.loc[image_corners_df['image_path'] == last_key, 'corners'].values[0]
            x_coords, y_coords = warped_image_corners[:, 0], warped_image_corners[:, 1]

            # Define source corners as a numpy array of four points
            corners = np.array([[x_coords[0], y_coords[0]],
                                [x_coords[1], y_coords[1]],
                                [x_coords[2], y_coords[2]],
                                [x_coords[3], y_coords[3]]], dtype=np.float32)

            x_min, y_min = np.min(corners, axis=0).astype(int)
            x_max, y_max = np.max(corners, axis=0).astype(int)

            # Crop the region of interest
            warped_image = warped_image[y_min:y_max, x_min:x_max]
            cv.imwrite(general_folder_path + 'warped_image.jpg', warped_image)

            # Save final images
            cv.imwrite(general_folder_path + 'panorama.jpg', dst_padded)
        else:
            print("Not enough points to compute homography.")

        # Load the images
        current_panorama = cv.imread(general_folder_path + 'panorama.jpg')
        new_image = cv.imread(general_folder_path + 'aligned_image.jpg')

        # Create a 3x3 translation matrix
        translation_matrix = np.float32([
            [1, 0, b_x_min - anchorX1],
            [0, 1, b_y_min - anchorY1],
            [0, 0, 1]
        ])

        # Apply the translation to the images
        dst_padded, warped_image, anchorX, anchorY = warp_perspective_padded(current_panorama, new_image, translation_matrix)
        idx = image_corners_df[image_corners_df['image_path'] == last_key].index[0]
        image_corners_df.at[idx, 'corners'] += [b_x_min - anchorX1 + anchorX, b_y_min -anchorY1 + anchorY]
        for img_path in image_corners_df['image_path']:
            if img_path != image_path1:
                idx = image_corners_df[image_corners_df['image_path'] == img_path].index[0]
                image_corners_df.at[idx, 'corners'] += [anchorX, anchorY]

        start = False

        # Create a mask where dst_padded has zero pixels
        mask_dst = cv.cvtColor(dst_padded, cv.COLOR_BGR2GRAY)
        mask_dst = (mask_dst == 0).astype(np.uint8)

        # Ensure mask has three channels
        mask_dst_3ch = cv.merge([mask_dst, mask_dst, mask_dst])

        # Combine images by filling zeros in dst_padded with pixels from warped_image
        combined_image = dst_padded.copy()
        combined_image[mask_dst_3ch == 1] = warped_image[mask_dst_3ch == 1]

        # Save the combined image
        cv.imwrite(general_folder_path + 'aligned_image.jpg', combined_image)
    
    # Save the final panorama.
    final_panorama_path = os.path.join(general_folder_path, 'final_panorama.jpg')
    cv2.imwrite(final_panorama_path, combined_image)
    
    torch.cuda.empty_cache()

    directory = "results/"

    for root, _, files in os.walk(directory):
        for file in files:
            if file == "aligned_image.jpg" or file == "warped_image.jpg" or file == "panorama.jpg":
                file_path = os.path.join(root, file)
                os.remove(file_path)

    return combined_image, image_corners_df
