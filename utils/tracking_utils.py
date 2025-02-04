import os
import cv2
import cv2 as cv
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch
import os
from PIL import Image, ImageDraw
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm
import re
import torchvision.transforms.functional as TF
from lightglue import LightGlue, DoGHardNet
from lightglue.utils import match_pair
from collections import defaultdict
import pandas as pd
import warnings


def select_frames(
    video_path, temp_folder, txt_folder, output_folder,
    start_frame, step_frame, end_frame, dimensions=(3840, 2160), offset=10
):
    """
    Selects frames from a video, processes them with inpainting based on bounding boxes, 
    and saves the results to an output folder.

    Parameters
    ----------
    video_path : str
        Path to the input video file.

    temp_folder : str
        Path to the temporary folder where the frames will be temporarily saved.

    txt_folder : str
        Path to the folder containing bounding box text files.

    output_folder : str
        Path to the folder where the inpainted results will be saved.

    start_frame : int
        The starting frame index to begin processing.

    step_frame : int
        The step size to select frames (e.g., every 10th frame).

    end_frame : int
        The ending frame index (non-inclusive) for processing.

    dimensions : tuple
        Target dimensions for resizing the frames (width, height).

    offset : int
        Offset to expand the bounding box region for masking.

    Returns
    -------
    None
    """
    # Ensure the necessary folders exist
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Clean the temporary folder
    for file in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, file)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

    # Clean the temporary folder
    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

    # Adjust start_frame to the nearest multiple of step_frame
    adjusted_start_frame = start_frame + (step_frame - (start_frame % step_frame)) % step_frame

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    frame_index = 0  # Counter to track the current frame in the video
    selected_frame_index = adjusted_start_frame  # Frame to start processing


    # Progress bar setup
    total_frames = (end_frame - adjusted_start_frame) // step_frame + 1
    progress_bar = tqdm(total=total_frames, desc="Processing frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Process the frame if it matches the selected frame index
        if frame_index == selected_frame_index:
            # Save the frame temporarily
            frame_filename = f"{selected_frame_index:04d}.jpg"
            frame_filepath = os.path.join(temp_folder, frame_filename)
            cv2.imwrite(frame_filepath, frame)

            image = Image.open(frame_filepath).convert("RGB")
            image = image.resize(dimensions)

            # Save the inpainted result
            output_filepath = os.path.join(output_folder, frame_filename)
            image.save(output_filepath)

            # Increment to the next selected frame
            selected_frame_index += step_frame
            progress_bar.update(1)

        # Stop if we exceed the end_frame
        if frame_index >= end_frame:
            break

        frame_index += 1  # Move to the next frame

    # Release the video capture and close the progress bar
    cap.release()
    progress_bar.close()

    # Clean the temporary folder after use
    for file in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, file)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

def crop_map(map_image, corners):
    """
    Crops the region from the map based on the given corners.

    Parameters
    ----------
    map_image : np.ndarray
        The full map image as a NumPy array in BGR or RGB format.

    corners : np.ndarray
        A (4, 2) array specifying the rectangular region to crop. Example:
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

    Returns
    -------
    np.ndarray
        The cropped portion of the map.
    """
    x_min = int(np.min(corners[:, 0]))
    x_max = int(np.max(corners[:, 0]))
    y_min = int(np.min(corners[:, 1]))
    y_max = int(np.max(corners[:, 1]))

    # Ensure the coordinates are within the bounds of the image
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(map_image.shape[1], x_max)
    y_max = min(map_image.shape[0], y_max)

    return map_image[y_min:y_max, x_min:x_max]

def rotate_image_and_tracking_data(image, tracking_file, angle):
    """
    Rotates the image and applies the same rotation to the tracking data.

    Parameters
    ----------
    image : np.ndarray
        The input image (as a NumPy array).

    tracking_file : str
        Path to the tracking file containing lines with format: id, x_center, y_center, confidence.

    angle : float
        The angle to rotate the image (in degrees).

    Returns
    -------
    rotated_image : np.ndarray
        The rotated image.

    rotated_tracking_points : list
        A list of transformed tracking points with format: [track_id, x_center_rotated, y_center_rotated].
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Compute the center of the image
    center = (width // 2, height // 2)

    # Create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the image rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Read and transform tracking data
    rotated_tracking_points = []
    with open(tracking_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue

            track_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])

            # Transform the (x, y) coordinates using the rotation matrix
            original_point = np.array([[x_center, y_center, 1.0]])  # Homogeneous coordinates
            rotated_point = original_point @ rotation_matrix.T  # Apply transformation

            x_center_rotated, y_center_rotated = rotated_point[0][:2]  # Extract rotated x, y

            # Save transformed tracking point
            rotated_tracking_points.append([track_id, x_center_rotated, y_center_rotated])

    return rotated_image, rotated_tracking_points

def track_vehicles_in_between_frames(
    map_image,
    df,
    images_folder,
    bounding_data_folder,
    start_frame,
    end_frame,
    step_frame=10,
    trajectories=None,
    device=None
):
    """
    Tracks vehicles between two frames and stores their global trajectories.

    Adjusts start frame to the nearest multiple of `step_frame` greater than or equal to `start_frame`.

    Parameters
    ----------
    [same as before]

    Returns
    -------
    dict
        A dictionary of trajectories keyed by vehicle ID.
    """
    # 1. Initialize 'trajectories' if not provided
    if trajectories is None:
        trajectories = defaultdict(list)

    # 2. Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. Extract corners for start_frame and end_frame from DataFrame
    df_start = df.loc[df['frame_number'] == start_frame]
    df_end = df.loc[df['frame_number'] == end_frame]

    if df_start.empty or df_end.empty:
        print(f"Frame data not found for frames {start_frame} or {end_frame}.")
        return trajectories

    corners_start = df_start.iloc[0]['corners'].astype(np.int32)
    corners_end = df_end.iloc[0]['corners'].astype(np.int32)

    # 4. Compute bounding box for the cropped region
    x_min = min(np.min(corners_start[:, 0]), np.min(corners_end[:, 0]))
    x_max = max(np.max(corners_start[:, 0]), np.max(corners_end[:, 0]))
    y_min = min(np.min(corners_start[:, 1]), np.min(corners_end[:, 1]))
    y_max = max(np.max(corners_start[:, 1]), np.max(corners_end[:, 1]))

    offset_corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    cropped_map = crop_map(map_image, offset_corners)

    # 5. Convert cropped_map to RGB and then to a tensor
    cropped_map_rgb = cv.cvtColor(cropped_map, cv.COLOR_BGR2RGB)
    cropped_map_pil = Image.fromarray(cropped_map_rgb)
    cropped_map_tensor = TF.to_tensor(cropped_map_pil).unsqueeze(0).to(device)

    # 6. Set up LightGlue feature extractor and matcher
    extractor = DoGHardNet(max_num_keypoints=None).eval().to(device)
    matcher = LightGlue(features='doghardnet').eval().to(device)

    print(f"Using offset_x={x_min}, offset_y={y_min} for global coordinates.")

    # 7. Adjust start_frame to the nearest multiple of step_frame
    adjusted_start_frame = start_frame + (step_frame - (start_frame % step_frame)) % step_frame

    # 8. Main loop: iterate through frames [adjusted_start_frame, end_frame) with step_frame
    for frame_number in range(adjusted_start_frame, end_frame, step_frame):
        frame_number_str = f"{frame_number:04d}"  # Ensure leading zeros
        # Load the frame
        frame_path = os.path.join(images_folder, f"{frame_number_str}.jpg")
        if not os.path.isfile(frame_path):
            print(f"Frame image not found: {frame_path}")
            continue

        frame_image_cv = cv.imread(frame_path)
        if frame_image_cv is None:
            print(f"Failed to load frame image: {frame_path}")
            continue

        # Convert frame to RGB -> Tensor
        frame_image_rgb = cv.cvtColor(frame_image_cv, cv.COLOR_BGR2RGB)
        frame_rotated, rotated_tracking_points = rotate_image_and_tracking_data(
            image=frame_image_rgb,
            tracking_file=os.path.join(bounding_data_folder, f"track_fr_{frame_number_str}.txt"),
            angle=df_end.iloc[0]['rotation_angle']
        )
        frame_pil = Image.fromarray(frame_rotated)
        frame_tensor = TF.to_tensor(frame_pil).unsqueeze(0).to(device)

        # Feature matching between this frame and the cropped map
        feats0, feats1, matches01 = match_pair(extractor, matcher, frame_tensor, cropped_map_tensor)
        points0 = feats0['keypoints'][matches01['matches'][..., 0]].cpu().numpy()
        points1 = feats1['keypoints'][matches01['matches'][..., 1]].cpu().numpy()

        if points0.shape[0] < 4:
            print("Not enough matches to compute homography.")
            continue

        # Estimate homography
        M, _ = cv.findHomography(points0, points1, cv.RANSAC, 5.0)
        if M is None:
            print("Homography computation failed.")
            continue

        # Read bounding box/tracking data for this frame
        bbox_file_path = os.path.join(bounding_data_folder, f"track_fr_{frame_number_str}.txt")
        if not os.path.isfile(bbox_file_path):
            print(f"Tracking data file not found: {bbox_file_path}")
            continue

        # Parse bounding box centers
        with open(bbox_file_path, 'r') as f:
            for i in range(len(rotated_tracking_points)):
                track_id = int(rotated_tracking_points[i][0])
                x_center = float(rotated_tracking_points[i][1])
                y_center = float(rotated_tracking_points[i][2])

                center_point = np.array([[[x_center, y_center]]], dtype=np.float32)
                transformed_center = cv.perspectiveTransform(center_point, M)[0][0]
                global_x = transformed_center[0] + x_min
                global_y = transformed_center[1] + y_min

                trajectories[track_id].append([global_x, global_y])

    return trajectories


