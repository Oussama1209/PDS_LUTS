import matplotlib.pyplot as plt
import cv2 as cv
import cv2
import glob
import pandas as pd
import re
import os
from PIL import Image, ImageDraw
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm
import shutil
import numpy as np
import torch

def clear_folder(folder_path):
    """
    Remove all files and subdirectories in the given folder.
    """
    if not os.path.exists(folder_path):
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory and its contents
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Function to read the contents of an SRT file
def read_srt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def parse_corners_string(s):
    """
    Convert a string of corner coordinates (without commas)
    into a NumPy array of integers.
    
    Example input:
      "[[1630 2782]\n [5469 2782]\n [5469 4941]\n [1630 4941]]"
    
    Returns:
      A NumPy array of shape (4, 2) with integer coordinates.
    """
    # Remove any surrounding whitespace and outer brackets
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    
    # Split the string by lines
    lines = s.splitlines()
    coords = []
    for line in lines:
        # Remove stray brackets and extra spaces from each line
        line = line.strip().strip('[]')
        if not line:
            continue
        # Split on whitespace
        parts = line.split()
        # Convert each number to int (using float conversion if needed)
        try:
            row_coords = [int(float(x)) for x in parts]
        except ValueError:
            row_coords = [float(x) for x in parts]  # fallback if needed
        coords.append(row_coords)
    return np.array(coords, dtype=np.int32)

def show_panorama(combined_image, image_corners_df):
    """
    Display the final panorama image with the frame corners highlighted.
    
    This function expects:
      - combined_image: a BGR image (as from OpenCV)
      - image_corners_df: a DataFrame with a 'corners' column where each
        entry is a string of the form:
          "[[1630 2782]\n [5469 2782]\n [5469 4941]\n [1630 4941]]"
    
    The function converts the string to a NumPy array of integers,
    then plots the points.
    """
    plt.figure(figsize=(40, 30))
    # Convert image from BGR (OpenCV) to RGB for correct display
    plt.imshow(cv.cvtColor(combined_image, cv.COLOR_BGR2RGB))

    # Iterate over DataFrame to draw corners
    for _, row in image_corners_df.iterrows():
        corners = row['corners']  # This should be a NumPy array if loaded correctly
        
        if isinstance(corners, np.ndarray):  # Ensure it's a valid array
            x_coords, y_coords = corners[:, 0], corners[:, 1]
            plt.plot(x_coords, y_coords, 'o-')
            plt.fill(x_coords, y_coords, alpha=0.3)

    plt.axis('on')
    plt.show()

def parse_srt_to_dataframe(srt_file):
    # Define the data structure
    data = {
        'FrameCnt': [], 'Start_Time': [], 'End_Time': [], 'DiffTime_ms': [],
        'ISO': [], 'Shutter': [], 'Fnum': [], 'EV': [], 'CT': [], 
        'Color_Mode': [], 'Focal_Length': [], 'Latitude': [], 
        'Longitude': [], 'Altitude': [], 'Image': [], 'Resolution_B': [], 'Resolution_S': [],
        'Corners_B': [], 'Corners_S': [], 'Homography': [], 'Neighbors': [],
    }

    # Read and parse the SRT file
    with open(srt_file, 'r') as file:
        content = file.read()
        blocks = content.split('\n\n')  # Split into blocks by double newlines

        for block in blocks:
            # Match each field within the block
            frame_count = re.search(r'FrameCnt : (\d+)', block)
            diff_time = re.search(r'DiffTime : (\d+)ms', block)
            time_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', block)
            iso = re.search(r'\[iso : (\d+)\]', block)
            shutter = re.search(r'\[shutter : ([\d/\.]+)\]', block)
            fnum = re.search(r'\[fnum : (\d+)\]', block)
            ev = re.search(r'\[ev : ([\d\.\-]+)\]', block)
            ct = re.search(r'\[ct : (\d+)\]', block)
            color_mode = re.search(r'\[color_md : (.*?)\]', block)
            focal_len = re.search(r'\[focal_len : (\d+)\]', block)
            latitude = re.search(r'\[latitude : ([-\d.]+)\]', block)
            longitude = re.search(r'\[longtitude : ([-\d.]+)\]', block)  # Using provided typo
            altitude = re.search(r'\[altitude: ([-\d.]+)\]', block)

            # Extract time data if available
            if time_match:
                start_time, end_time = time_match.groups()
            else:
                start_time, end_time = None, None

            # Append data to dictionary, handling None values where needed
            data['FrameCnt'].append(int(frame_count.group(1)) if frame_count else None)
            data['Start_Time'].append(start_time)
            data['End_Time'].append(end_time)
            data['DiffTime_ms'].append(int(diff_time.group(1)) if diff_time else None)
            data['ISO'].append(int(iso.group(1)) if iso else None)
            data['Shutter'].append(shutter.group(1) if shutter else None)
            data['Fnum'].append(int(fnum.group(1)) if fnum else None)
            data['EV'].append(float(ev.group(1)) if ev else None)
            data['CT'].append(int(ct.group(1)) if ct else None)
            data['Color_Mode'].append(color_mode.group(1) if color_mode else None)
            data['Focal_Length'].append(int(focal_len.group(1)) if focal_len else None)
            data['Latitude'].append(float(latitude.group(1)) if latitude else None)
            data['Longitude'].append(float(longitude.group(1)) if longitude else None)
            data['Altitude'].append(float(altitude.group(1)) if altitude else None)

    max_length = max(len(column) for column in data.values())
    for key, column in data.items():
        if len(column) < max_length:
            column.extend([None] * (max_length - len(column)))

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    df['FrameCnt'] = df['FrameCnt'].astype('Int64')
    return df

def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def select_frames_by_distance(df, distance_threshold):
    selected_frames = [0]  # Always start with frame 0
    last_lat, last_lon = df.loc[0, 'Latitude'], df.loc[0, 'Longitude']

    for i in range(1, len(df)):
        lat, lon = df.loc[i, 'Latitude'], df.loc[i, 'Longitude']
        dist = haversine(last_lat, last_lon, lat, lon)
        if dist >= distance_threshold:
            selected_frames.append(i)
            last_lat, last_lon = lat, lon
    return selected_frames

def is_not_blurry(frame, threshold=200.0):
    """
    Determine if an image is blurry using the Laplacian variance method.
    If the variance of the Laplacian is below the threshold, consider it blurry.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()
    return variance > threshold

def extract_frames_with_distance(video_path, output_folder, pkl_file, distance_threshold=50, blur_threshold=200.0):
    # Clear the output folder
    clear_folder(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Read the PICKLE and get the selected frames
    df = pd.read_pickle(pkl_file)
    selected_frames = select_frames_by_distance(df, distance_threshold)
    
    # Convert to a queue-like structure for processing
    # We'll move through this list as we find suitable frames
    target_indices = selected_frames.copy()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    saved_frames = 0
    final_selected_frames = []
    
    current_frame_index = 0
    target_idx_pos = 0  # Position in the target_indices list

    while True:
        ret, frame = cap.read()
        if not ret:
            # No more frames
            break

        # If we've processed all target frames, stop
        if target_idx_pos >= len(target_indices):
            break

        target_frame = target_indices[target_idx_pos]

        if current_frame_index < target_frame:
            # We haven't reached the target frame yet, just continue reading
            current_frame_index += 1
            continue

        if current_frame_index == target_frame:
            # Check if the frame at target_frame is blurry
            if is_not_blurry(frame, blur_threshold):
                # Not blurry, save it
                cv2.imwrite(os.path.join(output_folder, f'{str(current_frame_index).zfill(4)}.jpg'), frame)
                saved_frames += 1
                final_selected_frames.append(current_frame_index)
                target_idx_pos += 1
            else:
                pass

        else:
            # current_frame_index > target_frame
            # This means we already passed the target frame (because it was blurry)
            # and we're checking subsequent frames to find a non-blurry one.
            if is_not_blurry(frame, blur_threshold):
                # Found a suitable frame after the target frame
                cv2.imwrite(os.path.join(output_folder, f'{str(current_frame_index).zfill(4)}.jpg'), frame)
                saved_frames += 1
                final_selected_frames.append(current_frame_index)
                target_idx_pos += 1
            else:
                # Still blurry, keep reading next frames
                pass

        current_frame_index += 1

    cap.release()
    print(f"Extracted {saved_frames} frames from the video.")
    print(f"Selected frame indices (after checking blur): {final_selected_frames}")

def process_inpainting(
    input_folder_name,
    txt_folder_name,
    output_folder_name,
    dimensions=(3840, 2160),
    offset=10
):
    """
    Processes images for inpainting using SimpleLama.
    
    Parameters:
        general_path (str): The base directory for your project.
        input_folder_name (str): Name of the folder containing input images.
        txt_folder_name (str): Name of the folder containing text files with bounding box data.
        output_folder_name (str): Name of the folder to store the processed (inpainted) images.
        dimensions (tuple): The desired output image dimensions (width, height).
        offset (int): Offset margin added to bounding boxes.
    
    The function:
      - Initializes the inpainting model.
      - Clears/creates the output folder.
      - Finds matching image and text files using regular expressions.
      - For each frame number:
            * Resizes the image.
            * Creates a mask based on bounding box data (scaled and with an offset).
            * Performs inpainting.
            * Saves the resulting image.
    """
    # Initialize the inpainting model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    simple_lama = SimpleLama(device=device)

    # Set up folder paths
    input_folder = os.path.join(input_folder_name)
    txt_folder = os.path.join(txt_folder_name)
    output_folder = os.path.join(output_folder_name)

    # Clear the output folder (or create it if it doesn't exist)
    if os.path.exists(output_folder):
        clear_folder(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Define scaling factors (assuming original images are 3840x2160)
    scaling_factor_x = dimensions[0] / 3840
    scaling_factor_y = dimensions[1] / 2160

    # Regular expression patterns to extract frame numbers
    image_pattern = re.compile(r'(\d+)\.jpg$')
    txt_pattern = re.compile(r'det_fr_(\d+)\.txt$')

    # Create dictionaries mapping frame numbers to filenames
    image_files = {
        int(image_pattern.search(f).group(1)): f
        for f in os.listdir(input_folder)
        if image_pattern.search(f)
    }
    txt_files = {
        int(txt_pattern.search(f).group(1)): f
        for f in os.listdir(txt_folder)
        if txt_pattern.search(f)
    }

    # Get the sorted list of frame numbers that appear in both dictionaries
    frame_numbers = sorted(set(image_files.keys()) & set(txt_files.keys()))
    if not frame_numbers:
        print("No matching images and text files found.")
        return

    # Process each frame
    for frame_number in tqdm(frame_numbers, desc="Processing images"):
        image_file = image_files[frame_number]
        txt_file = txt_files[frame_number]

        # Load and resize the image
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        fhd_image = image.resize(dimensions)

        # Create a blank mask image
        mask = Image.new("L", dimensions, 0)

        # Load the corresponding text file with bounding box information
        txt_file_path = os.path.join(txt_folder, txt_file)
        draw = ImageDraw.Draw(mask)
        with open(txt_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # Assume the first 8 comma-separated values are integers for the 4 corners.
                values = list(map(int, line.strip().split(',')[:8]))
                # Scale and offset the bounding box coordinates
                x1, y1 = int(values[0] * scaling_factor_x), int(values[1] * scaling_factor_y)
                x2, y2 = int(values[2] * scaling_factor_x), int(values[3] * scaling_factor_y)
                x3, y3 = int(values[4] * scaling_factor_x), int(values[5] * scaling_factor_y)
                x4, y4 = int(values[6] * scaling_factor_x), int(values[7] * scaling_factor_y)
                x_min = min(x1, x2, x3, x4) - offset
                y_min = min(y1, y2, y3, y4) - offset
                x_max = max(x1, x2, x3, x4) + offset
                y_max = max(y1, y2, y3, y4) + offset
                # Draw the rectangle on the mask
                draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

        # Perform inpainting using SimpleLama
        result = simple_lama(fhd_image, mask)

        # Save the result
        output_image_path = os.path.join(output_folder, image_file)
        result.save(output_image_path)

    print("Processing complete. Inpainted images are saved in:", output_folder)
