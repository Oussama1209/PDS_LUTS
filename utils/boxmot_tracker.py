import os
import cv2
import numpy as np
from pathlib import Path
from boxmot import BotSort
import torch
import shutil

def run_tracking(
    video_path,
    bounding_box_dir,
    tracking_output_dir,
):
    # Use CPU if requested or if CUDA is not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
    # Initialize the tracker
    tracker = BotSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'),  # Path to ReID model
        device=device,
        half=False
    )
    
    # Open the existing video file
    vid = cv2.VideoCapture(video_path)
    
    # Check if video was opened successfully
    if not vid.isOpened():
        raise ValueError("Error opening video file")
    
    # Ensure the tracking output directory exists
    os.makedirs(tracking_output_dir, exist_ok=True)
    
    # Function to parse bounding box data for each frame
    def parse_bounding_boxes(line):
        data = line.strip().split(',')
        if len(data) < 10:
            return None  # Skip invalid entries

        try:
            # Parse bounding box coordinates
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, data[:8])
            x_min, y_min = min(x1, x2, x3, x4), min(y1, y2, y3, y4)
            x_max, y_max = max(x1, x2, x3, x4), max(y1, y2, y3, y4)
            score = float(data[9])  # Confidence score
            class_id = 0  # Assuming all are vehicles (class_id=0)
            
            return [x_min, y_min, x_max, y_max, score, class_id]
        
        except ValueError:
            return None
    
    def clear_folder(folder_path):
        """Delete all contents of a folder by removing and recreating it."""
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # Delete the entire folder
        os.makedirs(folder_path)  # Recreate the empty folder

    clear_folder(tracking_output_dir)
    
    def calculate_bbox_center(x_min, y_min, x_max, y_max):
        """Calculate and round the center of the bounding box."""
        center_x = round((x_min + x_max) / 2)
        center_y = round((y_min + y_max) / 2)
        return center_x, center_y
    
    # Function to compute Intersection over Union (IoU)
    def compute_iou(box1, box2):
        """Compute IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Determine the coordinates of the intersection rectangle
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Compute the area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Compute the area of both bounding boxes
        bb1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bb2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # Compute the IoU
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou

    # Initialize a resizable window
    cv2.namedWindow('BoXMOT + Pre-existing Bounding Boxes', cv2.WINDOW_NORMAL)

    frame_idx = 0
    while True:
        # Capture frame-by-frame from video file
        ret, frame = vid.read()
        if not ret:
            break

        # Read bounding boxes for the current frame from file
        bounding_box_file = os.path.join(bounding_box_dir, f'det_fr_{frame_idx:04d}.txt')
        
        if os.path.isfile(bounding_box_file):
            with open(bounding_box_file, 'r') as f:
                frame_detections = [parse_bounding_boxes(line) for line in f.readlines()]
                frame_detections = [detection for detection in frame_detections if detection is not None]
        else:
            frame_detections = []

        # Convert detections to numpy array format required by BoxMOT
        dets = np.array(frame_detections)
        
        # Update tracker if detections are available
        if dets.size > 0:
            res = tracker.update(dets, frame)

            # Plot tracking results on the frame
            tracker.plot_results(frame, show_trajectories=True)

            # Save tracking results to a text file
            tracking_file = os.path.join(tracking_output_dir, f'track_fr_{frame_idx:04d}.txt')
            with open(tracking_file, 'w') as f_track:
                for track in res:
                    x_min, y_min, x_max, y_max, track_id = track[:5]
                    # Attempt to find matching detection to get the confidence score
                    max_iou = 0
                    confidence = None
                    for det in dets:
                        det_x_min, det_y_min, det_x_max, det_y_max, det_confidence, _ = det
                        iou = compute_iou([x_min, y_min, x_max, y_max], [det_x_min, det_y_min, det_x_max, det_y_max])
                        if iou > max_iou:
                            max_iou = iou
                            confidence = det_confidence
                    # Write tracking information to file
                    # Format: id,x_min,y_min,x_max,y_max,confidence
                    if confidence is not None:
                        center_x, center_y = calculate_bbox_center(x_min, y_min, x_max, y_max)
                        f_track.write(f"{int(track_id)},{center_x},{center_y},{confidence}\n")
                    else:
                        center_x, center_y = calculate_bbox_center(x_min, y_min, x_max, y_max)
                        f_track.write(f"{int(track_id)},{center_x},{center_y}\n")
        else:
            res = []

        frame_idx += 1

    # Release resources
    vid.release()
    cv2.destroyAllWindows()

    print(f"Tracking data saved in '{tracking_output_dir}'")