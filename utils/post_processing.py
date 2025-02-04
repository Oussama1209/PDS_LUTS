import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random

def compute_turn_angles(points):
    """co
    Return list of turn angles between consecutive segments in degrees.
    """
    angles = []
    for i in range(len(points) - 2):
        v1 = points[i+1] - points[i]
        v2 = points[i+2] - points[i+1]
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)
        if mag_v1 == 0 or mag_v2 == 0:
            continue  # skip degenerate segments
        dot_product = np.dot(v1, v2)
        cos_angle = dot_product / (mag_v1 * mag_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # numerical stability
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        angles.append(angle_deg)
    return angles

def compute_bounding_radius(points):
    """
    Compute a simple bounding radius for a set of points:
    - centroid = average (x, y)
    - bounding radius = max distance of any point to centroid
    """
    if len(points) == 0:
        return 0.0
    centroid = np.mean(points, axis=0)  # shape (2,)
    distances = np.linalg.norm(points - centroid, axis=1)
    return np.max(distances)

def classify_trajectory(points, angle_threshold=90, fraction_allowed=0.3, radius_threshold=50):
    """
    Classify a trajectory into one of three categories:
      - 'big_anomaly'
      - 'small_anomaly'
      - 'normal'

    Criteria:
      1) Compute fraction_exceed = fraction of turn angles > angle_threshold.
      2) Compute bounding radius.
      3) If fraction_exceed > fraction_allowed:
           - If bounding_radius > radius_threshold => 'big_anomaly'
           - Else => 'small_anomaly'
         else => 'normal'
    """
    angles = compute_turn_angles(points)
    # If very few points => no angles => consider it normal or handle as you wish
    if len(angles) < 2:
        return 1

    # fraction of angles that exceed threshold
    count_exceed = sum(1 for a in angles if a > angle_threshold)
    fraction_exceed = count_exceed / len(angles)

    # compute bounding radius
    bounding_radius = compute_bounding_radius(points)

    # check anomaly logic
    if fraction_exceed > fraction_allowed:
        # It's an anomaly. Distinguish big vs. small based on radius
        if bounding_radius > radius_threshold:
            return 2
        else:
            return 1
    else:
        return 0
    
def post_process_trajectories(map_image, loaded_dict, marker_size=2):
    # Create a figure to display the panorama
    plt.figure(figsize=(40, 20))
    plt.imshow(cv.cvtColor(map_image, cv.COLOR_BGR2RGB))
    plt.axis('off')
    i=0
    j=0

    colors = {}
    random.seed(42) 

    for trajectories in loaded_dict:

        for vehicle_id, points in trajectories.items():
            if len(loaded_dict) >14 and trajectories == loaded_dict[14]:
                continue
            if vehicle_id not in colors:
                colors[vehicle_id] = [random.randint(0, 255) for _ in range(3)]
            # Convert points to NumPy and do your bounding checks
            points = np.array(points)
            height, width, _ = map_image.shape
            valid_indices = (
                (points[:, 0] >= 0) & (points[:, 0] < width) &
                (points[:, 1] >= 0) & (points[:, 1] < height)
            )
            points = points[valid_indices]

            # Classify the trajectory
            label = classify_trajectory(points,
                                        angle_threshold=60,
                                        fraction_allowed=0.4,
                                        radius_threshold=100)
            j+=1

            # Decide whether to plot
            if label == 2:
                # Skip it (don't plot)
                continue

            i+=1
            
            if label == 1:
                if points.shape[0] > 0:
                    centroid = np.mean(points, axis=0)
                    plt.plot(centroid[0], centroid[1], 'o',  # diamond marker
                            color=np.array(colors[vehicle_id])/255.0,
                            markersize=marker_size, label=f'Stopped ID {vehicle_id}')
                continue  # skip the rest of the logic below

            # Otherwise, it's normal => plot
            if points.shape[0] >= 2:
                plt.plot(points[:, 0], points[:, 1], '-',
                        color=np.array(colors[vehicle_id])/255.0,
                        linewidth=3, label=f'ID {vehicle_id}')
                # Start & end markers
                plt.plot(points[0, 0], points[0, 1], 'o',
                        color=np.array(colors[vehicle_id])/255.0,
                        markersize=marker_size)
                plt.plot(points[-1, 0], points[-1, 1], 's',
                        color=np.array(colors[vehicle_id])/255.0,
                        markersize=marker_size)
            elif points.shape[0] == 1:
                plt.plot(points[0, 0], points[0, 1], 'o',
                        color=np.array(colors[vehicle_id])/255.0,
                        markersize=marker_size, label=f'ID {vehicle_id}')

    print(f"number of vehicles shown is: {i}")
    print(f"number of vehicles skipped is: {j}")
    print(f"Percenatge of vehicles shown is: {i/j*100}")
    plt.tight_layout()
    plt.show()