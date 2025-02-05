import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def apply_heatmap_to_image(image_path, saliency_map_path, output_path, alpha=0.4):
    # Load the original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the saliency map
    saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)

    # Normalize the saliency map to range [0, 255]
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a colormap to the saliency map
    heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

    # Resize the heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap_resized = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB)

    # Blend the resized heatmap with the original image using alpha transparency
    blended = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_resized, alpha, 0)

    # Save the result
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, blended_bgr)

def process_directory(frames_dir, saliency_dir, output_dir, alpha=0.4):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all frame and saliency files
    frame_files = [f for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))]
    saliency_files = [f for f in os.listdir(saliency_dir) if os.path.isfile(os.path.join(saliency_dir, f))]

    # Create a dictionary to map frame numbers to saliency map files
    saliency_map_dict = {}
    for saliency_file in saliency_files:
        match = re.search(r'pred_sal_img(\d+)-\d+\.jpg', saliency_file)
        if match:
            frame_number = match.group(1).zfill(5)  # Ensure leading zeros
            saliency_map_dict[frame_number] = saliency_file

    for frame_file in frame_files:
        match = re.search(r'img_(\d+)\.jpg', frame_file)
        if not match:
            print(f"Skipping unrecognized frame file format: {frame_file}")
            continue

        frame_number = match.group(1).zfill(5)  # Ensure leading zeros match saliency files
        if frame_number in saliency_map_dict:
            saliency_path = os.path.join(saliency_dir, saliency_map_dict[frame_number])
            frame_path = os.path.join(frames_dir, frame_file)
            output_path = os.path.join(output_dir, frame_file)

            try:
                apply_heatmap_to_image(frame_path, saliency_path, output_path, alpha)
                print(f"Processed and saved heatmap for: {frame_file}")
            except Exception as e:
                print(f"Error processing {frame_file}: {e}")
        else:
            print(f"Saliency map not found for frame: {frame_file}")

# Paths to directories
frames_dir = "./data/video_frames/MEmoR/S01E01_000/"
saliency_dir = "./experiments_on_av_data/audio_visual/split4_results/memor/S01E01_000/"
output_dir = "./heatmap/S01E01_000/"

# Alpha value for blending
alpha = 0.4

# Process all frames and generate heatmaps
process_directory(frames_dir, saliency_dir, output_dir, alpha)
