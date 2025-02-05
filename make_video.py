import cv2
import os

def create_video_from_heatmaps(image_dir, output_video_path, fps=30):
    # List all files in the image directory, sorted by name
    image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    if not image_files:
        print(f"No images found in directory: {image_dir}")
        return

    print(f"Found {len(image_files)} images in {image_dir}")

    # Read the first image to get the frame size
    first_image_path = os.path.join(image_dir, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read the first image: {first_image_path}")
        return

    # Get the frame dimensions
    height, width, layers = frame.shape
    print(f"Frame size: {width}x{height}")

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate through the images and write them to the video
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image: {image_path}")
            continue

        video_writer.write(frame)
        print(f"Added frame: {image_path}")

    # Release the video writer
    video_writer.release()
    print(f"Video saved to: {output_video_path}")

# Paths and parameters
image_dir = "./heatmap/S01E01_000/"  # Directory containing heatmap images
output_video_path = "./heatmap/video/S01E01_002_heatmap_video.mp4"  # Output video file
fps = 23.976023976023978  # Frames per second for the video

# Create video from heatmap images
create_video_from_heatmaps(image_dir, output_video_path, fps)
