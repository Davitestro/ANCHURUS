import os
import cv2
import torch
import numpy as np
from tqdm import tqdm 

VIDEO_DIR = "videos"
DATASET_DIR = "processed_data"
FPS = 5  # Extract 1 frame every 5 frames

os.makedirs(DATASET_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained MiDaS depth estimation model (lightweight version)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Get list of video files to process
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".webm"))]

# Outer progress bar for videos
for video_file in tqdm(video_files, desc="Total Videos", unit="video"):
    video_path = os.path.join(VIDEO_DIR, video_file)
    video_name = os.path.splitext(video_file)[0]
    
    rgb_out = os.path.join(DATASET_DIR, "rgb", video_name)
    depth_out = os.path.join(DATASET_DIR, "depth", video_name)
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pbar = tqdm(total=total_frames, desc=f" Processing {video_file[:10]}...", leave=False, unit="fr")
    
    count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Process every FPS-th frame
        if count % FPS == 0:
            # Save RGB frame as JPEG
            rgb_path = os.path.join(rgb_out, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(rgb_path, frame)

            # Predict depth map using MiDaS model
            input_batch = midas_transforms(frame).to(DEVICE)
            with torch.no_grad():
                prediction = midas(input_batch)
                # Resize depth prediction to match original frame dimensions
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False
                ).squeeze()

            # Normalize depth to 0-255 range and save as PNG
            depth = prediction.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_path = os.path.join(depth_out, f"frame_{frame_id:04d}.png")
            cv2.imwrite(depth_path, (depth * 255).astype(np.uint8))
            
            frame_id += 1
        
        count += 1
        pbar.update(1)

    cap.release()
    pbar.close()

print("\n[COMPLETE] All depth maps generated successfully.")
