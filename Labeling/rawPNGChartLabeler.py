import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ColorChartLocalization.annotate import annotate
import os
import torch
import torchvision
from multiprocessing import Pool, cpu_count
import shutil



def slerp(v0,v1,t):
    """Spherical linear interpolation between two vectors v0 and v1 with interpolation factor t (0 <= t <= 1)"""
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    
    dot = np.clip(np.dot(v0,v1), -1.0,1.0)
    omega = np.arccos(dot)
    if omega < 1e-6:
        return v0
    sin_omega = np.sin(omega)
    return (np.sin((1-t) * omega)/sin_omega) * v0 + (np.sin(t*omega) / sin_omega) * v1

def angular_distance_deg(v1, v2):
    """Computes the angular distance in degrees between two vectors v1 and v2."""
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

def worker(args):
    """Worker function to process a single folder. This is necessary for multiprocessing."""
    from rawPNGChartLabeler import process_folder  # if needed to avoid circular imports or global issues
    folder_path, segment_background, device, get_viz, max_hole_size = args
    try:
        print(f"[INFO] Starting: {folder_path}")
        label_count = process_folder(folder_path, segment_background, device, get_viz, max_hole_size)
        print(f"[INFO] Finished: {folder_path}")
        return label_count
    except Exception as e:
        print(f"[ERROR] Failed on {folder_path}: {e}")


def label_image(image_path, out_csv_file=None, segment_background=False, device='cpu', get_viz=False, out_image_path=None, wb_override=None):
    """
    Labels a single image and returns the white balance triplet. 
    If out_csv_file is provided, saves the annotation results in a csv file.
    If out_image_path is provided, saves the white balanced image at that path.
    If wb_override is provided, uses those white balance parameters directly instead of running the annotation process.
    """
    # Load and normalize the image
    rgb = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    rgb = rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # If white balance parameters are provided, use them directly. Otherwise, run the annotation process to get them.
    if wb_override is not None:
        wbParameters = np.array(wb_override, dtype=np.float32)
    else:
        try:
            # Annotate using the YOLO based model. Output is saved in the out_csv_file.
            annotate(image_path, out_csv_file, segment_background, device, get_viz)
        except RuntimeError as e:
            print(f"[ERROR] Error during annotation: {e}")
            return None

        # Read the output csv file
        if out_csv_file is not None and os.path.exists(out_csv_file):
            df = pd.read_csv(out_csv_file)
        else:
            print(f"[ERROR] Missing CSV output for image: {image_path}")
            return None

        # Filter out valid white entries
        if "White" not in df.columns:
            print(f"[ERROR] 'White' column missing in CSV for image: {image_path}")
            return None

        white_triplets = []
        for val in df["White"].dropna():
            try:
                # Convert string representation of list to numpy array and normalize to [0, 1]
                triplet = np.array([float(x)/255 for x in val.strip('[]').split(',')], dtype=np.float32)
                if triplet.shape == (3,):
                    white_triplets.append(triplet)
            except Exception as e:
                print(f"[WARN] Skipping invalid white triplet '{val}' in {image_path}: {e}")

        if not white_triplets:
            print(f"[ERROR] No valid white triplets found in CSV for image: {image_path}")
            return None
        # Average all the white patches found in the image
        white_triplet = np.mean(white_triplets, axis=0)

        # If visualization is enabled and coordinates are available, draw red circles on the original image at the specified coordinates and save it
        if get_viz and "White_coords" in df.columns:
            try:
                dot_img = rgb.copy()
                for coord_str in df["White_coords"].dropna():
                    try:
                        x_center, y_center = [int(float(x)) for x in coord_str.strip("[]").split(",")]
                        cv2.circle(dot_img, (x_center, y_center), 5, (0, 0, 255), -1)  # Red dot
                    except Exception as coord_e:
                        print(f"[WARN] Skipping invalid coordinate '{coord_str}': {coord_e}")

                parent_folder = os.path.dirname(out_image_path)
                filename = os.path.basename(out_image_path)
                circle_folder = os.path.join(parent_folder, "circle_marked")
                os.makedirs(circle_folder, exist_ok=True)

                out_circle_path = os.path.join(circle_folder, filename)
                cv2.imwrite(out_circle_path, (dot_img * 255).astype(np.uint8))
            except Exception as e:
                print(f"[WARN] Failed to draw viz circles: {e}")
        # Normalize the white triplet to get the white balance parameters
        wbParameters = white_triplet / np.linalg.norm(white_triplet)

    # Apply white balancing
    white_balanced_rgb = rgb / (wbParameters + 1e-6)
    white_balanced_rgb = np.clip(white_balanced_rgb, 0, 1)
    # Save the white balanced image if an output path is provided
    if out_image_path is not None:
        cv2.imwrite(out_image_path, (white_balanced_rgb * 255).astype(np.uint8))

    return wbParameters.tolist()

def process_frame(image_path, out_csv_file=None, segment_background=False, device='cpu', get_viz=False, out_image_path=None, label_file=None):
    """
    Processes a single frame and saves the white balance triplet in the video label csv file
    """
    print(f"[INFO] Processing: {image_path}")
    try:
        white_triplet = label_image(image_path, out_csv_file, segment_background, device, get_viz, out_image_path)
        return white_triplet
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return None
    
def process_folder(folder_path, segment_background=False, device='cpu', get_viz=False, max_hole_size=10):
    """Processes all frames in a folder and saves the white balance triplets in a csv file. 
    Also saves the white balanced images in a subfolder."""
    raw_png_folder = os.path.join(folder_path, "rgb_png")
    output_base_folder = os.path.join(folder_path, "processed_labeled_RGB_png")

    # Delete the output folder if it already exists
    if os.path.exists(output_base_folder):
        print(f"[INFO] Deleting existing folder: {output_base_folder}")
        shutil.rmtree(output_base_folder)

    # Setup output folders and label file
    output_image_folder = os.path.join(output_base_folder, "frames")
    output_csv_folder = os.path.join(output_base_folder, "CSVs")

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_csv_folder, exist_ok=True)

    video_label_file = os.path.join(folder_path, "labels.csv")
    
    # Process each frame
    frame_files = sorted([f for f in os.listdir(raw_png_folder) if f.endswith(".png")])
    frame_labels = []
    successful_labels = 0  
    for filename in frame_files:
        # Construct paths for input image, output image, and output csv
        image_path = os.path.join(raw_png_folder, filename)
        frame_name = filename.replace(".png", "")
        out_image_path = os.path.join(output_image_folder, f"{frame_name}.png")
        out_csv_file = os.path.join(output_csv_folder, f"{frame_name}.csv")
        # Process the frame and get the white balance triplet
        triplet = process_frame(
            image_path,
            out_csv_file=out_csv_file,
            segment_background=segment_background,
            device=device,
            get_viz=get_viz,
            out_image_path=out_image_path
        )
        # Append the frame and its label to the list
        frame_labels.append((filename, triplet))
        if triplet is not None:
            successful_labels += 1  # Increment counter

    # Save final CSV
    with open(video_label_file, 'w') as f:
        f.write("frame,red,green,blue\n")
        #for frame, label in interpolated_labels:
        for frame, label in frame_labels:
            if label is not None:
                f.write(f"{frame},{label[0]},{label[1]},{label[2]}\n")
                   
    # If no images were successfully labeled, the input video is not usable, so we can delete it to save space
    if not os.listdir(output_image_folder):
        print(f"[WARN] No images were processed successfully in folder: {folder_path}, deleting folder...")
        shutil.rmtree(folder_path)
    print(f"[INFO] {successful_labels}/{len(frame_files)} frames successfully labeled in folder: {folder_path}")
    
    return successful_labels

def process_all_folders_in_directory(root_path, segment_background=False, device='cpu', get_viz=False, max_hole_size = 60):
    """
    Finds all folders in the root_path and applies process_folder() to each one in parallel.
    """
    
    # Get all subfolder paths in the root directory
    folder_paths = [
        os.path.join(root_path, name)
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]
    # Create a list of arguments for each folder to pass to the worker function
    args_list = [
        (folder_path, segment_background, device, get_viz,max_hole_size)
        for folder_path in folder_paths
    ]
    total_labeled = 0
    # Process folders in parallel using multiprocessing Pool
    with Pool(processes=min(cpu_count(), len(args_list))) as pool:
        result = pool.map(worker, args_list)
        total_labeled = sum(result)
        
    print(f"\n[SUMMARY] Total successfully labeled frames across all folders: {total_labeled}")
    
    
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python rawPNGChartLabeler.py <path_to_base_directory>")
        sys.exit(1)
        
    filename = sys.argv[1]
    process_all_folders_in_directory(filename, device='cuda', get_viz=False, segment_background=False, max_hole_size=150)
    
