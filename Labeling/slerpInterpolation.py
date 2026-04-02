import sys
import os
import re
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count

def slerp(v0, v1, t):
    """Spherical linear interpolation between two 3D vectors."""
    v0 = np.array(v0, dtype=np.float64)
    v1 = np.array(v1, dtype=np.float64)
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return (1.0 - t) * v0 + t * v1
    v0n = v0 / n0
    v1n = v1 / n1
    dot = np.clip(np.dot(v0n, v1n), -1.0, 1.0)
    omega = np.arccos(dot)
    if omega < 1e-6:
        return v0n
    sin_omega = np.sin(omega)
    return (np.sin((1.0 - t) * omega) / sin_omega) * v0n + (np.sin(t * omega) / sin_omega) * v1n

def correct_image(image_path, wb_override, out_image_path):
    """Apply white balance to the image using the provided RGB override and save the result."""
    rgb = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise ValueError(f"Could not read image: {image_path}")
    rgb = rgb.astype(np.float32) / 255.0
    wbParameters = np.array(wb_override, dtype=np.float32)
    white_balanced_rgb = rgb / (wbParameters + 1e-6)
    white_balanced_rgb = np.clip(white_balanced_rgb, 0, 1)
    os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
    cv2.imwrite(out_image_path, (white_balanced_rgb * 255).astype(np.uint8))

# --- Helpers for frame names ---
def _parse_frame_index(frame_name):
    digits = re.findall(r'(\d+)', frame_name)
    if not digits:
        raise ValueError(f"No digits found in frame name '{frame_name}'")
    return int(digits[-1])

def _format_frame_name(idx, pad):
    return f"frame_{idx:0{pad}d}.png"


def worker(args):
    """Worker function to process a single folder."""
    folder_path, segment_background, device, get_viz, max_hole_size = args
    try:
        print(f"[INFO] Worker starting: {folder_path}")
        # Define paths
        raw_png_folder = os.path.join(folder_path, "rgb_png")
        output_base_folder = os.path.join(folder_path, "processed_labeled_RGB_png")
        output_image_folder = os.path.join(output_base_folder, "frames")
        video_label_file = os.path.join(folder_path, "labels.csv")

        if not os.path.exists(video_label_file):
            print(f"[WARN] labels.csv not found in {folder_path} — skipping.")
            return

        # Load existing labels
        labels = {}
        original_frame_names = {}
        with open(video_label_file, 'r') as fh:
            header = fh.readline()
            for ln in fh:
                parts = ln.strip().split(',')
                if len(parts) < 4:
                    continue
                frame_name = parts[0].strip()
                try:
                    idx = _parse_frame_index(frame_name)
                    r, g, b = float(parts[1]), float(parts[2]), float(parts[3])
                    labels[idx] = np.array([r, g, b], dtype=np.float64)
                    original_frame_names[idx] = frame_name
                except Exception as e:
                    print(f"[WARN] Skipping malformed line: {ln} ({e})")

        if len(labels) < 2:
            print(f"[WARN] Not enough labeled frames to interpolate.")
            return

        # Determine padding
        pad_width = None
        if original_frame_names:
            some_name = next(iter(original_frame_names.values()))
            ds = re.findall(r'(\d+)', some_name)
            pad_width = len(ds[-1]) if ds else 5
        else:
            pad_width = 5

        # Interpolate gaps and save corrected images
        present_sorted = sorted(labels.keys())
        new_entries = 0
        for i in range(len(present_sorted) - 1):
            # Check gap size
            start_idx = present_sorted[i]
            end_idx = present_sorted[i + 1]
            gap = end_idx - start_idx - 1
            if gap <= 0:
                continue
            if gap > max_hole_size:
                print(f"[WARN] Hole size {gap} > max_hole_size, skipping.")
                continue
            # Perform SLERP interpolation for frames in the gap
            v_start = labels[start_idx]
            v_end = labels[end_idx]
            for k in range(start_idx + 1, end_idx):
                t = (k - start_idx) / (end_idx - start_idx)
                interp = slerp(v_start, v_end, t)
                labels[k] = interp
                new_entries += 1

                frame_name = _format_frame_name(k, pad_width)
                image_path = os.path.join(raw_png_folder, frame_name)
                out_image_path = os.path.join(output_image_folder, frame_name)
                if os.path.exists(image_path):
                    # Apply white balance and save interpolated image
                    try:  
                        correct_image(image_path, wb_override=interp, out_image_path=out_image_path)
                        print(f"[INFO] Saved interpolated image {frame_name}")
                    except Exception as e:
                        print(f"[ERROR] Failed to save interpolated image {frame_name}: {e}")
                else:
                    print(f"[WARN] Missing raw image for {frame_name}, skipping image save.")

        # Save updated labels.csv
        out_indices = sorted(labels.keys())
        with open(video_label_file, 'w') as fh:
            fh.write("frame,red,green,blue\n")
            for idx in out_indices:
                vec = labels[idx]
                fh.write(f"{_format_frame_name(idx, pad_width)},{vec[0]},{vec[1]},{vec[2]}\n")

        print(f"[INFO] Finished {folder_path}: interpolated {new_entries} frames.")
        return new_entries

    except Exception as e:
        print(f"[ERROR] Worker failed on {folder_path}: {e}")
        return 0

# --- Multiprocessing driver ---
from functools import partial

def process_all_folders_in_directory(root_path, segment_background=False, device='cpu', get_viz=False, max_hole_size=60):
    """Process all folders in the given root path using multiprocessing."""
    # Find all subdirectories
    folder_paths = [
        os.path.join(root_path, name)
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]
    # Prepare arguments for each worker
    args_list = [
        (folder_path, segment_background, device, get_viz, max_hole_size)
        for folder_path in folder_paths
    ]
    # Run workers in parallel
    total_interpolated = 0
    with Pool(processes=min(cpu_count(), len(args_list))) as pool:
        results = pool.map(worker, args_list)
        total_interpolated = sum(results)

    print(f"\n[SUMMARY] Total interpolated frames across all folders: {total_interpolated}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python slerpInterpolation.py <path_to_base_directory>")
        sys.exit(1)
    filename = sys.argv[1]
    process_all_folders_in_directory(filename, device='cuda', get_viz=True, segment_background=False, max_hole_size=180)
