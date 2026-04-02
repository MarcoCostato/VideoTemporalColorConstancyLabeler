import sys
import os
import re
import pandas as pd
import cv2
from multiprocessing import Pool, cpu_count
from ColorChartLocalization.annotate import save_detected_boxes

# --- Helpers ---
def _parse_frame_index(frame_name):
    digits = re.findall(r'(\d+)', frame_name)
    if not digits:
        raise ValueError(f"No digits found in frame name '{frame_name}'")
    return int(digits[-1])

def _format_frame_name(idx, pad):
    return f"frame_{idx:0{pad}d}"

def draw_black_boxes(image_path, bboxes, out_path):
    """Draws black boxes (filled) around detections and saves the image."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    for _, row in bboxes.iterrows():
        if {"bbox_y1", "bbox_x1", "bbox_y2", "bbox_x2"}.issubset(row.index):
            y1, x1, y2, x2 = map(int, [row["bbox_y1"], row["bbox_x1"], row["bbox_y2"], row["bbox_x2"]])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)

def worker(args):
    """Worker function to process a single folder."""
    folder_path, device = args
    try:
        print(f"[INFO] Processing: {folder_path}")
        # Build paths
        rgb_png_folder = os.path.join(folder_path, "RGB_png")
        csv_folder = os.path.join(folder_path, "processed_labeled_RGB_png", "CSVs")
        boxes_csv_folder = os.path.join(csv_folder, "BoxesCSVs")
        frame_folder = os.path.join(folder_path, "processed_labeled_RGB_png", "frames")
        boxes_folder = os.path.join(frame_folder, "boxes")

        if not os.path.exists(rgb_png_folder):
            print(f"[WARN] No RGB_png folder for {folder_path}")
            return
        if not os.path.exists(frame_folder):
            print(f"[WARN] No frames folder for {folder_path}")
            return

        os.makedirs(boxes_csv_folder, exist_ok=True)

        # --- Step 1: Generate box CSVs from RGB_png ---
        for fname in sorted(os.listdir(rgb_png_folder)):
            if fname.lower().endswith(".png"):
                img_path = os.path.join(rgb_png_folder, fname)
                save_detected_boxes(img_path, boxes_csv_folder, device=device)

        # --- Step 1b: Draw boxes for existing frames ---
        csv_files = [f for f in os.listdir(boxes_csv_folder) if f.lower().endswith(".csv")]
        if not csv_files:
            print(f"[WARN] No BoxesCSVs found in {boxes_csv_folder}")
            return

        frame_indices = []
        all_csv_data = {}
        pad_width = None
        for csv_file in csv_files:
            idx = _parse_frame_index(csv_file)
            frame_indices.append(idx)
            df = pd.read_csv(os.path.join(boxes_csv_folder, csv_file))
            all_csv_data[idx] = df
            digits = re.findall(r'(\d+)', csv_file)
            if digits:
                pad_width = len(digits[-1])

        if pad_width is None:
            pad_width = 5

        for idx in sorted(frame_indices):
            fname_noext = _format_frame_name(idx, pad_width)
            img_path = os.path.join(frame_folder, f"{fname_noext}.png")
            out_path = os.path.join(boxes_folder, f"{fname_noext}.png")
            if os.path.exists(img_path):
                draw_black_boxes(img_path, all_csv_data[idx], out_path)
            else:
                print(f"[WARN] Missing frame image for {fname_noext}")

        # --- Step 2: Fill holes (unchanged logic) ---
        sorted_indices = sorted(frame_indices)
        for i in range(len(sorted_indices) - 1):
            start_idx = sorted_indices[i]
            end_idx = sorted_indices[i + 1]
            gap = end_idx - start_idx - 1
            if gap <= 0:
                continue

            df_before = all_csv_data.get(start_idx)
            df_after = all_csv_data.get(end_idx)
            if df_before is None or df_after is None:
                continue

            combined_df = pd.concat([df_before, df_after], ignore_index=True)

            for k in range(start_idx + 1, end_idx):
                fname_noext = _format_frame_name(k, pad_width)
                csv_out_path = os.path.join(boxes_csv_folder, f"{fname_noext}.csv")
                combined_df.to_csv(csv_out_path, index=False)

                img_path = os.path.join(frame_folder, f"{fname_noext}.png")
                out_path = os.path.join(boxes_folder, f"{fname_noext}.png")
                if os.path.exists(img_path):
                    draw_black_boxes(img_path, combined_df, out_path)
                else:
                    print(f"[WARN] Missing raw image for {fname_noext}, skipping box draw.")

        print(f"[INFO] Done: {folder_path}")

    except Exception as e:
        print(f"[ERROR] Failed on {folder_path}: {e}")

def process_all_folders_in_directory(root_path, device="cpu"):
    """Process all folders in the given root path using multiprocessing."""
    folder_paths = [
        os.path.join(root_path, name)
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]
    args_list = [(folder_path, device) for folder_path in folder_paths]
    with Pool(processes=min(cpu_count(), len(args_list))) as pool:
        pool.map(worker, args_list)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: targetObfuscation.py <path_to_base_directory>")
        sys.exit(1)
    base_dir = sys.argv[1]
    process_all_folders_in_directory(base_dir, device = 'cuda')

