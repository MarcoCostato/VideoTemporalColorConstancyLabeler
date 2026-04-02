import sys
import os
import re
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial

import pandas as pd
import cv2
import numpy as np

# --- Helpers ---
def _parse_frame_index(frame_name):
    digits = re.findall(r'(\d+)', frame_name)
    if not digits:
        raise ValueError(f"No digits found in frame name '{frame_name}'")
    return int(digits[-1])

def _format_frame_name(idx, pad):
    return f"frame_{idx:0{pad}d}"

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _read_bboxes_from_csv(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return pd.DataFrame(columns=['x1','y1','x2','y2'])

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=['x1','y1','x2','y2'])
    except Exception as e:
        print(f"[WARN] Failed to read CSV {csv_path}: {e}")
        return pd.DataFrame(columns=['x1','y1','x2','y2'])

    if df.empty:
        return pd.DataFrame(columns=['x1','y1','x2','y2'])

    # Normalize all column names to lowercase, strip spaces
    df.columns = [c.strip().lower() for c in df.columns]

    # Map possible name variants
    name_map = {
        'x1': ['x1', 'bbox_x1', 'xmin', 'x_min', 'left'],
        'y1': ['y1', 'bbox_y1', 'ymin', 'y_min', 'top'],
        'x2': ['x2', 'bbox_x2', 'xmax', 'x_max', 'right'],
        'y2': ['y2', 'bbox_y2', 'ymax', 'y_max', 'bottom']
    }

    found = {}
    for key, aliases in name_map.items():
        for alias in aliases:
            if alias in df.columns:
                found[key] = alias
                break

    # if we found all 4 -> use them
    if len(found) == 4:
        return df[[found['x1'], found['y1'], found['x2'], found['y2']]].rename(
            columns={
                found['x1']: 'x1',
                found['y1']: 'y1',
                found['x2']: 'x2',
                found['y2']: 'y2'
            }
        )


    # No usable numeric data
    return pd.DataFrame(columns=['x1','y1','x2','y2'])


def _avg_fill_bboxes(img, bboxes_df):
    """
    img: numpy array (H,W,3) uint8
    bboxes_df: DataFrame with columns x1,y1,x2,y2 (may be floats)
    Modifies and returns the image with each bbox region filled by its mean color.
    """
    if img is None:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    for _, row in bboxes_df.iterrows():
        try:
            x1 = int(round(row['x1']))
            y1 = int(round(row['y1']))
            x2 = int(round(row['x2']))
            y2 = int(round(row['y2']))
        except Exception:
            continue
        # clip coords
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        # ensure x1<=x2 and y1<=y2 and at least 1 pixel
        if x2 <= x1 or y2 <= y1:
            continue
        patch = out[y1:y2+1, x1:x2+1]  # inclusive
        if patch.size == 0:
            continue
        # compute mean color per channel and fill
        # note: keep dtype uint8
        mean_color = np.round(patch.reshape(-1, patch.shape[2]).mean(axis=0)).astype(np.uint8)
        out[y1:y2+1, x1:x2+1] = mean_color
    return out

# --- Worker (per folder/video) ---
def worker(args):
    folder_path, output_base= args
    try:
        videoname = os.path.basename(folder_path.rstrip("/\\"))
        print(f"[INFO] Processing video folder: {videoname} -> {folder_path}")

        rgb_png_folder = os.path.join(folder_path, "RGB_png")
        csv_base = os.path.join(folder_path, "processed_labeled_RGB_png", "CSVs")
        faces_csv_folder = os.path.join(csv_base, "FacesCSVs")
        plates_csv_folder = os.path.join(csv_base, "PlatesCSVs")
        boxes_csv_folder = os.path.join(csv_base, "BoxesCSVs")  # for copying to output if present

        # Output paths
        out_video_folder = os.path.join(output_base, videoname)
        out_frames_folder = os.path.join(out_video_folder, "frames")
        _ensure_dir(out_frames_folder)

        if not os.path.exists(rgb_png_folder):
            print(f"[WARN] No RGB_png folder for {folder_path}. Skipping.")
            return

        # list image files
        img_files = sorted([f for f in os.listdir(rgb_png_folder) if f.lower().endswith(".png")])
        if not img_files:
            print(f"[WARN] No PNGs found in {rgb_png_folder}. Skipping.")
            return

        # For each image, try to find matching CSV(s) with same name (but .csv)
        for fname in img_files:
            img_path = os.path.join(rgb_png_folder, fname)
            base_noext = os.path.splitext(fname)[0]
            # candidate CSV paths
            csv_candidates = [
                os.path.join(faces_csv_folder, base_noext + ".csv"),
                os.path.join(plates_csv_folder, base_noext + ".csv"),
            ]
            combined_bboxes = []
            for csvp in csv_candidates:
                if os.path.exists(csvp):
                    df = _read_bboxes_from_csv(csvp)
                    if not df.empty:
                        combined_bboxes.append(df)
            if combined_bboxes:
                all_bboxes = pd.concat(combined_bboxes, ignore_index=True)
            else:
                # no bboxes, simply copy the input image to output location
                out_img_path = os.path.join(out_frames_folder, fname)
                # read and write to normalize behavior
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"[WARN] Could not read image {img_path}; skipping.")
                    continue
                cv2.imwrite(out_img_path, img)
                continue

            # read image, apply average fill for each bbox
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARN] Could not read image {img_path}; skipping.")
                continue

            # Ensure image has 3 channels (if grayscale, convert to BGR)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                # preserve alpha by separating it, operate on BGR only
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                bgr_filled = _avg_fill_bboxes(bgr, all_bboxes)
                img = np.dstack((bgr_filled, alpha))
            else:
                img = _avg_fill_bboxes(img, all_bboxes)

            out_img_path = os.path.join(out_frames_folder, fname)
            cv2.imwrite(out_img_path, img)

        # copy labels.csv if exists at video root
        labels_src = os.path.join(folder_path, "labels.csv")
        if os.path.exists(labels_src):
            shutil.copy2(labels_src, os.path.join(out_video_folder, "labels.csv"))

        # copy BoxesCSVs folder (if exists) into output/<videoname>/BoxesCSVs (preserve contents)
        if os.path.exists(boxes_csv_folder) and os.path.isdir(boxes_csv_folder):
            dst_boxes_folder = os.path.join(out_video_folder, "BoxesCSVs")
            if os.path.exists(dst_boxes_folder):
                shutil.rmtree(dst_boxes_folder)
            shutil.copytree(boxes_csv_folder, dst_boxes_folder)

        print(f"[INFO] Finished processing {videoname}; output at {out_video_folder}")

    except Exception as e:
        print(f"[ERROR] Exception processing {folder_path}: {e}")

def process_all_folders_in_directory(input_base, output_base):
    folder_paths = [
        os.path.join(input_base, name)
        for name in os.listdir(input_base)
        if os.path.isdir(os.path.join(input_base, name))
    ]
    if not folder_paths:
        print("[WARN] No subfolders found in input base directory.")
        return
    args_list = [(folder_path, output_base) for folder_path in folder_paths]
    procs = min(cpu_count(), len(args_list))
    with Pool(processes=procs) as pool:
        pool.map(worker, args_list)
        
        
def swap_red_blue_labels(dataset_path: str):
    """
    Swaps the 'red' and 'blue' label values in each video's labels.csv file.
    """
    for video_name in sorted(os.listdir(dataset_path)):
        video_path = os.path.join(dataset_path, video_name)
        if not os.path.isdir(video_path):
            continue

        labels_csv = os.path.join(video_path, "labels.csv")
        if not os.path.exists(labels_csv):
            continue

        try:
            df = pd.read_csv(labels_csv)

            if not {'red', 'blue'}.issubset(df.columns):
                print(f"[WARN] {labels_csv} missing 'red' or 'blue', skipping.")
                continue

            # Swap columns
            df[['red', 'blue']] = df[['blue', 'red']]

            df.to_csv(labels_csv, index=False)

        except Exception as e:
            print(f"[ERR] Failed in {labels_csv}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: datasetPreparation.py <input_base_dir> <output_base_dir> ")
        sys.exit(1)
    input_base = sys.argv[1]
    output_base = sys.argv[2]
    process_all_folders_in_directory(input_base, output_base)
    swap_red_blue_labels(output_base)

