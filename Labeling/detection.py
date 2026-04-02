import sys
import os
import re
import pandas as pd
import cv2
from multiprocessing import Pool, cpu_count
from ultralytics import YOLO


# --- Helpers ---

def _parse_frame_index(frame_name):
    digits = re.findall(r'(\d+)', frame_name)
    if not digits:
        raise ValueError(f"No digits found in frame name '{frame_name}'")
    return int(digits[-1])

def _format_frame_name(idx, pad):
    return f"frame_{idx:0{pad}d}"


def detect_objects_yolo(image_path, model, device="cpu"):
    """
    Runs YOLO object detection on a single image.
    Returns detections as a list of dicts with keys: x1, y1, x2, y2.
    """
    results = model.predict(source=image_path, device=device, verbose=False)
    detections = []
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return detections


def draw_boxes(image_path, detections, out_path):
    """
    Draws bounding box perimeters (not filled) around detections and saves the image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def worker(args):
    """Worker function to process a single folder."""
    folder_path, device, model_path, label = args
    try:
        print(f"[INFO] Processing: {folder_path}")

        # Build paths using the provided label (e.g. "faces", "plates", "cars")
        label_lower  = label.lower()
        label_titled = label.capitalize()

        frame_folder     = os.path.join(folder_path, "processed_labeled_RGB_png", "frames")
        csv_folder       = os.path.join(folder_path, "processed_labeled_RGB_png", "CSVs")
        output_csv_dir   = os.path.join(csv_folder,   f"{label_titled}CSVs")
        output_img_dir   = os.path.join(frame_folder, label_lower)

        if not os.path.exists(frame_folder):
            print(f"[WARN] No frames folder found: {frame_folder}")
            return

        os.makedirs(output_csv_dir, exist_ok=True)
        os.makedirs(output_img_dir, exist_ok=True)

        # Load YOLO model (once per worker process)
        model = YOLO(model_path)

        # Process each PNG frame
        for fname in sorted(os.listdir(frame_folder)):
            if not fname.lower().endswith(".png"):
                continue

            frame_path = os.path.join(frame_folder, fname)
            detections = detect_objects_yolo(frame_path, model, device=device)

            # Save detections as CSV
            out_csv = os.path.join(output_csv_dir, fname.replace(".png", ".csv"))
            pd.DataFrame(detections).to_csv(out_csv, index=False)

            # Save annotated image
            out_img = os.path.join(output_img_dir, fname)
            draw_boxes(frame_path, detections, out_img)

        print(f"[INFO] Done: {folder_path}")

    except Exception as e:
        print(f"[ERROR] Failed on {folder_path}: {e}")


def process_all_folders_in_directory(root_path, model_path, label, device="cpu"):
    """
    Spawns one worker process per sub-folder found in root_path.
    """
    folder_paths = [
        os.path.join(root_path, name)
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

    if not folder_paths:
        print(f"[WARN] No sub-folders found in: {root_path}")
        return

    args_list = [(fp, device, model_path, label) for fp in folder_paths]
    with Pool(processes=min(cpu_count(), len(args_list))) as pool:
        pool.map(worker, args_list)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: detection.py <base_directory> <yolo_model_path> <detection_label>")
        print("  detection_label  label for the detected object, e.g.: face  plate  car")
        print("")
        print("Examples:")
        print("  python detection.py /data/session yolov8n-face.pt face")
        print("  python detection.py /data/session yolov8n-plate.pt plate")
        sys.exit(1)

    base_dir    = sys.argv[1]
    model_path  = sys.argv[2]
    label       = sys.argv[3]

    process_all_folders_in_directory(base_dir, model_path, label, device="cuda")
