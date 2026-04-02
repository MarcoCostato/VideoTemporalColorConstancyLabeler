import sys
import os
import pandas as pd
import numpy as np

def angular_distance_deg(v1, v2):
    """Calculate the angular distance in degrees between two 3D vectors."""
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)


def remove_outliers_from_triplets(frame_labels, window=2, threshold_deg=2.0):
    """Remove outliers from a list of (frame, label) tuples based on angular distance to neighbors."""
    cleaned = frame_labels.copy()
    n = len(frame_labels)
    hole_length = 0
    just_exited_hole = False
    removed_count = 0

    for i in range(n):
        frame, label = frame_labels[i]
        
        # If current label is None, we are in a hole. Increase hole length and skip outlier check.
        if label is None:
            hole_length += 1
            just_exited_hole = False
            continue

        if hole_length >= 30:
            just_exited_hole = True
        else:
            just_exited_hole = False
        hole_length = 0  # reset hole length since current is not None

        # gather neighbors
        neighbors = []
        for offset in range(1, window + 1):
            if i - offset >= 0 and frame_labels[i - offset][1] is not None:
                neighbors.append(frame_labels[i - offset][1])
            if i + offset < n and frame_labels[i + offset][1] is not None:
                neighbors.append(frame_labels[i + offset][1])

        if not neighbors:
            continue  # No context to judge
        # Compute average angular distance to neighbors
        dists = [angular_distance_deg(label, n_label) for n_label in neighbors]
        avg_dist = np.mean(dists)
        # If average distance exceeds threshold, mark as outlier (set label to None, to be interpolated later)
        if avg_dist > threshold_deg:
            if not just_exited_hole:
                print(f"[OUTLIER] Frame '{frame}' removed due to high angular distance ({avg_dist:.1f}°)")
                cleaned[i] = (frame, None)
                removed_count += 1

    return cleaned, removed_count


def process_labels_file(labels_path, window=2, threshold_deg=2.0):
    """Process a single labels.csv file, remove outliers, and save back."""
    if not os.path.exists(labels_path):
        print(f"[ERROR] Labels file not found: {labels_path}")
        return 0, 0

    # Load CSV
    df = pd.read_csv(labels_path)
    if not {"frame", "red", "green", "blue"}.issubset(df.columns):
        print(f"[ERROR] Invalid labels file format: {labels_path}")
        return 0, 0

    # Convert into list format, skip rows with empty labels
    frame_labels = []
    skipped_empty = 0
    for _, row in df.iterrows():
        frame = row["frame"]
        if pd.isna(row["red"]) or pd.isna(row["green"]) or pd.isna(row["blue"]):
            skipped_empty += 1
            continue  # skip entirely
        label = [row["red"], row["green"], row["blue"]]
        frame_labels.append((frame, label))

    total_labels = len(frame_labels) + skipped_empty

    # Run outlier removal
    cleaned, removed_count = remove_outliers_from_triplets(frame_labels, window=window, threshold_deg=threshold_deg)

    # Only keep rows with valid labels (already guaranteed by frame_labels)
    cleaned_filtered = [(frame, label) for frame, label in cleaned if label is not None]

    # Save back to CSV
    with open(labels_path, "w") as f:
        f.write("frame,red,green,blue\n")
        for frame, label in cleaned_filtered:
            f.write(f"{frame},{label[0]},{label[1]},{label[2]}\n")

    print(f"[INFO] Cleaned labels saved back to {labels_path} "
          f"({removed_count + skipped_empty}/{total_labels} removed or empty)")

    return removed_count + skipped_empty, total_labels




def process_all_labels_in_directory(root_path, window=2, threshold_deg=2.0):
    """Process all labels.csv files in subdirectories of the given root path."""
    # Find all subdirectories
    folder_paths = [
        os.path.join(root_path, name)
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

    total_removed = 0
    total_labels = 0
    # Process each labels.csv file found in the subdirectories
    for folder_path in folder_paths:
        labels_path = os.path.join(folder_path, "labels.csv")
        if os.path.exists(labels_path):
            print(f"[INFO] Processing {labels_path}")
            removed, labels = process_labels_file(labels_path, window, threshold_deg)
            total_removed += removed
            total_labels += labels
        else:
            print(f"[WARN] No labels.csv in {folder_path}")

    print(f"\n[SUMMARY] Removed {total_removed} outliers out of {total_labels} total labels "
          f"({(total_removed/total_labels*100 if total_labels else 0):.1f}%).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python outlierRemoval.py <path_to_base_directory>")
        sys.exit(1)

    base_path = sys.argv[1]
    process_all_labels_in_directory(base_path, window=1, threshold_deg=2)
