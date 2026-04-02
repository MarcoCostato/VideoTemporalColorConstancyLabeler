import argparse 
import cv2
import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir))  # ColorChartLocalization
sys.path.insert(0, os.path.join(current_dir, "ultralytics"))  # optional
sys.path.insert(0, os.path.join(current_dir, "find_patches"))
from ultralytics import YOLO
import matplotlib.pyplot as plt
from find_patches.findpatches import findpatches, get_triplets
from find_patches.det_and_seg import detect, segment
import pandas as pd
import warnings

import ipdb


def annotate(image_path, out_file, segment_background, device, get_viz):
    patches_names = ["Dark Skin", "Light Skin", "Blue Sky", "Foliage", "Blue Flower", "Bluish Green", 
                     "Orange", "Purplish Blue", "Moderate Red", "Purple", "Yellow Green", "Orange Yellow", 
                     "Blue", "Green", "Red", "Yellow", "Magenta", "Cyan", 
                     "White","Neutral 8", "Neutral 6.5", "Neutral 5", "Neutral 3.5", "Black"]


    out_file_name = out_file if out_file is not None else image_path.split(".")[0] + ".csv"
    
    detector = YOLO(r'C:\Users\marco\Desktop\TesiAWB\labeling\ColorChartLocalization\ultralytics\runs\detect\train\weights\best.pt') # Initialize the detection model
    segmenter = YOLO(r'C:\Users\marco\Desktop\TesiAWB\labeling\ColorChartLocalization\ultralytics\runs\detect\train\weights\best.pt') if segment_background else None # Initialize the segmentation model if needed

    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    crops = detect(img, detector, device) # Get the predictions of the detection model

    if len(crops) == 0:
        raise RuntimeError("Error finding patches")
    
    
    #crops = [crops[0]] # For now, only use the first prediction 


    out = []
    for c in crops:
        crop = c[0]
        displacement = c[1]
        y1,x1 = displacement
        y2, x2 = y1 + crop.shape[0], x1 + crop.shape[1]

        try:
            mask = segment(crop, segmenter, device) if segment_background else None # Segment the image if needed
            # # Find the patches
            patches, radius, viz = findpatches(crop, mask=mask, get_viz = get_viz, t1=80, t2=170, tolerance=np.pi/9) # Find the patches in the image

            if get_viz:
                # plt.imshow(viz[:,:,::-1])
                # plt.show()
                viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                viz_folder = os.path.join(os.path.dirname(out_file_name), "viz_images")
                os.makedirs(viz_folder, exist_ok=True)
                viz_path = os.path.join(viz_folder, f"{base_name}_viz.jpg")
                
                cv2.imwrite(viz_path, viz) # Save the visualization image
        
        except:
            raise RuntimeError("Error finding patches")

        r  = radius*np.max(crop.shape[:2])/256
        patches = [p["coords"]*np.max(crop.shape[:2])/256 + np.array(displacement) for p in patches] # Get the coordinates of the patches and convert them to the original image coordinates
        
        rgb_triplets = get_triplets(img, patches, r) # Get the rgb triplets of the patches

        rgb_triplets = np.array(rgb_triplets).reshape(-1, 3) # Reshape the rgb triplets to a 2D array
        row = {}

        for i, triplet in enumerate(rgb_triplets):
            row[f"{patches_names[i]}"] = triplet.tolist()
            if patches_names[i] == "White":
                # patches[i] is top-left (y, x)
                y_top_left, x_top_left = patches[i]
                row["White_coords"] = [x_top_left, y_top_left]
                row["White_radius"] = r


        row["bbox_y1"] = int(y1)
        row["bbox_x1"] = int(x1)
        row["bbox_y2"] = int(y2)
        row["bbox_x2"] = int(x2)

        
        out.append(row) # Append the row to the output list

    df = pd.DataFrame(out) # Create a dataframe from the output list
    df.to_csv(out_file_name, index=False) # Save the dataframe to a csv file
    
    
def save_detected_boxes(image_path, out_dir, device="cpu"):
    """
    Runs YOLO detection on an image and saves bounding boxes in CSV.
    Only bbox_y1, bbox_x1, bbox_y2, bbox_x2 are saved.
    
    """
    detector = YOLO(r'C:\Users\marco\Desktop\TesiAWB\labeling\ColorChartLocalization\ultralytics\runs\detect\train\weights\best.pt')
    os.makedirs(out_dir, exist_ok=True)

    # Read image
    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"Could not read image {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run detection
    crops = detect(img_rgb, detector, device)  # Your existing detect() method
    if len(crops) == 0:
        print(f"[WARN] No detections found for {image_path}")
        return

    out_rows = []
    for c in crops:
        _, displacement = c
        y1, x1 = displacement
        h, w = c[0].shape[:2]
        y2 = y1 + h
        x2 = x1 + w

        out_rows.append({
            "bbox_y1": int(y1),
            "bbox_x1": int(x1),
            "bbox_y2": int(y2),
            "bbox_x2": int(x2)
        })

    # Build CSV path
    base_name = os.path.splitext(os.path.basename(image_path))[0] + ".csv"
    out_file = os.path.join(out_dir, base_name)

    # Save CSV
    pd.DataFrame(out_rows).to_csv(out_file, index=False)
    print(f"[INFO] Saved: {out_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python save_boxes.py <images_folder> <weights_path>")
        sys.exit(1)

    images_folder = sys.argv[1]
    weights_path = sys.argv[2]

    detector = YOLO(weights_path)

    # Walk through frames folder and save boxes in BoxesCSVs
    for root, _, files in os.walk(images_folder):
        if os.path.basename(root) == "frames":
            boxes_csv_folder = os.path.join(os.path.dirname(root), "CSVs", "BoxesCSVs")
            for fname in files:
                if fname.lower().endswith(".png"):
                    image_path = os.path.join(root, fname)
                    save_detected_boxes(image_path, boxes_csv_folder, detector, device="cpu")

if __name__ == "__main__":
    
    # USAGE EXAMPLE: # python annotate.py path/to/image.jpg --segment_background --device cuda --get_viz
    
    
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description="Annotate an image with patches")
    parser.add_argument("image_path", type=str, help="Path to the image to annotate")
    parser.add_argument("--segment_background", action="store_true", help="Segment the background of the image")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for the models (cpu or cuda)")
    parser.add_argument("--get_viz", action="store_true", help="Get visualization of the patches")
    parser.add_argument("--out_file",  default=None, help="Output file to save the annotations")
    args = parser.parse_args()
    
    rgb_triplets = annotate(args.image_path, args.out_file, args.segment_background, args.device, args.get_viz)
    



        


    