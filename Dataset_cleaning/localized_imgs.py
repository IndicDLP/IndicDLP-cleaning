import os
import pandas as pd


import os
import cv2
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def load_annotations(file_path):
    """Load YOLO-style annotations from a text file."""
    if not os.path.exists(file_path):
        return []
    annotations = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts[:5])
            annotations.append((int(class_id), x_center, y_center, width, height))
    return annotations

def yolo_to_absolute(annotations, img_width, img_height):
    """Convert YOLO format to absolute coordinates."""
    return [
        (
            class_id,
            int((x_center - width / 2) * img_width),
            int((y_center - height / 2) * img_height),
            int((x_center + width / 2) * img_width),
            int((y_center + height / 2) * img_height)
        )
        for class_id, x_center, y_center, width, height in annotations
    ]

def compute_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def process_file(args):
    """Process a single file to find missed ground truth boxes."""
    gt_file, gt_folder, pred_folder, image_folder, default_iou_threshold, hierarchical_classes, low_iou_threshold = args
    base_name = os.path.splitext(gt_file)[0]
    gt_path = os.path.join(gt_folder, gt_file)
    pred_path = os.path.join(pred_folder, gt_file)
    img_path = next((os.path.join(image_folder, base_name + ext) for ext in [".png", ".jpg", ".jpeg"] if os.path.exists(os.path.join(image_folder, base_name + ext))), None)
    
    if not img_path:
        print(f"Warning: Image for {base_name} not found.")
        return []
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image {img_path}")
        return []
    
    img_width, img_height = img.shape[1], img.shape[0]
    gt_boxes = yolo_to_absolute(load_annotations(gt_path), img_width, img_height)
    pred_boxes = yolo_to_absolute(load_annotations(pred_path), img_width, img_height) if os.path.exists(pred_path) else []
    
    missed_gt_images = set()  # Use a set to avoid duplicate images
    for gt in gt_boxes:
        gt_class, *gt_box = gt
        
        # Use a different IoU threshold for hierarchical classes
        iou_threshold = low_iou_threshold if gt_class in hierarchical_classes else default_iou_threshold
        
        best_iou = max((compute_iou(gt_box, pred_box[1:]) for pred_box in pred_boxes if pred_box[0] == gt_class), default=0)
        if best_iou < iou_threshold:
            missed_gt_images.add(base_name)
    
    return list(missed_gt_images)


import pandas as pd

def find_badly_localized_images(map_csv, output_csv, drop_threshold=0.3):
    """
    Identify badly localized images based on high mAP drop across IoU thresholds.

    Parameters:
    - map_csv: Path to CSV file containing mAP values at different IoUs.
    - output_csv: Path to save badly localized image names.
    - drop_threshold: Minimum drop in mAP between IoU=0.5 and higher IoUs to consider as badly localized.
    """
    # Load the mAP CSV file
    df = pd.read_csv(map_csv)

    # Ensure necessary columns exist
    required_columns = ["image_name", "mAP_0.5", "mAP_0.6", "mAP_0.7", "mAP_0.8", "mAP_0.9"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")

    # Compute the drop in mAP between consecutive IoU thresholds
    df["mAP_drop_0.5_to_0.6"] = df["mAP_0.5"] - df["mAP_0.6"]
    df["mAP_drop_0.5_to_0.7"] = df["mAP_0.6"] - df["mAP_0.7"]
    df["mAP_drop_0.5_to_0.8"] = df["mAP_0.7"] - df["mAP_0.8"]
    df["mAP_drop_0.5_to_0.9"] = df["mAP_0.8"] - df["mAP_0.9"]

    # Identify images where mAP drop is significant
    badly_localized = df[
        (df["mAP_drop_0.5_to_0.6"] > drop_threshold) | 
        (df["mAP_drop_0.5_to_0.7"] > drop_threshold) | 
        (df["mAP_drop_0.5_to_0.8"] > drop_threshold) 
        
    ]["image_name"]

    # Save results to CSV
    badly_localized_df = pd.DataFrame(badly_localized, columns=["image_name"])
    badly_localized_df.to_csv(output_csv, index=False)

    print(f"Badly localized images saved to {output_csv}")

# Define paths
map_csv = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/map_filtered_.8.csv"
output_csv = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/badly_localized_images_.8.csv"

# Run the function
find_badly_localized_images(map_csv, output_csv)
