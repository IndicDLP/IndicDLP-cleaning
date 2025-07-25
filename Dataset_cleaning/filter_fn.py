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

def find_missed_gt_boxes(gt_folder, pred_folder, image_folder, output_csv, hierarchical_classes, low_iou_threshold=0.4, default_iou_threshold=0.4):
    """Find ground truth boxes that have no corresponding predictions and save results to CSV using multiprocessing."""
    gt_files = [f for f in sorted(os.listdir(gt_folder)) if f.endswith(".txt")]
    
    with Pool(cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap(process_file, [(gt_file, gt_folder, pred_folder, image_folder, default_iou_threshold, hierarchical_classes, low_iou_threshold) for gt_file in gt_files]), total=len(gt_files), desc="Processing files"))
    
    # Flatten the results (list of lists) and get unique image names
    missed_gt_images = set(item for sublist in results for item in sublist)
    
    # Save the unique image names to the CSV
    df = pd.DataFrame(list(missed_gt_images), columns=["image_name"])
    df.to_csv(output_csv, index=False)
    print(f"Missed ground truth images saved to {output_csv}")

# Define folder paths
image_folder = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/images"
gt_folder = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/gt"
pred_folder = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/predictions"
output_csv = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/fn_imgs_sample.csv"

# Define hierarchical classes
hierarchical_classes = {11, 14, 19, 20, 29, 33, 31, 35, 10, 26}

# Run the function
find_missed_gt_boxes(gt_folder, pred_folder, image_folder, output_csv, hierarchical_classes)
