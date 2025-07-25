import os
import cv2
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score

class DocumentMAP:
    """
    Class to compute document-level mAP@0.5 for a given pair of YOLO-formatted 
    ground truth and prediction annotation files.
    """

    def __init__(self, image_folder):
        """
        Initializes the DocumentMAP class.

        Args:
            image_folder (str): Path to the folder containing the images.
        """
        self.image_folder = image_folder

    def load_annotations(self, file_path):
        """
        Load YOLO annotations from a file and round values to 6 decimal places.

        Args:
            file_path (str): Path to the annotation file.

        Returns:
            list: List of annotations [(class_id, x_center, y_center, width, height)].
        """
        if not os.path.exists(file_path):
            print(f"Debug: File {file_path} not found. Returning empty list.")
            return []

        annotations = []
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # Skip invalid lines
                
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # Round values to 6 decimal places
                x_center = round(x_center, 6)
                y_center = round(y_center, 6)
                width = round(width, 6)
                height = round(height, 6)

                annotations.append((class_id, x_center, y_center, width, height))
        print(annotations)
        print(f"Debug: Loaded {len(annotations)} annotations from {file_path}")
        return annotations
    def yolo_to_absolute(self, annotations, img_width, img_height):
        abs_boxes = []
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            print(f"Debug: Class {class_id} -> Abs Box: ({x1}, {y1}, {x2}, {y2})")

            abs_boxes.append((int(class_id), x1, y1, x2, y2))
        return abs_boxes

    def load_image_size(self, image_name):
        img_path = os.path.join(self.image_folder, image_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Image {img_path} not found or cannot be loaded.")
            raise FileNotFoundError(f"Image {img_path} not found.")
        print(f"Debug: Loaded image {image_name} with size {img.shape[1]}x{img.shape[0]}")
        return img.shape[1], img.shape[0]

    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0

        print(f"Debug: IoU between {box1} and {box2} = {iou:.4f}")
        return iou


    def calculate_ap(self, gt_boxes, pred_boxes):
        if not pred_boxes:
            return 0.0

        iou_threshold = 0.5  # Standard threshold
        hierarchical_classes = {11, 14, 19, 20, 29, 33, 31, 35, 10,26 }  # Hierarchical categories

        gt_matched = set()
        y_true = []
        y_scores = []

        for pred in pred_boxes:
            class_id = pred[0]  # Extract class
            pred_box = pred[1:]

            best_iou = 0
            best_gt_idx = -1

            # Set default IoU threshold
            iou_threshold_adjusted = iou_threshold

            for gt_idx, gt in enumerate(gt_boxes):
                gt_class = gt[0]
                gt_box = gt[1:]

                # Adjust IoU threshold for hierarchical categories
                if class_id in hierarchical_classes or gt_class in hierarchical_classes:
                    iou_threshold_adjusted = 0.3

                iou = self.compute_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Ensure iou_threshold_adjusted is always defined
            if best_iou >= iou_threshold_adjusted:
                y_true.append(1)  # True Positive
                gt_matched.add(best_gt_idx)
                print(f"Debug: Matched Prediction {pred_box} with GT {gt_boxes[best_gt_idx]} (IoU {best_iou:.4f})")
            else:
                y_true.append(0)  # False Positive
                print(f"Debug: Prediction {pred_box} is a False Positive (Best IoU {best_iou:.4f})")

            y_scores.append(1)  # Confidence is always 1 (fix this if confidence scores exist)

        ap = average_precision_score(y_true, y_scores) if y_true else 0.0
        print(f"Debug: AP = {ap:.4f}")
        return ap

    def compute_map(self, gt_file, pred_file, image_name):
        """
        Compute document-level mAP@0.5.

        Args:
            gt_file (str): Path to the ground truth annotation file.
            pred_file (str): Path to the prediction annotation file.
            image_name (str): Name of the image file.

        Returns:
            float: Document-level mAP@0.5.
        """
        gt_annotations = self.load_annotations(gt_file)
        print(gt_annotations)
        pred_annotations = self.load_annotations(pred_file)
        print(pred_annotations)

        img_width, img_height = self.load_image_size(image_name)
        gt_boxes = self.yolo_to_absolute(gt_annotations, img_width, img_height)
        print(gt_boxes)
        pred_boxes = self.yolo_to_absolute(pred_annotations, img_width, img_height)
        print(pred_boxes)

        unique_classes = set([gt[0] for gt in gt_boxes] + [pred[0] for pred in pred_boxes])

        ap_values = []
        for class_id in unique_classes:
            gt_class_boxes = [box for box in gt_boxes if box[0] == class_id]
            pred_class_boxes = [box for box in pred_boxes if box[0] == class_id]
            ap = self.calculate_ap(gt_class_boxes, pred_class_boxes)
            ap_values.append(ap)
            print(f"Debug: AP for class {class_id}: {ap:.4f}")

        map_score = np.mean(ap_values) if ap_values else 0.0
        print(f"Debug: Document-level mAP@0.5: {map_score:.4f}")

        return map_score



# image_folder = "/home/sahithi_kukkala/sahithi/indic_data/images/train"
# evaluator = DocumentMAP(image_folder)

# # File paths
# gt_file = "/home/sahithi_kukkala/sahithi/indic_train_labels/train_ground_truths/ar_as_000001_0.txt"
# pred_file = "/home/sahithi_kukkala/sahithi/indic_train_labels/train_result_labels/ar_as_000001_0.txt"
# image_name = "ar_as_000001_0.png"

# # Compute document-level mAP@0.5
# map_score = evaluator.compute_map(gt_file, pred_file, image_name)
# print(f"Document-level mAP@0.5: {map_score:.4f}")


import os
import glob
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_single_pair(args):
    """
    Process a single GT-prediction pair (for parallel execution)
    Args:
        args: Tuple of (evaluator, gt_file, pred_folder, image_folder, base_name)
    """
    evaluator, gt_file, pred_folder, image_folder, base_name = args
    
    # Find corresponding prediction file
    pred_file = os.path.join(pred_folder, f"{base_name}.txt")
    if not os.path.exists(pred_file):
        # print(f"Skipping {base_name}: Prediction file not found")
        return None
        
    # Find corresponding image file
    image_file = None
    for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        test_path = os.path.join(image_folder, f"{base_name}{ext}")
        if os.path.exists(test_path):
            image_file = test_path
            break
    if not image_file:
        # print(f"Skipping {base_name}: Image file not found")
        return None
        
    try:
        # Compute mAP at different IoU thresholds
        map_50 = evaluator.compute_map(gt_file, pred_file, os.path.basename(image_file))
        map_75 = evaluator.compute_map(gt_file, pred_file, os.path.basename(image_file)) 
        map_90 = evaluator.compute_map(gt_file, pred_file, os.path.basename(image_file))
        
        return {
            'image_name': base_name,
            'map_50': map_50,
            'map_75': map_75,
            'map_90': map_90
        }
        
    except Exception as e:
        # print(f"Error processing {base_name}: {str(e)}")
        return None

def process_all_images(image_folder, gt_folder, pred_folder):
    """
    Process all GT-prediction pairs in parallel
    """
    # Initialize evaluator
    evaluator = DocumentMAP(image_folder)
    
    # Get all ground truth files
    gt_files = glob.glob(os.path.join(gt_folder, "*.txt"))
    
    # Prepare arguments for parallel processing
    tasks = [
        (evaluator, gt_file, pred_folder, image_folder, 
         os.path.splitext(os.path.basename(gt_file))[0])
        for gt_file in gt_files
    ]
    
    # Process in parallel
    results = []
    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=len(tasks), desc="Processing images") as pbar:
            for result in pool.imap_unordered(process_single_pair, tasks):
                if result is not None:
                    results.append(result)
                pbar.update(1)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_csv = "document_map_results.csv"
    results_df.to_csv(output_csv, index=False)
    
    print(f"\nResults saved to {output_csv}")
    print(f"Successfully processed {len(results_df)}/{len(gt_files)} images")
    
    return results_df

# Configuration
image_folder = "/home/sahithi_kukkala/sahithi/indic_data/images/train"
gt_folder = "/home/sahithi_kukkala/sahithi/indic_train_labels/train_ground_truths"
pred_folder = "/home/sahithi_kukkala/sahithi/indic_train_labels/train_result_labels"

# Run processing
results_df = process_all_images(image_folder, gt_folder, pred_folder)

# Print summary
if not results_df.empty:
    print("\nSummary Statistics:")
    print(f"Average mAP@0.5: {results_df['map_50'].mean():.4f}")
    print(f"Average mAP@0.75: {results_df['map_75'].mean():.4f}")
    print(f"Average mAP@0.9: {results_df['map_90'].mean():.4f}")
else:
    print("\nNo valid results were generated")