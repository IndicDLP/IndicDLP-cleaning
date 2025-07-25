import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sklearn.metrics import average_precision_score, precision_score, recall_score

NUM_CORES = max(1, cpu_count() - 1)

class DocumentMAP:
    def __init__(self, image_folder):
        self.image_folder = image_folder

    def load_annotations(self, file_path):
        """Load YOLO-style annotations with confidence scores if available."""
        if not os.path.exists(file_path):
            return []
        annotations = []
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id, x_center, y_center, width, height = map(float, parts[:5])
                confidence = float(parts[5]) if len(parts) > 5 else 1.0  # Default confidence 1.0 if not provided
                annotations.append((int(class_id), x_center, y_center, width, height, confidence))
        return annotations

    def yolo_to_absolute(self, annotations, img_width, img_height):
        """Convert YOLO format to absolute coordinates."""
        return [
            (
                class_id,
                int((x_center - width / 2) * img_width),
                int((y_center - height / 2) * img_height),
                int((x_center + width / 2) * img_width),
                int((y_center + height / 2) * img_height),
                confidence
            )
            for class_id, x_center, y_center, width, height, confidence in annotations
        ]

    def find_image_path(self, base_name):
        """Find the image file with different possible extensions."""
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = os.path.join(self.image_folder, base_name + ext)
            if os.path.exists(img_path):
                return img_path
        return None

    def load_image_size(self, base_name):
        """Load image dimensions (width, height)."""
        img_path = self.find_image_path(base_name)
        if not img_path:
            raise FileNotFoundError(f"Image for {base_name} not found.")
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image {img_path}")
            
        return img.shape[1], img.shape[0]

    def compute_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def calculate_ap(self, gt_boxes, pred_boxes, iou_threshold):
        """Calculate AP, precision and recall for a single class using confidence scores."""
        if not gt_boxes or not pred_boxes:
            return 0.0, 0.0, 0.0  # AP, precision, recall
        # hierarchical_classes = {11, 14, 19, 20, 29, 33, 31, 35, 10, 26}
        y_true, y_scores = [], []
        matched_gts = set()

        for pred in pred_boxes:
            class_id, *pred_box = pred
            best_iou, best_gt_idx = 0, -1
            # iou_thresh = 0.3 if class_id in hierarchical_classes else iou_threshold


        # Sort predictions by confidence (descending)
        pred_boxes = sorted(pred_boxes, key=lambda x: x[-1], reverse=True)
        
        y_true = []
        y_scores = []
        matched_gts = set()

        for pred in pred_boxes:
            class_id, *pred_box, confidence = pred
            best_iou, best_gt_idx = 0, -1

            for gt_idx, gt in enumerate(gt_boxes):
                gt_class, *gt_box, _ = gt
                if gt_class != class_id:
                    continue

                iou = self.compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gt_idx

            if best_iou >= iou_threshold and best_gt_idx not in matched_gts:
                y_true.append(1)
                matched_gts.add(best_gt_idx)
            else:
                y_true.append(0)
            y_scores.append(confidence)

        if not y_true:  # No matches at all
            return 0.0, 0.0, 0.0

        ap = average_precision_score(y_true, y_scores)
        precision = precision_score(y_true, [1 if x > 0.5 else 0 for x in y_scores], zero_division=0)
        recall = recall_score(y_true, [1 if x > 0.5 else 0 for x in y_scores], zero_division=0)
        
        return ap, precision, recall

    def compute_map(self, gt_file, pred_file, base_name, iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
        """Compute mAP and metrics for multiple IoU thresholds."""
        try:
            gt_annotations = self.load_annotations(gt_file)
            pred_annotations = self.load_annotations(pred_file)
            img_width, img_height = self.load_image_size(base_name)

            gt_boxes = self.yolo_to_absolute(gt_annotations, img_width, img_height)
            pred_boxes = self.yolo_to_absolute(pred_annotations, img_width, img_height)

            # Add default confidence of 1.0 to ground truth boxes
            gt_boxes = [(*box[:5], 1.0) if len(box) == 5 else box for box in gt_boxes]
            
            unique_classes = set([gt[0] for gt in gt_boxes] + [pred[0] for pred in pred_boxes])

            results = []
            for iou_threshold in iou_thresholds:
                ap_values = []
                precision_values = []
                recall_values = []
                
                for cls in unique_classes:
                    cls_gt = [gt for gt in gt_boxes if gt[0] == cls]
                    cls_pred = [pred for pred in pred_boxes if pred[0] == cls]
                    ap, precision, recall = self.calculate_ap(cls_gt, cls_pred, iou_threshold)
                    ap_values.append(float(ap))  # Convert numpy to native float
                    precision_values.append(float(precision))
                    recall_values.append(float(recall))

                results.append({
                    'iou_threshold': iou_threshold,
                    'mAP': float(np.mean(ap_values)) if ap_values else 0.0,
                    'mean_precision': float(np.mean(precision_values)) if precision_values else 0.0,
                    'mean_recall': float(np.mean(recall_values)) if recall_values else 0.0,
                    'ap_values': [float(x) for x in ap_values],  # Convert all to native float
                    'precision_values': [float(x) for x in precision_values],
                    'recall_values': [float(x) for x in recall_values],
                    'classes': list(unique_classes)
                })

            return results
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
            return None

# [Rest of the code remains the same until results processing...]
# Configuration
image_folder = "/home/sahithi_kukkala/sahithi/indicDLP/data/indic_data/images/train"
gt_folder = "/home/sahithi_kukkala/sahithi/indicDLP/data/indic_data/indic_train_labels/train_ground_truths"
pred_folder = "/home/sahithi_kukkala/sahithi/indicDLP/data/indic_data/indic_train_labels/train_result_labels2"
output_csv = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/dmAP_all_clean.csv"

# Initialize class
doc_map = DocumentMAP(image_folder)

# List all GT files
gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".txt")])

def process_single_file(gt_file):
    """Function to process a single file (used for multiprocessing)."""
    pred_file = os.path.join(pred_folder, gt_file)
    gt_file_path = os.path.join(gt_folder, gt_file)
    base_name = os.path.splitext(gt_file)[0]

    if not os.path.exists(pred_file):
        print(f"Warning: Prediction file missing for {gt_file}")
        return None

    try:
        mAP_results = doc_map.compute_map(gt_file_path, pred_file, base_name)
        if mAP_results is None:
            return None
            
        return (base_name, mAP_results)
    except Exception as e:
        print(f"Error processing {gt_file}: {str(e)}")
        return None

# [Previous imports and class definition remain the same until the results processing]

if __name__ == "__main__":
    with Pool(NUM_CORES) as pool:
        results = list(tqdm(pool.imap(process_single_file, gt_files), total=len(gt_files), desc="Computing mAP"))

    # Process results into the desired wide format
    output_data = []
    for res in results:
        if res is None:
            continue
            
        base_name, mAP_results = res
        
        # Initialize row with image name
        row = {'image_name': base_name}
        
        # Add metrics for each IoU threshold
        for threshold_metrics in mAP_results:
            iou = threshold_metrics['iou_threshold']
            row[f'mAP_{int(iou*10)}'] = round(threshold_metrics['mAP'], 4)
            row[f'Precision_{int(iou*10)}'] = round(threshold_metrics['mean_precision'], 4)
            row[f'Recall_{int(iou*10)}'] = round(threshold_metrics['mean_recall'], 4)
        
        output_data.append(row)

    # Create DataFrame with consistent columns (in case some files are missing)
    columns = ['image_name']
    for iou in [5, 6, 7, 8, 9]:
        columns.extend([f'mAP_{iou}', f'Precision_{iou}', f'Recall_{iou}'])
    
    df = pd.DataFrame(output_data, columns=columns)
    
    # Fill any missing values with 0 (if some IoUs weren't computed)
    df = df.fillna(0)
    
    # Save to CSV with tab separator
    df.to_csv(output_csv, sep='\t', index=False, float_format='%.4f')
    print(f"Results saved to {output_csv}")