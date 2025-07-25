# main.py
import os
import json
import yaml # Requires PyYAML: pip install pyyaml
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from tqdm import tqdm # Optional progress bar: pip install tqdm

from metrics import BoundingBox
from errors import ErrorCalculator

# --- Configuration Loading ---
def load_config(config_path="config.yaml") -> Optional[Dict]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Configuration loaded successfully from {config_path}")
        # Basic validation
        required_keys = ['annotation_file', 'prediction_dir', 'output_dir',
                         'iou_threshold', 'confidence_threshold',
                         'metadata_keys', 'class_id_to_name_mapping']
        if not all(key in config for key in required_keys):
             missing = [key for key in required_keys if key not in config]
             print(f"Error: Missing required keys in config file: {missing}")
             return None
        if not isinstance(config.get('metadata_keys'), dict):
             print("Error: 'metadata_keys' in config must be a dictionary.")
             return None
        if not isinstance(config.get('class_id_to_name_mapping'), dict):
             print("Error: 'class_id_to_name_mapping' in config must be a dictionary.")
             return None
        # Convert class IDs from YAML (might be read as strings) back to int
        try:
            config['class_id_to_name_mapping'] = {int(k): v for k, v in config['class_id_to_name_mapping'].items()}
        except ValueError:
            print("Error: Class IDs in 'class_id_to_name_mapping' must be integers.")
            return None

        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file {config_path}: {e}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred loading config: {e}")
         return None


# --- Data Loading --- (load_coco_style_data, load_prediction_annotations - unchanged from previous version)
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import json
import os
from metrics import BoundingBox # Assuming metrics.py contains BoundingBox definition

def load_coco_style_data(annotation_file: str) -> Tuple[Optional[Dict[int, List[BoundingBox]]], Optional[Dict[int, Dict]]]:
    """
    Loads ground truth annotations and image metadata from a COCO-style JSON file.
    (Code from previous answer - unchanged)
    """
    gt_boxes_by_image_id = defaultdict(list)
    images_info = {}

    print(f"Loading annotation file: {annotation_file}")
    try:
        with open(annotation_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found: {annotation_file}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in annotation file {annotation_file}: {e}")
        return None, None
    except Exception as e:
        print(f"Error reading annotation file {annotation_file}: {e}")
        return None, None

    # Process images to get filenames and potentially metadata
    if 'images' not in data or not isinstance(data['images'], list):
         print(f"Error: 'images' key not found or not a list in {annotation_file}")
         return None, None
    print(f"Processing {len(data['images'])} image entries...")
    for img_info in data['images']:
         if 'id' not in img_info or 'file_name' not in img_info:
              print(f"Warning: Skipping image entry missing 'id' or 'file_name': {img_info.get('id', 'N/A')}")
              continue
         img_id = img_info['id']
         if not isinstance(img_id, int):
              print(f"Warning: Image ID is not an integer for file {img_info['file_name']}. Skipping.")
              continue
         images_info[img_id] = img_info # Store the whole dict

    # Process annotations for GT boxes
    if 'annotations' not in data or not isinstance(data['annotations'], list):
        print(f"Warning: 'annotations' key not found or not a list in {annotation_file}. No GT boxes will be loaded.")
    else:
         print(f"Processing {len(data['annotations'])} GT annotations...")
         ann_count = 0
         skipped_ann_count = 0
         # Use tqdm here if installed and desired
         for ann in data['annotations']: # Removed tqdm wrapping for simplicity if not installed
            if not all(k in ann for k in ['image_id', 'category_id', 'bbox']):
                skipped_ann_count += 1
                continue

            image_id = ann["image_id"]
            if not isinstance(image_id, int) or image_id not in images_info:
                skipped_ann_count += 1
                continue

            category_id = ann["category_id"]
            bbox = ann["bbox"]

            if not isinstance(bbox, list) or len(bbox) != 4:
                 skipped_ann_count += 1
                 continue

            try:
                # --- Assuming Pascal VOC format [x_min, y_min, x_max, y_max] ---
                x_min, y_min, x_max, y_max = map(float, bbox)
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                # --- If your bbox is COCO [x_min, y_min, width, height], use this instead: ---
                # x_min_f, y_min_f, width_f, height_f = map(float, bbox)
                # x_min, y_min = int(x_min_f), int(y_min_f)
                # x_max, y_max = int(x_min_f + width_f), int(y_min_f + height_f)
                # ---------------------------------------------------------------------

                if x_min >= x_max or y_min >= y_max or x_min < 0 or y_min < 0:
                     skipped_ann_count += 1
                     continue

                gt_box = BoundingBox(category_id, x_min, y_min, x_max, y_max, confidence=1.0)
                gt_boxes_by_image_id[image_id].append(gt_box)
                ann_count += 1

            except (ValueError, TypeError):
                skipped_ann_count += 1
                continue
         print(f"Successfully loaded {ann_count} GT annotations, skipped {skipped_ann_count}.")

    print(f"Found image info for {len(images_info)} images.")
    print(f"Loaded GT boxes for {len(gt_boxes_by_image_id)} images.")
    return dict(gt_boxes_by_image_id), images_info


def load_prediction_annotations(file_path: str) -> List[BoundingBox]:
    """
    Loads prediction annotations from a text file.
    Expected format: class_id x_min y_min x_max y_max confidence
    (Code from previous answer - unchanged)
    """
    boxes = []
    if not os.path.exists(file_path):
        return boxes
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if not parts: continue
                try:
                    if len(parts) != 6:
                        print(f"Warning: Skipping invalid prediction line {i+1} in {file_path}: {line.strip()} - Expected 6 parts")
                        continue
                    class_id = int(parts[0])
                    x_min = int(float(parts[1]))
                    y_min = int(float(parts[2]))
                    x_max = int(float(parts[3]))
                    y_max = int(float(parts[4]))
                    confidence = float(parts[5])
                    if x_min >= x_max or y_min >= y_max or not (0.0 <= confidence <= 1.0) or x_min < 0 or y_min < 0:
                         print(f"Warning: Skipping invalid prediction box data line {i+1} in {file_path}: {line.strip()}")
                         continue
                    boxes.append(BoundingBox(class_id, x_min, y_min, x_max, y_max, confidence))
                except ValueError as e:
                    print(f"Warning: Skipping line {i+1} due to ValueError in {file_path}: {line.strip()} - {e}")
                    continue
    except Exception as e:
        print(f"Error reading prediction file {file_path}: {e}")
    return boxes


# --- Main Execution ---
def main(config_path="config.yaml"):
    """
    Main function to calculate and aggregate object detection errors using config file.
    """
    config = load_config(config_path)
    if config is None:
        print("Exiting due to configuration error.")
        return

    # --- Get config values ---
    annotation_file = config['annotation_file']
    pred_dir = config['prediction_dir']
    output_dir = config['output_dir']
    iou_threshold = config['iou_threshold']
    conf_threshold = config['confidence_threshold']
    metadata_keys_config = config['metadata_keys']
    class_mapping = config['class_id_to_name_mapping'] # Already parsed IDs to int
    sort_key = config.get('sort_images_by', 'MissedDetectionRate')
    sort_desc = config.get('sort_descending', True)

    print("\n--- Starting Error Analysis ---")
    # (Print statements for config values - unchanged)
    print(f"Annotation File (GT): {annotation_file}")
    print(f"Prediction Directory: {pred_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"IoU Threshold: {iou_threshold}")
    print(f"Confidence Threshold (for FP rate): {conf_threshold}")
    print(f"Metadata Keys for Aggregation: {metadata_keys_config}")
    print(f"Class Mapping: Loaded {len(class_mapping)} classes.")
    print(f"Sorting images by: {sort_key} ({'Descending' if sort_desc else 'Ascending'})")


    # --- Initialization ---
    error_calculator = ErrorCalculator(iou_threshold=iou_threshold, conf_threshold=conf_threshold)

    gt_boxes_by_image_id, images_info = load_coco_style_data(annotation_file)

    if gt_boxes_by_image_id is None or images_info is None:
        print("Exiting due to error loading annotation data.")
        return
    if not images_info:
        print("No image information loaded from annotation file. Cannot proceed.")
        return

    # Prepare structures for results
    image_results = {} # {image_filename: {rates, counts, image_id}}
    # Overall and Metadata Aggregation (based on image-level counts)
    aggregate_counts_metadata = {
        'overall': defaultdict(float),
    }
    for group_key_name in metadata_keys_config.keys():
        if metadata_keys_config[group_key_name]:
             aggregate_counts_metadata[group_key_name] = defaultdict(lambda: defaultdict(float))

    # Class-wise Aggregation (based on individual detections)
    aggregate_counts_by_class = defaultdict(lambda: defaultdict(float))


    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # --- Process Each Image ---
    print(f"\n--- Processing {len(images_info)} Images ---")
    processed_count = 0
    missing_pred_count = 0

    # Use tqdm here if installed and desired
    for image_id, img_info in images_info.items(): # Removed tqdm wrapping for simplicity
        image_filename = img_info.get('file_name')
        if not image_filename:
            print(f"Warning: Skipping image ID {image_id} because 'file_name' is missing.")
            continue

        pred_filename_base = os.path.splitext(image_filename)[0]
        pred_filepath = os.path.join(pred_dir, f"{pred_filename_base}.txt")

        # Load GT and Predictions
        gt_boxes = gt_boxes_by_image_id.get(image_id, [])
        pred_boxes = load_prediction_annotations(pred_filepath)

        if not os.path.exists(pred_filepath):
             missing_pred_count += 1

        # --- Calculate image-level results ---
        # Get detailed matching results for the image using the internal method
        matches, unmatched_gt_indices, unmatched_pred_indices = error_calculator._match_boxes(gt_boxes, pred_boxes)

        # Calculate image summary counts using the public method
        # This is still useful for per-image reporting and overall/metadata aggregation
        image_counts = error_calculator.calculate_image_error_counts(gt_boxes, pred_boxes)
        image_rates = error_calculator.calculate_rates_from_counts(image_counts)
        image_results[image_filename] = {"image_id": image_id, "rates": image_rates, "counts": dict(image_counts)} # Convert counts defaultdict

        # --- Aggregate Overall/Metadata Counts (using image_counts) ---
        error_calculator.aggregate_counts(aggregate_counts_metadata['overall'], image_counts)
        for group_key_name, meta_key in metadata_keys_config.items():
             if meta_key and meta_key in img_info :
                 value = img_info[meta_key]
                 if value is not None and value != "":
                    if group_key_name in aggregate_counts_metadata:
                         error_calculator.aggregate_counts(aggregate_counts_metadata[group_key_name][str(value)], image_counts)


        # --- Aggregate Class-wise Counts (using detailed match results) ---
        # Denominators per class for this image
        local_gt_counts = defaultdict(int)
        local_pred_counts = defaultdict(int)
        local_high_conf_pred_counts = defaultdict(int)
        for gt_box in gt_boxes:
            local_gt_counts[gt_box.class_id] += 1
        for pred_box in pred_boxes:
            local_pred_counts[pred_box.class_id] += 1
            if pred_box.confidence >= conf_threshold:
                local_high_conf_pred_counts[pred_box.class_id] += 1

        # Add local denominator counts to global class counts
        for class_id, count in local_gt_counts.items():
             aggregate_counts_by_class[class_id]['total_gt'] += count
        for class_id, count in local_pred_counts.items():
             aggregate_counts_by_class[class_id]['total_pred'] += count
        for class_id, count in local_high_conf_pred_counts.items():
             aggregate_counts_by_class[class_id]['total_high_conf_pred'] += count

        # Numerators based on matching results for this image
        # True Positives (Correctly matched)
        for gt_idx, pred_idx, iou in matches:
            gt_class_id = gt_boxes[gt_idx].class_id
            pred_class_id = pred_boxes[pred_idx].class_id

            # Increment total matches (needed for classification error rate denominator later)
            # This count should arguably be associated with the GT class
            aggregate_counts_by_class[gt_class_id]['num_matched_pairs'] += 1


            if gt_class_id == pred_class_id: # True Positive for this class
                aggregate_counts_by_class[gt_class_id]['num_matched_correct_label'] += 1
                aggregate_counts_by_class[gt_class_id]['sum_iou_correct_matches'] += iou
            else: # Classification Error
                # Considered a classification error for the GT class
                aggregate_counts_by_class[gt_class_id]['num_matched_incorrect_label'] += 1
                # It's also effectively a False Positive for the predicted class category
                # aggregate_counts_by_class[pred_class_id]['classification_fp'] += 1 # Optional: track this if needed

        # False Negatives (Unmatched GT)
        for gt_idx in unmatched_gt_indices:
            gt_class_id = gt_boxes[gt_idx].class_id
            aggregate_counts_by_class[gt_class_id]['num_unmatched_gt'] += 1

        # False Positives (High-Conf Unmatched Preds)
        for pred_idx in unmatched_pred_indices:
            pred_box = pred_boxes[pred_idx]
            pred_class_id = pred_box.class_id
            if pred_box.confidence >= conf_threshold:
                aggregate_counts_by_class[pred_class_id]['num_high_conf_unmatched_pred'] += 1


        processed_count += 1
        # Optional: print progress every N images if tqdm is not used

    print(f"\n--- Processing Summary ---")
    print(f"Successfully processed {processed_count} images.")
    print(f"Could not find prediction files for {missing_pred_count} images.")


    # --- Calculate Final Aggregate Rates ---
    print("\n--- Calculating Aggregate Rates ---")
    aggregate_results = {}

    # Overall and Metadata-based Aggregates
    print("Calculating Overall / Metadata Aggregates...")
    aggregate_results['overall'] = {
        'counts': dict(aggregate_counts_metadata['overall']),
        'rates': error_calculator.calculate_rates_from_counts(aggregate_counts_metadata['overall'])
    }
    for group_key_name in metadata_keys_config.keys():
        if group_key_name in aggregate_counts_metadata:
            aggregate_results[group_key_name] = {}
            print(f"Aggregating for: {group_key_name}")
            for category_value, counts_dict in aggregate_counts_metadata[group_key_name].items():
                aggregate_results[group_key_name][category_value] = {
                    'counts': dict(counts_dict),
                    'rates': error_calculator.calculate_rates_from_counts(counts_dict)
                }
            print(f"  Found {len(aggregate_results[group_key_name])} unique values.")

    # Class-wise Aggregates
    print("Calculating Class-wise Aggregates...")
    aggregate_results['class'] = {}
    # Sort class results by class ID for consistent output order
    sorted_class_ids = sorted(aggregate_counts_by_class.keys())
    for class_id in sorted_class_ids:
        counts = aggregate_counts_by_class[class_id]
        # Ensure all required keys for rate calculation exist, defaulting to 0
        required_rate_keys = ['total_gt', 'total_pred', 'total_high_conf_pred',
                              'num_matched_pairs', 'num_matched_correct_label',
                              'num_matched_incorrect_label', 'num_unmatched_gt',
                              'num_unmatched_pred', 'num_high_conf_unmatched_pred',
                              'sum_iou_correct_matches']
        final_counts_for_class = {k: counts.get(k, 0.0) for k in required_rate_keys}

        rates = error_calculator.calculate_rates_from_counts(final_counts_for_class)
        class_name = class_mapping.get(class_id, f"Unknown ID: {class_id}")
        aggregate_results['class'][class_name] = {
            'class_id': class_id,
            'counts': final_counts_for_class, # Use the cleaned dict
            'rates': rates
        }
    print(f"  Calculated rates for {len(aggregate_results['class'])} classes.")


    # --- Sort Image Results --- (Code unchanged from previous version)
    print(f"\n--- Sorting Image Results by '{sort_key}' ({'Desc' if sort_desc else 'Asc'}) ---")
    def get_sort_value(item):
        rates_dict = item[1].get('rates', {})
        value = rates_dict.get(sort_key)
        if sort_key == 'image_id': return item[1].get('image_id', 0)
        if sort_key == 'image_filename': return item[0]
        if value is None: return float('-inf') if sort_desc else float('inf')
        return value
    try:
        image_results_list = list(image_results.items())
        sorted_image_results_list = sorted(image_results_list, key=get_sort_value, reverse=sort_desc)
    except Exception as e:
         print(f"Warning: Could not sort image results by key '{sort_key}'. Error: {e}. Results will be unsorted.")
         sorted_image_results_list = list(image_results.items())

    # --- Save Results --- (Code mostly unchanged, ensure structure is serializable)
    print("\n--- Saving Results ---")
    output_filepath = os.path.join(output_dir, "error_analysis_results.json")

    # Convert defaultdicts to dicts for saving
    def convert_to_dict(item):
        if isinstance(item, defaultdict):
            item = {k: convert_to_dict(v) for k, v in item.items()}
        elif isinstance(item, dict):
            item = {k: convert_to_dict(v) for k, v in item.items()}
        elif isinstance(item, list):
            item = [convert_to_dict(i) for i in item]
        return item

    results_to_save = {
        "config_used": config, # Already a dict
        "processing_summary": {
            "images_processed": processed_count,
            "missing_prediction_files": missing_pred_count,
        },
        "image_results_sorted": convert_to_dict(sorted_image_results_list), # List of [str, dict]
        "aggregate_results": convert_to_dict(aggregate_results) # Contains nested dicts
    }

    try:
        with open(output_filepath, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f"Results saved successfully to: {output_filepath}")
    except Exception as e:
        print(f"An unexpected error occurred saving results to JSON: {e}")

    # --- Optional: Print Summary to Console --- (Code unchanged)
    print("\n--- Overall Aggregate Rate Summary ---")
    if 'overall' in aggregate_results and 'rates' in aggregate_results['overall']:
        for rate_name, value in aggregate_results['overall']['rates'].items():
            print(f"  {rate_name}: {value:.4f}" if value is not None else f"  {rate_name}: N/A")
    else:
         print("  Overall results not available.")

    print(f"\n--- Top 5 Images by '{sort_key}' ({'Desc' if sort_desc else 'Asc'}) ---")
    for i, (img_filename, data) in enumerate(sorted_image_results_list[:5]):
        rate_val = data.get('rates',{}).get(sort_key, "N/A")
        id_val = data.get('image_id', 'N/A')
        print(f"{i+1}. Image: {img_filename} (ID: {id_val}), {sort_key}: {rate_val if isinstance(rate_val, str) or rate_val is None else f'{rate_val:.4f}'}")

    # Optional: Print summary for top classes by error rate
    if 'class' in aggregate_results:
        print(f"\n--- Top 5 Classes by MissedDetectionRate ---")
        try:
            sorted_classes = sorted(
                aggregate_results['class'].items(),
                key=lambda item: item[1]['rates'].get('MissedDetectionRate', 0) if item[1]['rates'].get('MissedDetectionRate') is not None else 0,
                reverse=True
            )
            for i, (class_name, data) in enumerate(sorted_classes[:5]):
                 rate = data['rates'].get('MissedDetectionRate')
                 print(f"{i+1}. Class: {class_name}, Missed Rate: {rate:.4f}" if rate is not None else f"{i+1}. Class: {class_name}, Missed Rate: N/A")
        except Exception as e:
            print(f"Could not sort or display class results: {e}")


    print("\nAnalysis complete.")


if __name__ == "__main__":
    main(config_path="config.yaml")