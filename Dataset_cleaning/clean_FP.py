# import os

# # Directories
# gt_dir = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/gt"
# pred_dir = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/predictions"
# output_dir = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/FP_gt_new"

# # Ensure output directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Load false positive image names
# false_positive_list = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/fp_imgs_sample.csv"

# with open(false_positive_list, "r") as f:
#     false_positive_images = {line.strip().split(",")[0] for line in f.readlines() if line.strip()}

# # Process each false positive image
# for img_name in false_positive_images:
#     gt_file = os.path.join(gt_dir, img_name + ".txt")
#     pred_file = os.path.join(pred_dir, img_name + ".txt")
#     output_file = os.path.join(output_dir, img_name + ".txt")

#     # Read GT file
#     gt_boxes = []
#     if os.path.exists(gt_file):
#         with open(gt_file, "r") as f:
#             gt_boxes = [line.strip() for line in f.readlines() if line.strip()]

#     # Read Prediction file
#     pred_boxes = []
#     if os.path.exists(pred_file):
#         with open(pred_file, "r") as f:
#             pred_boxes = [line.strip().split() for line in f.readlines() if line.strip()]

#     # Extract false positives (confidence > 0.6), removing confidence score
#     fp_boxes = [f"{line[0]} " + " ".join(line[1:5]) for line in pred_boxes if len(line) > 5 and float(line[-1]) > 0.6]
#     print(fp_boxes)
#     # Combine original GT boxes and false positives
#     updated_gt_boxes = gt_boxes + fp_boxes

#     # Save the modified GT file
#     with open(output_file, "w") as f:
#         f.write("\n".join(updated_gt_boxes) + "\n")

# print("✅ False positives added to ground truth files in 'FP_gt_new'.")



import os
import shutil

# Directories
gt_dir = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/gt"
pred_dir = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/predictions"
output_dir = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/FP_gt_new"
os.makedirs(output_dir, exist_ok=True)

# Load false positive image names
fp_list_path = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.8/fp_imgs_sample.csv"
with open(fp_list_path, "r") as f:
    false_positive_images = {line.strip() for line in f.readlines()[1:]}  # Skip header

def load_annotations(file_path):
    """Load annotations from a YOLO-style text file."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as file:
        lines = file.readlines()
    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:  # Ensure confidence exists
            continue
        class_id, x_center, y_center, width, height, confidence = map(float, parts)
        annotations.append((int(class_id), x_center, y_center, width, height, confidence))
    return annotations

def save_annotations(file_path, annotations):
    """Save annotations to a YOLO-style text file."""
    with open(file_path, "w") as file:
        for ann in annotations:
            file.write(f"{ann[0]} {ann[1]} {ann[2]} {ann[3]} {ann[4]}\n")  # No confidence for GT

def replace_gt_with_fp(gt_file, pred_file, output_file):
    """Replace GT with false positives if multiple predictions exist for one GT."""
    gt_boxes = load_annotations(gt_file)
    pred_boxes = load_annotations(pred_file)

    # Convert GT boxes to dictionary for easier lookup
    gt_dict = {(x, y, w, h): cls for cls, x, y, w, h, _ in gt_boxes}
    
    new_gt_boxes = []
    
    for cls, x, y, w, h, conf in pred_boxes:
        if (x, y, w, h) in gt_dict:
            # If prediction matches GT, keep it
            new_gt_boxes.append((cls, x, y, w, h))
        else:
            # If a different prediction exists, replace GT with the highest-confidence box
            matched_preds = [p for p in pred_boxes if p[:5] == (cls, x, y, w, h)]
            if len(matched_preds) > 1:  # Multiple predictions for one GT box
                best_pred = max(matched_preds, key=lambda p: p[5])  # Highest confidence
                if best_pred[5] > 0.6:  # Replace only if confidence > 0.6
                    new_gt_boxes.append((best_pred[0], best_pred[1], best_pred[2], best_pred[3], best_pred[4]))
            else:
                # If no multiple matches, just add it
                new_gt_boxes.append((cls, x, y, w, h))
    
    save_annotations(output_file, new_gt_boxes)

# Process each false positive image
for img_name in false_positive_images:
    gt_file = os.path.join(gt_dir, img_name + ".txt")
    pred_file = os.path.join(pred_dir, img_name + ".txt")
    output_file = os.path.join(output_dir, img_name + ".txt")
    
    if os.path.exists(gt_file) and os.path.exists(pred_file):
        replace_gt_with_fp(gt_file, pred_file, output_file)
    elif os.path.exists(gt_file):  # If no prediction file, copy original GT
        shutil.copy(gt_file, output_file)

print("Updated GT files saved in:", output_dir)
