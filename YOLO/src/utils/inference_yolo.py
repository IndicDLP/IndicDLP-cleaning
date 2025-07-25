from ultralytics import YOLO
import torch
import os
import json

def process_images_with_yolo(folder_path, model_path, output_json_path=None):
    # Load the YOLO model
    model = YOLO(model_path)

    category_mapping = {
        "advertisement": 0, "answer": 1, "author": 2, "chapter-title": 3, "contact-info": 4,
        "dateline": 5, "figure": 6, "figure-caption": 7, "first-level-question": 8, "flag": 9,
        "folio": 10, "footer": 11, "footnote": 12, "formula": 13, "header": 14, "headline": 15,
        "index": 16, "jumpline": 17, "options": 18, "ordered-list": 19, "page-number": 20,
        "paragraph": 21, "placeholder-text": 22, "quote": 23, "reference": 24,
        "second-level-question": 25, "section-title": 26, "sidebar": 27, "sub-headline": 28,
        "sub-ordered-list": 29, "sub-section-title": 30, "subsub-ordered-list": 31,
        "subsub-section-title": 32, "sub-unordered-list": 33, "subsub-headline": 34,
        "subsub-unordered-list": 35, "table": 36, "table-caption": 37,
        "table-of-contents": 38, "third-level-question": 39, "unordered-list": 40,
        "website-link": 41
    }

    # Prepare the JSON structure
    results_json = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": cname} for cname, cid in category_mapping.items()]
    }

    annotation_id = 0
    image_id = 0  # Added this

    def save_results_as_json(result, image_path, output_json, image_id):
        nonlocal annotation_id

        # Image information
        image_info = {
            "file_name": os.path.basename(image_path),
            "id": image_id,
            "height": result.orig_shape[0],
            "width": result.orig_shape[1]
        }
        output_json["images"].append(image_info)

        # Check if detections exist
        if result.boxes is None or len(result.boxes) == 0:
            return

        boxes = result.boxes.xyxy.cpu().numpy().tolist()  # Convert to list
        scores = result.boxes.conf.cpu().numpy().tolist()  # Convert to list
        labels = result.boxes.cls.cpu().numpy().tolist()  # Convert to list

        for box, score, label in zip(boxes, scores, labels):
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(label),
                "bbox": box,
                "score": float(score)
            }
            output_json["annotations"].append(annotation)
            annotation_id += 1

    # Get all image files from the folder
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    batch_size = 128

    # Process each image in batches
    for batch_idx in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_idx:batch_idx + batch_size]

        try:
            # Pass the batch of image paths to the model
            results = model(batch_paths, device=0, imgsz=1024)  # Removed [0] -> single device
            
            # Process results for this batch
            for image_path, result in zip(batch_paths, results):
                save_results_as_json(result, image_path, results_json, image_id)  # Fixed function call
                image_id += 1  # Increment image_id
                print(f"Processed {image_path}: {len(result.boxes) if result.boxes else 0} objects detected.")

        except Exception as e:
            print(f"Error processing batch {batch_idx//batch_size}: {e}")

    # Optionally save the final JSON to disk
    if output_json_path:
        with open(output_json_path, 'w') as json_file:
            json.dump(results_json, json_file, indent=4)
        print(f"Processing complete. JSON result saved to {output_json_path}")

    return results_json

# Paths
folder_path = '/home/sahithi_kukkala/sahithi/GR-deskewed/gr_deskewed'
model_path = '/home/sahithi_kukkala/sahithi/best_latest_md2_indicdlp_ft.pt'
output_json_path = "/home/sahithi_kukkala/sahithi/gr_deskewed_indic.json"

# Run function
results_json = process_images_with_yolo(folder_path, model_path, output_json_path)
