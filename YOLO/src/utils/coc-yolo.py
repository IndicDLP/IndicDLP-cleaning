
import os
import json
import cv2


# Paths
json_path = "/home/sahithi_kukkala/indicDLP/publaynet/train.json"
image_folder = '/home/sahithi_kukkala/indicDLP/publaynet/train'
output_folder = "/home/sahithi_kukkala/indicDLP/publaynet/labels/train"

os.makedirs(output_folder, exist_ok=True)

# Read JSON data
with open(json_path, "r") as f:
    data = json.load(f)


# Preprocess annotations for fast lookup
annotations_dict = {}
for ann in data["annotations"]:
    img_id = ann["image_id"]
    if img_id not in annotations_dict:
        annotations_dict[img_id] = []
    annotations_dict[img_id].append(ann)

# Process each image
for image in data["images"]:
    print(image)
    img_id = image["id"]
    img_filename = image["file_name"]
    img_path = os.path.join(image_folder, img_filename)

    # Get image dimensions from JSON instead of loading it
    img_width = image.get("width", None)
    img_height = image.get("height", None)
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_filename} not found, skipping...")
        continue
    if img_width is None or img_height is None:
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print(f"❌ Could not read image: {img_path}, skipping...")
            continue
        img_height, img_width = img_cv.shape[:2]
        print(f"📸 Fetched dimensions from image for {img_filename}: {img_width}x{img_height}")

    # Get annotations for this image
    image_annotations = annotations_dict.get(img_id, [])

    # Write YOLO label file
    txt_filename = os.path.splitext(img_filename)[0] + ".txt"
    txt_filepath = os.path.join(output_folder, txt_filename)

    with open(txt_filepath, "w") as txt_file:
        for ann in image_annotations:
            category_id = ann["category_id"]

            # Convert bounding box format
            x_min, y_min, box_width, box_height = ann["bbox"]
            x_center = x_min + box_width / 2
            y_center = y_min + box_height / 2

            # Normalize
            txt_file.write(f"{category_id} {x_center/img_width:.6f} {y_center/img_height:.6f} {box_width/img_width:.6f} {box_height/img_height:.6f}\n")
            print("YOLO annotation files created successfully!")


# import os
# import json
# import cv2

# # Paths
# json_path = "/home/sahithi_kukkala/sahithi/indic_train_pred.json"
# image_folder = '/home/sahithi_kukkala/sahithi/indicDLP/data/indic_data/images/train'
# output_folder = "/home/sahithi_kukkala/sahithi/train_result_labels2"

# os.makedirs(output_folder, exist_ok=True)

# # Read JSON data
# with open(json_path, "r") as f:
#     data = json.load(f)

# # Preprocess annotations for fast lookup
# annotations_dict = {}
# for ann in data["annotations"]:
#     img_id = ann["image_id"]
#     if img_id not in annotations_dict:
#         annotations_dict[img_id] = []
#     annotations_dict[img_id].append(ann)

# # Process each image
# for image in data["images"]:
#     img_id = image["id"]
#     img_filename = image["file_name"]
#     img_path = os.path.join(image_folder, img_filename)

#     # Get image dimensions from JSON
#     img_width = image["width"]
#     img_height = image["height"]

#     # if not os.path.exists(img_path):
#     #     print(f"Warning: Image {img_filename} not found, skipping...")
#     #     continue

#     # Get annotations for this image
#     image_annotations = annotations_dict.get(img_id, [])

#     # Write YOLO label file with confidence scores
#     txt_filename = os.path.splitext(img_filename)[0] + ".txt"
#     txt_filepath = os.path.join(output_folder, txt_filename)

#     with open(txt_filepath, "w") as txt_file:
#         for ann in image_annotations:
#             category_id = ann["category_id"]
#             score = ann.get("score", 1.0)  # Use default 1.0 if score missing

#             # Convert bounding box format
#             x_min, y_min, x_max, y_max = ann["bbox"]
#             width = x_max - x_min
#             height = y_max - y_min
#             x_center = x_min + width / 2
#             y_center = y_min + height / 2

#             # Normalize and write with confidence score
#             txt_file.write(
#                 f"{category_id} {x_center/img_width:.6f} {y_center/img_height:.6f} "
#                 f"{width/img_width:.6f} {height/img_height:.6f} {score:.6f}\n"
#             )

# print("YOLO annotation files with confidence scores created successfully!")



# import os
# import json

# # Input paths
# json_folder = "/home/sahithi_kukkala/indicDLP/GR_layout/INDICYOLO/json_main_run_corrected"  # folder containing multiple JSON files
# image_folder = '/home/sahithi_kukkala/sahithi/indicDLP/data/indic_data/images/train'
# output_folder = "/home/sahithi_kukkala/indicDLP/GR_layout/INDICYOLO/gr_235_layout"

# os.makedirs(output_folder, exist_ok=True)

# # Loop through each JSON file in the folder
# for json_file in os.listdir(json_folder):
#     if not json_file.endswith(".json"):
#         continue

#     json_path = os.path.join(json_folder, json_file)

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     # Preprocess annotations for fast lookup
#     annotations_dict = {}
#     for ann in data["annotations"]:
#         img_id = ann["image_id"]
#         if img_id not in annotations_dict:
#             annotations_dict[img_id] = []
#         annotations_dict[img_id].append(ann)

#     # Process each image in the JSON
#     for image in data["images"]:
#         img_id = image["id"]
#         img_filename = image["file_name"]
#         img_path = os.path.join(image_folder, img_filename)

#         # Get image dimensions from JSON
#         img_width = image["width"]
#         img_height = image["height"]

#         # Get annotations for this image
#         image_annotations = annotations_dict.get(img_id, [])

#         # Output YOLO label file
#         txt_filename = os.path.splitext(img_filename)[0] + ".txt"
#         txt_filepath = os.path.join(output_folder, txt_filename)

#         with open(txt_filepath, "w") as txt_file:
#             for ann in image_annotations:
#                 category_id = ann["category_id"]
#                 score = ann.get("score", 1.0)  # default to 1.0 if missing

#                 # Convert bounding box format
#                 x_min, y_min, x_max, y_max = ann["bbox"]
#                 width = x_max - x_min
#                 height = y_max - y_min
#                 x_center = x_min + width / 2
#                 y_center = y_min + height / 2

#                 # Normalize and write
#                 txt_file.write(
#                     f"{category_id} {x_center/img_width:.6f} {y_center/img_height:.6f} "
#                     f"{width/img_width:.6f} {height/img_height:.6f} {score:.6f}\n"
#                 )

#     print(f"Finished processing: {json_file}")

# print("All YOLO annotation files with confidence scores created successfully!")