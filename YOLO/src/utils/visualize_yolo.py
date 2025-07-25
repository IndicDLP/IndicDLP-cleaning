# import os
# import cv2
# import numpy as np
# from PIL import Image  # Import Pillow for DPI handling

# # Paths
# image_folder = "/home/sahithi_kukkala/sahithi/indic_data/images/train"
# label_folder = "/home/sahithi_kukkala/sahithi/indic_train_labels/train_result_labels"
# output_mask_name_folder = "/home/sahithi_kukkala/sahithi/indic_result_annotated"

# os.makedirs(output_mask_name_folder, exist_ok=True)

# # Define colors
# color_mapping = {
#     'advertisement': '#1ABC9C',
#     'answer': '#2980B9',
#     'author': '#2980B9',
#     'chapter-title': '#F7C8E0',
#     'contact-info': '#D35400',
#     'dateline': '#FFFF00',
#     'figure': '#8E44AD',
#     'figure-caption': '#FDCB6E',
#     'first-level-question': '#E74C3C',
#     'flag': '#E74C3C',
#     'folio': '#74B9FF',
#     'footer': '#81ECEC',
#     'footnote': '#A29BFE',
#     'formula': '#2ECC71',
#     'header': '#2C3E50',
#     'headline': '#D35400',
#     'index': '#FFCCB3',
#     'jumpline': '#F39C12',
#     'options': '#E8DFCA',
#     'ordered-list': '#2C3E50',
#     'page-number': '#A0BCC2',
#     'paragraph': '#55EFC4',
#     'placeholder-text': '#3498DB',
#     'quote': '#8CC0DE',
#     'reference': '#27AE60',
#     'second-level-question': '#9B59B6',
#     'section-title': '#F39C12',
#     'sidebar': '#F1C40F',
#     'sub-headline': '#8E44AD',
#     'sub-ordered-list': '#E67E22',
#     'sub-section-title': '#C0392B',
#     'subsub-ordered-list': '#AAD9BB',
#     'subsub-section-title': '#27AE60',
#     'sub-unordered-list': '#E0AED0',
#     'subsub-headline': '#D5B4B4',
#     'subsub-unordered-list': '#AC87C5',
#     'table': '#FAB1A0',
#     'table-caption': '#E17055',
#     'table-of-contents': '#9ED2C6',
#     'third-level-question': '#FF9494',
#     'unordered-list': '#8E44AD',
#     'website-link': '#8CC0DE'
# }




# def hex_to_bgr(hex_color):
#     return tuple(int(hex_color[i:i+2], 16) for i in (5, 3, 1))

# def darken_color(color, factor=0.6):
#     return tuple(int(c * factor) for c in color)

# bgr_colors = {label: hex_to_bgr(color) for label, color in color_mapping.items()}
# label_mapping = {i: (label, bgr_colors[label]) for i, label in enumerate(bgr_colors.keys())}

# # Function to save image with DPI
# def save_image_with_dpi(image, path, dpi=600):
#     pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL
#     pil_image.save(path, dpi=(dpi, dpi))  # Save with specified DPI

# for label_file in os.listdir(label_folder):
#     if not label_file.endswith(".txt"):
#         continue

#     label_path = os.path.join(label_folder, label_file)
#     # print(f"Processing: {label_path}")
    
#     # Get base name without extension
#     base_name = os.path.splitext(label_file)[0]
    
#     # Find image with any extension
#     image_path = None
#     image_ext = None
#     for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
#         test_path = os.path.join(image_folder, base_name + ext)
#         if os.path.exists(test_path):
#             image_path = test_path
#             image_ext = ext  # Store the extension
#             break
    
#     if image_path is None:
#         print(f"Could not find image for {label_file}")
#         continue
    
#     # print(f"Found image: {image_path}")

#     if not os.path.exists(image_path):
#         continue

#     img = cv2.imread(image_path)
#     if img is None:
#         continue

#     img_height, img_width = img.shape[:2]

#     boxes = []
#     with open(label_path, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue

#             class_id = int(parts[0])
#             if class_id not in label_mapping:
#                 continue

#             label_name, color = label_mapping[class_id]
#             x_center, y_center, width, height = map(float, parts[1:])

#             x_min = int((x_center - width / 2) * img_width)
#             y_min = int((y_center - height / 2) * img_height)
#             x_max = int((x_center + width / 2) * img_width)
#             y_max = int((y_center + height / 2) * img_height)

#             boxes.append((x_min, y_min, x_max, y_max, label_name, color))

#     mask_with_text = img.copy()
#     mask_without_text = img.copy()

#     for x_min, y_min, x_max, y_max, label_name, color in boxes:
#         overlay_text = mask_with_text.copy()
#         overlay_no_text = mask_without_text.copy()

#         cv2.rectangle(overlay_text, (x_min, y_min), (x_max, y_max), color, -1)
#         cv2.rectangle(overlay_no_text, (x_min, y_min), (x_max, y_max), color, -1)

#         cv2.addWeighted(overlay_text, 0.4, mask_with_text, 0.6, 0, mask_with_text)
#         cv2.addWeighted(overlay_no_text, 0.4, mask_without_text, 0.6, 0, mask_without_text)

#         cv2.rectangle(mask_with_text, (x_min, y_min), (x_max, y_max), darken_color(color), 1)
#         cv2.rectangle(mask_without_text, (x_min, y_min), (x_max, y_max), darken_color(color), 1)

#         # Label drawing (only mask_with_text)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.6  
#         font_thickness = 1  
#         text_size, baseline = cv2.getTextSize(label_name, font, font_scale, font_thickness)
#         textbox_height = text_size[1] + 30  
#         text_x, text_y = max(5, x_min), y_min
#         text_bg_color = darken_color(color)
#         bg_start = (text_x, text_y - textbox_height)
#         bg_end = (text_x + text_size[0] + 20, text_y)

#         cv2.rectangle(mask_with_text, bg_start, bg_end, text_bg_color, -1)
#         text_org = (text_x + 10, text_y - (textbox_height - text_size[1]) // 2)
#         cv2.putText(mask_with_text, label_name, text_org, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

#     # Save images with DPI
#     output_path = os.path.join(output_mask_name_folder, base_name + image_ext)
#     save_image_with_dpi(mask_with_text, output_path, dpi=600)

# print("Masked images saved successfully with 300 DPI!")


import os
import cv2
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

# Paths
image_folder = "/home/sahithi_kukkala/sahithi/indicDLP/data/indic_data/images/train"
label_folder = "/home/sahithi_kukkala/sahithi/indicDLP/data/indic_data/indic_train_labels/FP_gt_new"
output_mask_name_folder = "/home/sahithi_kukkala/sahithi/fp_gt_new_mask"

os.makedirs(output_mask_name_folder, exist_ok=True)

# Define colors
color_mapping = {
    'advertisement': '#1ABC9C',
    'answer': '#2980B9',
    'author': '#2980B9',
    'chapter-title': '#F7C8E0',
    'contact-info': '#D35400',
    'dateline': '#FFFF00',
    'figure': '#8E44AD',
    'figure-caption': '#FDCB6E',
    'first-level-question': '#E74C3C',
    'flag': '#E74C3C',
    'folio': '#74B9FF',
    'footer': '#81ECEC',
    'footnote': '#A29BFE',
    'formula': '#2ECC71',
    'header': '#2C3E50',
    'headline': '#D35400',
    'index': '#FFCCB3',
    'jumpline': '#F39C12',
    'options': '#E8DFCA',
    'ordered-list': '#2C3E50',
    'page-number': '#A0BCC2',
    'paragraph': '#55EFC4',
    'placeholder-text': '#3498DB',
    'quote': '#8CC0DE',
    'reference': '#27AE60',
    'second-level-question': '#9B59B6',
    'section-title': '#F39C12',
    'sidebar': '#F1C40F',
    'sub-headline': '#8E44AD',
    'sub-ordered-list': '#E67E22',
    'sub-section-title': '#C0392B',
    'subsub-ordered-list': '#AAD9BB',
    'subsub-section-title': '#27AE60',
    'sub-unordered-list': '#E0AED0',
    'subsub-headline': '#D5B4B4',
    'subsub-unordered-list': '#AC87C5',
    'table': '#FAB1A0',
    'table-caption': '#E17055',
    'table-of-contents': '#9ED2C6',
    'third-level-question': '#FF9494',
    'unordered-list': '#8E44AD',
    'website-link': '#8CC0DE'
}

# Convert hex to BGR
def hex_to_bgr(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (5, 3, 1))

# Darken color
def darken_color(color, factor=0.6):
    return tuple(int(c * factor) for c in color)

# Convert color mapping to BGR
bgr_colors = {label: hex_to_bgr(color) for label, color in color_mapping.items()}
label_mapping = {i: (label, bgr_colors[label]) for i, label in enumerate(bgr_colors.keys())}

# Save image with DPI
def save_image_with_dpi(image, path, dpi=600):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image.save(path, dpi=(dpi, dpi))

# Process a single file
def process_file(label_file):
    if not label_file.endswith(".txt"):
        return

    label_path = os.path.join(label_folder, label_file)
    base_name = os.path.splitext(label_file)[0]

    # Find corresponding image
    image_path = None
    image_ext = None
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        test_path = os.path.join(image_folder, base_name + ext)
        if os.path.exists(test_path):
            image_path = test_path
            image_ext = ext
            break

    if image_path is None:
        # print(f"Could not find image for {label_file}")
        return

    img = cv2.imread(image_path)
    if img is None:
        return

    img_height, img_width = img.shape[:2]
    boxes = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            if class_id not in label_mapping:
                continue

            label_name, color = label_mapping[class_id]
            x_center, y_center, width, height = map(float, parts[1:])
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)
            boxes.append((x_min, y_min, x_max, y_max, label_name, color))

    mask_with_text = img.copy()
    mask_without_text = img.copy()

    for x_min, y_min, x_max, y_max, label_name, color in boxes:
        overlay_text = mask_with_text.copy()
        overlay_no_text = mask_without_text.copy()

        cv2.rectangle(overlay_text, (x_min, y_min), (x_max, y_max), color, -1)
        cv2.rectangle(overlay_no_text, (x_min, y_min), (x_max, y_max), color, -1)

        cv2.addWeighted(overlay_text, 0.4, mask_with_text, 0.6, 0, mask_with_text)
        cv2.addWeighted(overlay_no_text, 0.4, mask_without_text, 0.6, 0, mask_without_text)

        cv2.rectangle(mask_with_text, (x_min, y_min), (x_max, y_max), darken_color(color), 1)
        cv2.rectangle(mask_without_text, (x_min, y_min), (x_max, y_max), darken_color(color), 1)

        # Add label text to mask_with_text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 1
        text_size, _ = cv2.getTextSize(label_name, font, font_scale, font_thickness)
        textbox_height = text_size[1] + 30
        text_x, text_y = max(5, x_min), y_min
        text_bg_color = darken_color(color)
        bg_start = (text_x, text_y - textbox_height)
        bg_end = (text_x + text_size[0] + 20, text_y)

        cv2.rectangle(mask_with_text, bg_start, bg_end, text_bg_color, -1)
        text_org = (text_x + 10, text_y - (textbox_height - text_size[1]) // 2)
        cv2.putText(mask_with_text, label_name, text_org, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Save images
    output_path = os.path.join(output_mask_name_folder, base_name + image_ext)
    save_image_with_dpi(mask_with_text, output_path, dpi=600)

# Main function to process files in parallel
if __name__ == "__main__":
    label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]

    # Use multiprocessing
    num_workers = max(1, cpu_count() - 1)  # Use all available cores except one
    with Pool(num_workers) as pool:
        pool.map(process_file, label_files)

    print("Masked images saved successfully with 600 DPI!")
