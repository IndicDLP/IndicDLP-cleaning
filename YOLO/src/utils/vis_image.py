import os
import cv2
import numpy as np
from PIL import Image  # Import Pillow for DPI handling

# Paths (Modify these if you want to process a single file)
image_folder = "/home/sahithi_kukkala/sahithi/indic_data/images/train"
label_folder = "/home/sahithi_kukkala/sahithi/indic_train_labels/train_result_labels"
output_mask_name_folder = "/home/sahithi_kukkala/sahithi/indic_result_annotated_img"

# Optional: Set these to process a single image and label file
single_image_path = image_folder = "/home/sahithi_kukkala/sahithi/indic_data/images/train/ar_en_001067.png"
 # Example: "/home/sahithi_kukkala/sahithi/indic_data/images/train/image1.png"
single_label_path = '/home/sahithi_kukkala/sahithi/indic_train_labels/train_ground_truths/ar_en_001067.txt' # Example: "/home/sahithi_kukkala/sahithi/indic_train_labels/train_result_labels/image1.txt"

os.makedirs(output_mask_name_folder, exist_ok=True)

# Define colors
color_mapping = {
    'advertisement': '#1ABC9C', 'answer': '#2980B9', 'author': '#2980B9',
    'chapter-title': '#F7C8E0', 'contact-info': '#D35400', 'dateline': '#FFFF00',
    'figure': '#8E44AD', 'figure-caption': '#FDCB6E', 'first-level-question': '#E74C3C',
    'flag': '#E74C3C', 'folio': '#74B9FF', 'footer': '#81ECEC', 'footnote': '#A29BFE',
    'formula': '#2ECC71', 'header': '#2C3E50', 'headline': '#D35400', 'index': '#FFCCB3',
    'jumpline': '#F39C12', 'options': '#E8DFCA', 'ordered-list': '#2C3E50', 'page-number': '#A0BCC2',
    'paragraph': '#55EFC4', 'placeholder-text': '#3498DB', 'quote': '#8CC0DE', 'reference': '#27AE60',
    'second-level-question': '#9B59B6', 'section-title': '#F39C12', 'sidebar': '#F1C40F',
    'sub-headline': '#8E44AD', 'sub-ordered-list': '#E67E22', 'sub-section-title': '#C0392B',
    'subsub-ordered-list': '#AAD9BB', 'subsub-section-title': '#27AE60', 'sub-unordered-list': '#E0AED0',
    'subsub-headline': '#D5B4B4', 'subsub-unordered-list': '#AC87C5', 'table': '#FAB1A0',
    'table-caption': '#E17055', 'table-of-contents': '#9ED2C6', 'third-level-question': '#FF9494',
    'unordered-list': '#8E44AD', 'website-link': '#8CC0DE'
}

# Convert hex to BGR (OpenCV format)
def hex_to_bgr(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (5, 3, 1))

# Darken color for better visibility
def darken_color(color, factor=0.6):
    return tuple(int(c * factor) for c in color)

# Create color mappings
bgr_colors = {label: hex_to_bgr(color) for label, color in color_mapping.items()}
label_mapping = {i: (label, bgr_colors[label]) for i, label in enumerate(bgr_colors.keys())}

# Function to save image with DPI
def save_image_with_dpi(image, path, dpi=600):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image.save(path, dpi=(dpi, dpi))

# Process a single image if specified, otherwise process all images in the folder
def process_image(image_path, label_path, output_folder):
    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print(f"Skipping: Image or label file missing - {image_path}, {label_path}")
        return None

    print(f"Processing: {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return None

    img_height, img_width = img.shape[:2]

    # Load annotations
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
            print(boxes)

    # Copy image for annotation
    mask_with_text = img.copy()

    for x_min, y_min, x_max, y_max, label_name, color in boxes:
        overlay_text = mask_with_text.copy()

        # Draw filled rectangle
        cv2.rectangle(overlay_text, (x_min, y_min), (x_max, y_max), color, -1)
        cv2.addWeighted(overlay_text, 0.4, mask_with_text, 0.6, 0, mask_with_text)

        # Draw border
        cv2.rectangle(mask_with_text, (x_min, y_min), (x_max, y_max), darken_color(color), 1)

        # Draw text label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6  
        font_thickness = 1  
        text_size, baseline = cv2.getTextSize(label_name, font, font_scale, font_thickness)
        textbox_height = text_size[1] + 30  
        text_x, text_y = max(5, x_min), y_min
        text_bg_color = darken_color(color)
        bg_start = (text_x, text_y - textbox_height)
        bg_end = (text_x + text_size[0] + 20, text_y)

        cv2.rectangle(mask_with_text, bg_start, bg_end, text_bg_color, -1)
        text_org = (text_x + 10, text_y - (textbox_height - text_size[1]) // 2)
        cv2.putText(mask_with_text, label_name, text_org, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Save the annotated image
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    save_image_with_dpi(mask_with_text, output_path, dpi=600)

    print(f"Saved: {output_path}")
    return output_path

# --- Execution Logic ---
if single_image_path and single_label_path:
    # Process single file
    output_path = process_image(single_image_path, single_label_path, output_mask_name_folder)
    if output_path:
        print(f"Single image processed and saved at: {output_path}")
else:
    # Process all files in the folder
    for label_file in os.listdir(label_folder):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(label_folder, label_file)
        base_name = os.path.splitext(label_file)[0]

        # Find corresponding image
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = os.path.join(image_folder, base_name + ext)
            if os.path.exists(test_path):
                image_path = test_path
                break

        if image_path:
            process_image(image_path, label_path, output_mask_name_folder)

print("Processing complete!")
