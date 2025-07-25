import os
import shutil
import pandas as pd

def copy_files(csv_file, train_folder, gt_folder, pred_folder, output_folder):
    # Load the CSV containing image names
    df = pd.read_csv(csv_file)

    # Create output folders for images, ground truths, and predictions
    images_output_folder = os.path.join(output_folder, 'images')
    gt_output_folder = os.path.join(output_folder, 'gt')
    pred_output_folder = os.path.join(output_folder, 'predictions')

    os.makedirs(images_output_folder, exist_ok=True)
    os.makedirs(gt_output_folder, exist_ok=True)
    os.makedirs(pred_output_folder, exist_ok=True)

    # Loop through each row in the CSV to process each image
    for _, row in df.iterrows():
        image_name = row['image_name']  # Assuming column name is 'image_name'

        # Define possible extensions
        possible_extensions = [".png", ".jpg", ".jpeg"]

        # Try to find the image with one of the extensions
        image_file = None
        for ext in possible_extensions:
            img_path = os.path.join(train_folder, image_name + ext)
            if os.path.exists(img_path):
                image_file = img_path
                break
        
        # Debugging: Print image name and attempted paths
        if image_file is None:
            print(f"Image not found for {image_name} in train folder. Tried extensions: {possible_extensions}")
        
        # Define ground truth and prediction file paths
        gt_file = os.path.join(gt_folder, image_name + '.txt')
        pred_file = os.path.join(pred_folder, image_name + '.txt')

        # Debugging: Print paths for GT and predictions
        if not os.path.exists(gt_file):
            print(f"Ground truth file not found for {image_name}: {gt_file}")
        
        if not os.path.exists(pred_file):
            print(f"Prediction file not found for {image_name}: {pred_file}")

        # Check if the required files exist and copy them
        if image_file and os.path.exists(gt_file) and os.path.exists(pred_file):
            # Copy image file
            shutil.copy(image_file, images_output_folder)
            # Copy ground truth file
            shutil.copy(gt_file, gt_output_folder)
            # Copy prediction file
            shutil.copy(pred_file, pred_output_folder)
            # print(f"Copied: {image_name}")
        else:
            print(f"Skipping {image_name}: Files not found")

# Define file paths


# Define file paths

# Define file paths
csv_file = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/map_filtered_.7.csv"  # CSV with 'image_name' column
train_folder = "/home/sahithi_kukkala/sahithi/indicDLP/data/indic_data/train_gt_mask"
gt_folder = "/home/sahithi_kukkala/sahithi/indic_train_labels/train_ground_truths"
pred_folder = "/home/sahithi_kukkala/sahithi/indicDLP/data/indic_data/indic_train_labels/train_result_labels2"
output_folder = "/home/sahithi_kukkala/sahithi/indicDLP/Dataset_cleaning/outputs/filtered_.7"

# Run the function
copy_files(csv_file, train_folder, gt_folder, pred_folder, output_folder)
