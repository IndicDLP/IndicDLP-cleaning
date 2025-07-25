import os
import shutil
import random
from typing import List, Tuple

class DatasetSampler:
    """
    A class to sample images and their corresponding label files from
    a flat dataset structure where domain is inferred from filename.
    """

    def __init__(self, base_image_dir: str, base_label_dir: str, output_root_dir: str):
        """
        Initializes the DatasetSampler for a flat directory structure.

        Args:
            base_image_dir (str): The directory containing all images (e.g., /home/sahithi_kukkala/indicDLP/data/indic_data/images/test).
            base_label_dir (str): The directory containing all labels (e.g., /home/sahithi_kukkala/indicDLP/data/indic_data/labels/test).
            output_root_dir (str): The root directory where sampled images and labels will be saved.
        """
        self.base_image_dir = base_image_dir
        self.base_label_dir = base_label_dir
        self.output_root_dir = output_root_dir
        self.output_images_dir = os.path.join(output_root_dir, "sampled_images")
        self.output_labels_dir = os.path.join(output_root_dir, "sampled_labels")

        self._create_output_directories()

    def _create_output_directories(self):
        """Creates the necessary output directories if they don't exist."""
        os.makedirs(self.output_images_dir, exist_ok=True)
        os.makedirs(self.output_labels_dir, exist_ok=True)
        print(f"Output image directory created: {self.output_images_dir}")
        print(f"Output label directory created: {self.output_labels_dir}")

    def _get_all_image_files(self) -> List[str]:
        """
        Retrieves all image files from the base image directory.

        Returns:
            List[str]: A list of image filenames (e.g., 'ar_as_000027_0.png').
        """
        image_files = []
        for f in os.listdir(self.base_image_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(f)
        return image_files

    def sample_by_domain(self, num_samples_per_domain: int = 100):
        """
        Samples images and labels, grouping by domain inferred from the filename.

        Args:
            num_samples_per_domain (int): The number of images to sample from each domain.
        """
        all_image_files = self._get_all_image_files()
        if not all_image_files:
            print(f"No images found in {self.base_image_dir}. Exiting.")
            return

        # Group images by inferred domain
        domain_files = {}
        for filename in all_image_files:
            # Assuming domain is the first part of the filename (e.g., 'ar' from 'ar_as_000027_0.png')
            domain = filename.split('_')[0]
            if domain not in domain_files:
                domain_files[domain] = []
            domain_files[domain].append(filename)

        if not domain_files:
            print("No domains could be inferred from image filenames. Exiting.")
            return

        print(f"Found {len(domain_files)} domains: {list(domain_files.keys())}")

        for domain_name, image_filenames in domain_files.items():
            # Ensure we don't try to sample more images than available
            actual_samples = min(num_samples_per_domain, len(image_filenames))
            sampled_files = random.sample(image_filenames, actual_samples)

            print(f"\nSampling {actual_samples} images from domain: {domain_name}")

            for image_filename in sampled_files:
                original_image_full_path = os.path.join(self.base_image_dir, image_filename)

                # Label filenames are expected to have the same base name but with a .txt extension
                label_filename = os.path.splitext(image_filename)[0] + ".txt"
                original_label_full_path = os.path.join(self.base_label_dir, label_filename)

                # Create domain-specific subfolders in the output directories
                output_domain_image_dir = os.path.join(self.output_images_dir, domain_name)
                output_domain_label_dir = os.path.join(self.output_labels_dir, domain_name)
                os.makedirs(output_domain_image_dir, exist_ok=True)
                os.makedirs(output_domain_label_dir, exist_ok=True)

                # Define destination paths
                dest_image_path = os.path.join(output_domain_image_dir, image_filename)
                dest_label_path = os.path.join(output_domain_label_dir, label_filename)

                # Copy image
                try:
                    shutil.copy(original_image_full_path, dest_image_path)
                    # print(f"  Copied image: {image_filename}") # Uncomment for verbose logging
                except FileNotFoundError:
                    print(f"  Warning: Image not found at {original_image_full_path}. Skipping label and continuing.")
                    continue
                except Exception as e:
                    print(f"  Error copying image {image_filename}: {e}. Skipping label and continuing.")
                    continue

                # Copy label
                try:
                    if os.path.exists(original_label_full_path):
                        shutil.copy(original_label_full_path, dest_label_path)
                        # print(f"  Copied label: {label_filename}") # Uncomment for verbose logging
                    else:
                        print(f"  Warning: Label file not found for {image_filename} at {original_label_full_path}.")
                except Exception as e:
                    print(f"  Error copying label {label_filename}: {e}.")

        print("\nSampling process complete for all domains.")

# --- Configuration and Execution ---
if __name__ == "__main__":
    # Define your base directories
    BASE_IMAGES_DIR = "/home/sahithi_kukkala/indicDLP/data/indic_data/images/test"
    BASE_LABELS_DIR = "/home/sahithi_kukkala/indicDLP/data/indic_data/labels/test"
    OUTPUT_ROOT_DIR = "/home/sahithi_kukkala/indicDLP/data/sampled_data_100" # New root for sampled output

    # Initialize the sampler
    sampler = DatasetSampler(BASE_IMAGES_DIR, BASE_LABELS_DIR, OUTPUT_ROOT_DIR)

    # Run the sampling process for all domains
    # Each domain will have 100 images and their labels sampled
    sampler.sample_by_domain(num_samples_per_domain=100)

    print(f"\nSampled images saved to: {sampler.output_images_dir}")
    print(f"Corresponding labels saved to: {sampler.output_labels_dir}")