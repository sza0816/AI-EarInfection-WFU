from collections import defaultdict
import os

def get_valid_classes(root_dir, min_images=50):
    class_counts = defaultdict(int)
    for root, _, files in os.walk(root_dir):
        class_name = os.path.basename(root)
        if class_name not in ['.', '..']:
            class_counts[class_name] += len([f for f in files if f.endswith('.tiff') or f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')])
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_images]
    print(f"Classes with more than {min_images} images: {valid_classes}")
    return valid_classes, class_counts

def get_valid_classes_with_folders(root_dir, threshold=50):
    valid_classes = []
    class_counts = defaultdict(int)
    
    # Traverse the directory
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        
        if os.path.isdir(folder_path):
            # Count image files directly under the folder
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.tiff', '.tif', '.jpg', '.png', '.bmp'))]
            num_images = len(image_files)

            # Count subfolders (image bags) under the folder
            subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            num_bags = len(subdirs)
            
            # If the folder contains more than the threshold of images or bags, it's valid
            if num_images >= threshold or num_bags >= threshold:
                valid_classes.append(folder_path)
                class_counts[folder] = max(num_images, num_bags)

    # Output valid classes
    print(f"Classes with more than {threshold} images or image bags: {valid_classes}")
    
    return valid_classes, class_counts