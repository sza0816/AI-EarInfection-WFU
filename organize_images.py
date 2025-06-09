#%%
import os
import pandas as pd
import shutil
import glob

def organize_images_by_label(image_dir, excel_file, destination_dir):
    # Step 1: Load the Excel file into a DataFrame
    df = pd.read_excel(excel_file)
    
    # Step 2: Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        image_name_prefix = row['ImageName']  # This is the AMxxx identifier
        label = row['AN_Type']  # The classification (Normal, Multiple, etc.)
        
        # Find all files that match the prefix 'AMxxx' (e.g., 'AM1L', 'AM2L', etc.)
        matching_files = glob.glob(os.path.join(image_dir, f"{image_name_prefix}*.bmp"))
        
        if not matching_files:
            print(f"No matching image found for {image_name_prefix} in {image_dir}")
            continue
        
        # Create the directory for the label if it doesn't exist
        label_dir = os.path.join(destination_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        # Copy all matching files to the correct folder
        for matching_file in matching_files:
            # Get the file name without the path
            file_name = os.path.basename(matching_file)
            
            # Build the destination file path
            destination_path = os.path.join(label_dir, file_name)
            
            # Copy the image to the correct folder
            shutil.copy(matching_file, destination_path)
            print(f"Copied {file_name} to {label_dir}")

# Parameters
image_dir = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_Selected_Still_Frames/All_Selected_Still_Frames_org"  # Replace with the actual directory path where images are stored
excel_file = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_Selected_Still_Frames/1-20-21-All_list.xlsx"  # Replace with the path to your Excel file
destination_dir = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_Selected_Still_Frames/All_Selected_Still_Frames"  # Where you want to move the images (the base directory)

# Run the function
organize_images_by_label(image_dir, excel_file, destination_dir)

# %%
