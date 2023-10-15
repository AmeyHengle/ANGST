import os
import argparse
import shutil

def main(source_folder, output_folder):
    # Loop through the main folder
    for root, dirs, files in os.walk(source_folder):
        # Check if 'best_model' is in the current path
        if 'best_model' in root.split(os.sep):
            match = False
            # Loop through the files in the 'best_model' folder
            for file in files:
                # If the file ends with .csv
                if file.endswith('.csv'):
                    # Extract the subfolder name from the path
                    subfolder_name = root.split(os.sep)[-2]
                    # Create a new file name using the subfolder name
                    new_file_name = f"{subfolder_name}.csv"
                    # Path to store the csv file in the output directory
                    destination = os.path.join(output_folder, new_file_name)
                    # Copy the file to the output directory
                    shutil.copy2(os.path.join(root, file), destination)
                    match = True
            if not match:
                print(root.split(os.sep)[-2], root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy .csv files from best_model directories.')
    parser.add_argument('source_folder', type=str, help='Path to the main folder to search.')
    parser.add_argument('output_folder', type=str, help='Path to the folder where the .csv files should be copied.')
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    main(args.source_folder, args.output_folder)
    
    
"""
python src/postprocess.py ./anxiety_label/ ./predictions/binaryCLF/anxiety
python src/postprocess.py ./depression_label ./predictions/binaryCLF/depression
python src/postprocess.py ./AIMH ./predictions/multilabelCLF/finetuned/
"""
