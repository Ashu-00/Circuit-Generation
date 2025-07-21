import os
import glob
import shutil

def count_cir_files(folder_path):
    """
    Count the number of .cir files in the specified folder and its subfolders.
    
    Args:
        folder_path (str): Path to the folder to search in
    
    Returns:
        int: Number of .cir files found
    """
    # Use glob to find all .cir files recursively
    pattern = os.path.join(folder_path, "**", "*.cir")
    cir_files = glob.glob(pattern, recursive=True)
    
    return len(cir_files)

def copy_cir_files_with_ids(cir_files, destination_folder="organized_cir_files"):
    """
    Copy all .cir files to a new folder with sequential ID numbers as filenames.
    
    Args:
        cir_files (list): List of paths to .cir files
        destination_folder (str): Name of the destination folder
    
    Returns:
        dict: Mapping of new filename to original file path
    """
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    file_mapping = {}
    
    for i, file_path in enumerate(cir_files, 1):
        # Create new filename with sequential ID
        new_filename = f"{i}.cir"
        new_file_path = os.path.join(destination_folder, new_filename)
        
        # Copy the file
        shutil.copy2(file_path, new_file_path)
        
        # Store the mapping
        file_mapping[new_filename] = file_path
        
        print(f"Copied: {file_path} -> {new_file_path}")
    
    print(f"\nSuccessfully copied {len(cir_files)} files to '{destination_folder}' folder.")
    return file_mapping

def main():
    # Specify the folder path
    spice_datasets_folder = "spice-datasets"
    
    # Check if the folder exists
    if not os.path.exists(spice_datasets_folder):
        print(f"Error: Folder '{spice_datasets_folder}' does not exist.")
        return
    
    # Count .cir files
    file_count = count_cir_files(spice_datasets_folder)
    
    print(f"Number of .cir files in '{spice_datasets_folder}' and its subfolders: {file_count}")
    
    # Optional: List all found files for verification
    pattern = os.path.join(spice_datasets_folder, "**", "*.cir")
    cir_files = glob.glob(pattern, recursive=True)
    
    if cir_files:
        print("\nFound files:")
        # for file_path in sorted(cir_files):
        #     print(f"  {file_path}")
        print(len(cir_files), "files found.")
        
        # Copy files with sequential IDs
        file_mapping = copy_cir_files_with_ids(cir_files)
        
        # Optionally save the mapping to a text file
        with open("file_mapping.txt", "w") as f:
            for new_name, original_path in file_mapping.items():
                f.write(f"{new_name} -> {original_path}\n")
        print("File mapping saved to 'file_mapping.txt'")
        
    else:
        print("\nNo .cir files found.")
    
    return cir_files

if __name__ == "__main__":
    main()