import os
import shutil

def find_empty_schematics(directory):
    """
    Finds KiCad schematic files containing only '.title KiCad schematic' and '.end'.

    Args:
        directory (str): The directory to search within.

    Returns:
        tuple: A tuple containing:
            - int: The number of empty schematic files found.
            - list: A list of filepaths for the empty schematic files.
    """
    empty_files_count = 0
    empty_files_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".cir"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    expected_content = ".title KiCad schematic\n.end"
                    if content == expected_content:
                        empty_files_count += 1
                        empty_files_list.append(filepath)
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")

    return empty_files_count, empty_files_list

def move_non_empty_files(source_dir, dest_dir):
    """
    Moves all non-empty KiCad schematic files from a source directory to a destination directory.

    Args:
        source_dir (str): The directory to move files from.
        dest_dir (str): The directory to move files to.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    empty_files_count, empty_files_list = find_empty_schematics(source_dir)
    
    for filename in os.listdir(source_dir):
        if filename.endswith(".cir"):
            filepath = os.path.join(source_dir, filename)
            if filepath not in empty_files_list:
                dest_path = os.path.join(dest_dir, filename)
                try:
                    shutil.move(filepath, dest_path)
                    print(f"Moved {filename} to {dest_dir}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")

if __name__ == "__main__":
    directory_to_search = "organized_cir_files"
    count, file_list = find_empty_schematics(directory_to_search)

    if count > 0:
        print("Files:")
        for filepath in file_list:
            print(filepath)
        print(f"Number of files with only '.title KiCad schematic' and '.end': {count}")

    non_empty_dir = "non_empty_cir_files"
    move_non_empty_files(directory_to_search, non_empty_dir)

