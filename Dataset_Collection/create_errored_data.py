import os
import glob
from pathlib import Path
from add_errors import inject_error

def create_errored_data():
    """
    Read all .cir files from organized_cir_files folder, add 2 errors to each,
    and save them in a new folder with '_err' suffix.
    """
    # Define paths
    input_folder = Path("/home/ashed/Documents/Fyp/organized_cir_files")
    output_folder = Path("/home/ashed/Documents/Fyp/errored_cir_files")
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True)
    
    # Find all .cir files in the input folder and its subdirectories
    cir_files = glob.glob(str(input_folder / "**/*.cir"), recursive=True)
    
    if not cir_files:
        print(f"No .cir files found in {input_folder}")
        return
    
    print(f"Found {len(cir_files)} .cir files")
    
    for cir_file_path in cir_files:
        try:
            # Read the original file
            with open(cir_file_path, 'r', encoding='utf-8') as file:
                original_content = file.read()
            
            # Inject 2 errors
            errored_content, error_descriptions = inject_error(
                original_content, 
                error_type='random', 
                num_errors=5
            )
            
            # Create output filename with '_err' suffix
            original_filename = Path(cir_file_path).stem
            output_filename = f"{original_filename}_err.cir"
            output_path = output_folder / output_filename
            
            # Save the errored file
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(errored_content)
            
            # Create a log file with error descriptions
            log_filename = f"{original_filename}_err_log.txt"
            log_path = output_folder / log_filename
            
            with open(log_path, 'w', encoding='utf-8') as log_file:
                log_file.write(f"Original file: {cir_file_path}\n")
                log_file.write(f"Errored file: {output_path}\n")
                log_file.write(f"Number of errors injected: {len(error_descriptions)}\n\n")
                log_file.write("Error descriptions:\n")
                for i, error_desc in enumerate(error_descriptions, 1):
                    log_file.write(f"{i}. {error_desc}\n")
            
            print(f"Processed: {Path(cir_file_path).name} -> {output_filename}")
            print(f"  Errors injected: {len(error_descriptions)}")
            for error_desc in error_descriptions:
                print(f"    - {error_desc}")
            
        except Exception as e:
            print(f"Error processing {cir_file_path}: {str(e)}")
            continue
    
    print(f"\nProcessing complete! Errored files saved to: {output_folder}")

if __name__ == "__main__":
    create_errored_data()
