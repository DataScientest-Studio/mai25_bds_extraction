import os
import platform

def remove_ds_store_files(directory: str) -> None:
    """
    Recursively removes all .DS_Store files from the specified directory and its subdirectories.
    
    This function traverses the given directory and all its subdirectories using os.walk().
    It checks for the presence of any file named ".DS_Store" and removes it. If an error occurs
    during file removal, the exception is caught and an error message is displayed.
    
    Parameters:
    directory (str): The path to the root directory where the search for .DS_Store files will start.
    
    Returns:
    None: This function does not return any value. It simply prints messages about the files removed
          or any errors encountered during the process.
    
    Example:
    remove_ds_store_files("/path/to/your/folder")
    """
    if platform.system() == 'Darwin':
        for folder, _, files in os.walk(directory):
            for file in files:
                if file == ".DS_Store":
                    file_path = os.path.join(folder, file)
                    try:
                        os.remove(file_path)
                        print(f"Removed: {file_path}")
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
     