import os
from tqdm import tqdm
from pathlib import Path

import os
import platform

def count_files_by_extension(folder: str|Path, extension: str, display_tqdm = True):
    """
    Counts the number of files with a specific extension within subdirectories of a given folder.

    Parameters:
    -----------
    folder : str or Path
        The path to the top-level folder where the search will begin. 
        Must be a valid directory.
        
    extension : str
        The file extension to count (e.g., '.txt', '.csv'). 
        The comparison is case-insensitive.
        
    display_tqdm : bool, optional (default=True)
        Whether to display a progress bar (using tqdm) during the counting process.

    Returns:
    --------
    int
        The total number of files that match the given extension across all subdirectories.

    Raises:
    -------
    AssertionError
        If the provided folder path is not a valid directory.

    Notes:
    ------
    - The function only searches within immediate subdirectories of the given folder.
    - File extensions are matched in a case-insensitive manner (e.g., '.TXT' will match '.txt').
    """

    assert os.path.isdir(folder), f"Unable to access {folder}. Please check that it is a proper foldername."
    dirs = list(filter(lambda x: os.path.isdir(x), Path(folder).iterdir()))
    c = 0
    if display_tqdm:
        dirs = tqdm(dirs)
    for d in dirs:
        for _, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(extension):
                    c += 1
    return c



def remove_ds_store_files(directory: str, verbose=False) -> None:
    """
    Recursively removes all '.DS_Store' files from a given directory and its subdirectories (macOS only).

    Parameters:
    -----------
    directory : str
        The root directory in which to search for and delete '.DS_Store' files.

    verbose : bool, optional (default=False)
        If True, prints detailed information about each removed file and any errors encountered.

    Returns:
    --------
    None

    Notes:
    ------
    - This function only runs on macOS systems (Darwin). On other platforms, it exits without performing any action.
    - It walks through all subdirectories and attempts to delete any file named '.DS_Store'.
    - Displays a summary of the number of files removed and any that could not be deleted.

    Example:
    --------
    >>> remove_ds_store_files("/Users/yourname/projects", verbose=True)

    """
    if platform.system() == 'Darwin':
        removed = 0
        not_removed = 0
        for folder, _, files in os.walk(directory):
            for file in files:
                if file == ".DS_Store":
                    file_path = os.path.join(folder, file)
                    try:
                        os.remove(file_path)
                        removed += 1
                        if verbose:
                            print(f"Removed: {file_path}")
                    except Exception as e:
                        not_removed += 1
                        if verbose:
                            print(f"Error removing {file_path}: {e}")
        print(f"Done.\nRemoved {removed} .DS_Store files ({not_removed} remaining)")
    else:
        print("This function is to be used on mac systems only. Exiting with no action.")
     