#Utility setup functions for App
#To be imported into main file

import os
import datetime
from pathlib import Path

#Checks if a directory exists and creates one if not
def ensure_directory_exists(directory_path):
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    #parents=True: creates all necessary parent directories
    #exist_ok=True: prevents errors if directory already exists
    return directory_path

#Initializes a timestamp for naming files
def generate_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #Formats date/time as: YYYYMMDD_HHMMSS (year, month, day, hour, minute, second)

#Creates a new filename for processed image
def create_processed_filename(original_path, suffix="_processed"):
    directory = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    name, ext = os.path.splitext(filename)
    #Splits the original path into components and splits name and extension text

    return os.path.join(directory, f"{name}{suffix}{ext}")
    #Constructs new filename with suffix, indicating that the image has been processed

#Gets the size of the file
def get_file_size(file_path):
    if not os.path.exists(file_path):
        return "File not found"
    #Returns message if file not found

    size_bytes = os.path.getsize(file_path)
    #Gets file size in bytes

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    #Converts to kivy-readable units
    
    # For large files
    return f"{size_bytes:.2f} TB"