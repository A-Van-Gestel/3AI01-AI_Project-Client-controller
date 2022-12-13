from os import makedirs, path


# Function to check if a directory exists, if not, make this directory
def check_dir(directory: str):
    dir_exists = path.isdir(directory)
    if not dir_exists:
        makedirs(directory)
    return directory
