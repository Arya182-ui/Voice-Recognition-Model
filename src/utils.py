import os

def ensure_dir(directory):
    """
    Ensure a directory exists.

    Args:
        directory (str): Directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
