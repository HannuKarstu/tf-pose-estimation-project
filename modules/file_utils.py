import os
import random
import requests
import json


def open_json_as_dict(filepath: str):
    """
    Open JSON file as dictionary

    Args:
        filepath (str): Path to JSON file

    Returns:
        Data as dictionary
    """

    if not os.path.isfile(filepath) and not filepath.endswith(".json"):
        raise Exception(f"Invalid JSON file path: '{filepath}'")

    with open(filepath) as f:
        data = json.load(f)

    return data


def filepaths_from_path(
    files_path: str,
    extensions: tuple,
    limit: int = None,
    shuffle: bool = True,
    filenames: list = None
):
    """
    Returns the paths of files from a folder that match given extensions.

    Args:
        files_path (str): Path from which to search files
        extensions (tuple): Extensions of filepaths to return. For example: (".png", ".jpg", ".jpeg")
        limit (int): Limit amount of image paths
        shuffle (bool): Default True. Shuffle output filepath list.
        filenames (list): Default None. Return only filenames found that match this list.

    Returns:
        filepaths (list): List of filepath strings
    """

    def is_supported_ext(file_path):
        return os.path.isfile(file_path) and file_path.lower().endswith(extensions)

    def is_in_filenames_list(file_path):
        if filenames:
            return os.path.basename(file_path) in filenames
        return True

    filepaths = [os.path.join(files_path, f) for f in os.listdir(
        files_path) if is_supported_ext(os.path.join(files_path, f)) and is_in_filenames_list(f)]

    if shuffle:
        random.shuffle(filepaths)

    if limit:
        filepaths = filepaths[0:limit]

    return filepaths


def download_file_to_folder(
    url: str,
    folder: str,
    output_filename: str,
    overwrite: bool = False
):
    """
    Downloads file from url and saves it to selected folder

    Args:
        files_path (str): Path from which to search files
        url (str): Web address of file to download
        folder (str): Folder to save the file to
        output_filename (str): Filename of the downloaded file
        overwrite (bool): Default False. Whether to overwrite the downloaded file if already exists.

    Returns:
        filepath (str): Path to downloaded or already existing file.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = os.path.join(folder, output_filename)

    if not os.path.isfile(filepath) or overwrite:
        with open(filepath, "wb") as f:
            f.write(requests.get(url, timeout=30).content)
            print(f"Downloaded '{filepath}'")
    else:
        print(f"File '{filepath}' already found, not downloading.")

    return filepath
