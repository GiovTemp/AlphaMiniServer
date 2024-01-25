import json
import numpy as np
import os


def load_json_data(file_path):
    """
    Load JSON data from a specified file.

    This function attempts to open a JSON file located at the given file path. If the file is found,
    it reads the file and loads the JSON data into a Python data structure.
    If the file is not found (FileNotFoundError), it returns an empty list.

    :param file_path: The path to the JSON file to be read. This should be a string representing the relative or absolute path.

    :return: Returns the loaded JSON data as a Python data structure. If the file doesn't exist, returns an empty list.
    """
    try:
        with open(file_path, "r") as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        return []


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def save_data_to_json(data, file_path):
    """
    Save given data to a JSON file at the specified path.

    :param data: Data to be serialized to JSON (e.g., list, dict).
    :param file_path: Path for the JSON file to save data.
    """

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, cls=NumpyEncoder)
