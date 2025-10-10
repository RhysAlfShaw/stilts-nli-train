import json
import os


def add_cmd_to_data(file_path):
    """
    Reads a JSON file, adds a new key 'cmd' with the file name (without extension)
    to each item in the JSON array, and writes the modified data back to the file.

    Args:
        file_path (str): The path to the JSON file.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item["cmd"] = file_name_without_ext
                else:
                    print(
                        f"Warning: Item in JSON array is not a dictionary in {file_path}. Skipping 'cmd' addition."
                    )
        elif isinstance(data, dict):
            data["cmd"] = file_name_without_ext
        else:
            print(
                f"Warning: JSON data in {file_path} is neither a list nor a dictionary. Skipping 'cmd' addition."
            )

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Successfully added 'cmd' key to {file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from {file_path}. Is it a valid JSON file?"
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":

    dummy_file_path = "DATA/NOTEBOOKLM/tpipe.json"
    add_cmd_to_data(file_path=dummy_file_path)
