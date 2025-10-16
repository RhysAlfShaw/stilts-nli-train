import os
import json

TRAIN_FILE_CMDS_DIR = [
    "NOTEBOOKLM",
    # "cone",
    # "mocshape",
    # "pixfoot",
    # "plot2plane",
    # "tapquery",
    # "tcat",
    # "tcopy",
    # "tmatch2",
    # "tmatchn",
    # "tpipe",
]
DATA_DIR = "DATA/"
cmd_to_check = [
    "tapquery.json",
    "tpipe.json",
    "tcat.json",
    "tmatchn.json",
    "tmatch2.json",
    "tcopy.json",
    "mocshape.json",
    "pixfoot.json",
    "plot2plane.json",
    "cone-examples.json",
    "plot2sky.json",
    "tcatn.json",
]

list_of_files = []
for directory in TRAIN_FILE_CMDS_DIR:
    dir_path = os.path.join(DATA_DIR, directory)
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                list_of_files.append(os.path.join(directory, filename))


total_entries = {}
failure_modes = [
    "Error: No handler for output format",
    "(Hint: arguments containing spaces must be quoted)",
    "Unused arguments",
    "/bin/sh",
    "Error: No file, URL or command",
    "Error: Unknown processing comm",
    "Error: Row index",
    "Error: Row count",
    "Error: Bad value for parameter",
    "Error: Bad Expression",
    "Error: Can't guess output format",
    "No such file or directory",
    "Premature end of file",
]

# for each failure mode collect and save each element in the json to a file named by failure type.
# also save the cmd they come from to keep track.
output_dir = "failed_data"
os.makedirs(output_dir, exist_ok=True)

failure_data = {mode: [] for mode in failure_modes}

for cmd_file in cmd_to_check:
    for file_path in list_of_files:
        if cmd_file in file_path:
            with open(os.path.join(DATA_DIR, file_path), "r") as f:
                data = json.load(f)
                for entry in data:
                    if entry.get("chat.jar") == "failed":
                        error_message = entry.get("error", "")
                        for mode in failure_modes:
                            if mode in error_message:
                                entry["source_cmd_file"] = cmd_file
                                failure_data[mode].append(entry)
                                break

for mode, entries in failure_data.items():
    # Sanitize mode name for filename
    sanitized_mode = "".join(c if c.isalnum() else "_" for c in mode)
    filename = os.path.join(output_dir, f"{sanitized_mode}.json")
    with open(filename, "w") as f:
        json.dump(entries, f, indent=4)

print("\nFailed entries saved by failure mode to the 'failed_data' directory.")
