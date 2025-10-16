DATA_DIR = "DATA/"

cmd_to_check = [
    "tapquery",
    "tpipe",
    "tcat",
    "tmatchn",
    "tmatch2",
    "tcopy",
    "mocshape",
    "pixfoot",
    "plot2plane",
    "cone-examples",
    "plot2sky",
    "tcatn",
]

# for each of these cmd_to_check lists grab the files from the list
# that is in the directory and count the number of entries.
import os
import json

TRAIN_FILE = f"{DATA_DIR}/training_data.json"
# all the training data files are in DATA but in only certain directories.
TRAIN_FILE_CMDS_DIR = [
    "cone",
    # "descriptions",
    "mocshape",
    # "other",
    "pixfoot",
    "plot2plane",
    "tapquery",
    "tcat",
    "tcopy",
    "tmatch2",
    "tmatchn",
    "tpipe",
    "NOTEBOOKLM",
]

# open all of the dirs and get the locations of the files inside them.

list_of_files = []
for directory in TRAIN_FILE_CMDS_DIR:
    dir_path = os.path.join(DATA_DIR, directory)
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                list_of_files.append(os.path.join(directory, filename))

total_entries = {}
for cmd in cmd_to_check:
    print(cmd)
    total_entries[cmd] = 0
    for file in list_of_files:
        if cmd in file:
            print(file)
            with open(DATA_DIR + file, "r") as f:
                data = json.load(f)
                total_entries[cmd] += len(data)
total = 0
for cmd, count in total_entries.items():
    total = total + count
    print(f"Command: {cmd}, Total entries: {count}")

print(f"Total Number of entries: {total}")
