import os
import json

TRAIN_FILE_CMDS_DIR = [
    "NOTEBOOKLM",
    "cone",
    "mocshape",
    "pixfoot",
    "plot2plane",
    "tapquery",
    "tcat",
    "tcopy",
    "tmatch2",
    "tmatchn",
    "tpipe",
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

# open all of the dirs and get the locations of the files inside them.

list_of_files = []
for directory in TRAIN_FILE_CMDS_DIR:
    dir_path = os.path.join(DATA_DIR, directory)
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                list_of_files.append(os.path.join(directory, filename))


total_entries = {}
failure_modes = {
    "Error: No handler for output format": 0,
    "(Hint: arguments containing spaces must be quoted)": 0,
    "Unused arguments": 0,
    "/bin/sh": 0,
    "Error: No file, URL or command": 0,
    "Error: Unknown processing comm": 0,
    "Error: Row index": 0,
    "Error: Row count": 0,
    "Error: Bad value for parameter": 0,
    "Error: Bad Expression": 0,
    "Error: Can't guess output format": 0,
    "No such file or directory": 0,
    "Premature end of file": 0,
}
for cmd in cmd_to_check:
    print(cmd)
    total_entries[cmd] = 0
    for file in list_of_files:
        if cmd in file:
            # print(file)
            with open(DATA_DIR + file, "r") as f:
                data = json.load(f)
                for entry in data:
                    if entry.get("chat.jar") == "failed":
                        error_message = entry.get("error", "")
                        found_failure_mode = False
                        for mode in failure_modes:
                            if mode in error_message:
                                failure_modes[mode] += 1
                                found_failure_mode = True
                                break
                total_entries[cmd] += len(data)

print("\nFailure Mode Statistics:")
for mode, count in failure_modes.items():
    print(f"  {mode}: {count}")
total = 0
for cmd, count in total_entries.items():
    total = total + count
    print(f"Command: {cmd}, Total entries: {count}")

print(f"Total Number of entries: {total}")

total = 0
