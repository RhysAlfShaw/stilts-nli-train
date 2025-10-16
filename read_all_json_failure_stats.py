import json
import os


def analyze_json_files(data_dir="DATA"):
    """
    Analyzes all JSON files in the specified directory to count
    items based on their 'chat.jar' status.
    """
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

    import os
    import json

    TRAIN_FILE = f"{data_dir}/training_data.json"
    TRAIN_FILE_CMDS_DIR = [
        "NOTEBOOKLM",
    ]

    list_of_files = []
    for directory in TRAIN_FILE_CMDS_DIR:
        dir_path = os.path.join(data_dir, directory)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".json"):
                    list_of_files.append(os.path.join(directory, filename))
    json_files = list_of_files
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    total_average = 0
    for filename in json_files:
        file_path = os.path.join(data_dir, filename)
        print(f"\n--- Analyzing {filename} ---")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"  Skipping {filename}: Not a list of items.")
                continue

            failed_count = 0
            success_count = 0
            other_count = 0
            total_items = len(data)

            for item in data:
                chat_jar_status = item.get("chat.jar")
                if chat_jar_status == "failed":
                    failed_count += 1
                elif chat_jar_status == "executed successfully":
                    success_count += 1
                else:
                    other_count += 1

            print(f"  Total items: {total_items}")
            print(f"  'chat.jar' = 'failed': {failed_count}")
            print(f"  'chat.jar' = 'executed successfully': {success_count}")
            print(f"  Other 'chat.jar' status or missing: {other_count}")
            # fail percentage
            average = (success_count / total_items) * 100 if total_items > 0 else 0
            total_average = total_average + average
            print(f"Average Vaild rate: {average:.2f}%")
        except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON from {filename}")
        except Exception as e:
            print(f"  An unexpected error occurred with {filename}: {e}")
    print(f"Total Average Invalid: {total_average/len(json_files)} %")


if __name__ == "__main__":
    analyze_json_files()
