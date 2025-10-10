# READ in the data from NOTEBOOKLM and remove all the failed data.
# combine into a single table.
# randomly split 10% into test/eval data.
# save both training and test data into DATA/final_data_training
# as train.json and test.json
import random
import os
import json


def create_training_and_test_data(
    base_dir="DATA/NOTEBOOKLM",
    output_dir="DATA/final_data_training",
    test_split=0.1,
):
    """
    Reads JSON data from the specified base directory, filters out failed entries,
    combines them, and splits them into training and test sets.
    The resulting training and test data are saved as JSON files in the output directory.
    """
    all_data = []

    print(f"\n--- Reading files from {base_dir} ---")
    for filename in os.listdir(base_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(base_dir, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    print(f"  Skipping {filename}: Not a list of items.")
                    continue

                for item in data:
                    status = str(item.get("chat.jar", "")).lower().strip()
                    if status != "failed":
                        all_data.append(item)

            except (json.JSONDecodeError, Exception) as e:
                print(f"  Could not process {filename}: {e}")

    # --- Process the fixed_failed_data subdirectory ---
    fixed_data_dir = os.path.join(base_dir, "fixed_failed_data")
    if not os.path.isdir(fixed_data_dir):
        print(f"Warning: Subdirectory '{fixed_data_dir}' not found. Skipping.")
    else:
        print(f"\n--- Reading files from {fixed_data_dir} ---")
        for filename in os.listdir(fixed_data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(fixed_data_dir, filename)
                try:
                    with open(file_path, "r") as f:

                        data = json.load(f)

                    if not isinstance(data, list):
                        print(f"  Skipping {filename}: Not a list of items.")
                        continue

                    for item in data:
                        status = str(item.get("chat.jar", "")).lower().strip()
                        # In the fixed_failed_data, we only care about 'fixed' or successful items
                        # 'failed' items in this directory are considered 'removed'
                        if status in {
                            "fixed",
                            "executed successfully",
                            "skipped",
                            "skipped - no 'stilts' prefix",
                            "skipped-plot",
                            "skipped-operaton in param",
                        }:
                            all_data.append(item)

                except (json.JSONDecodeError, Exception) as e:
                    print(f"  Could not process {filename}: {e}")

    # Shuffle and split data
    random.shuffle(all_data)
    test_size = int(len(all_data) * test_split)
    test_data = all_data[:test_size]
    train_data = all_data[test_size:]

    # print the sizes
    print(f"\nTotal samples collected: {len(all_data)}")
    print(f"Training samples: {len(train_data)}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save training data
    train_output_path = os.path.join(output_dir, "train.json")
    with open(train_output_path, "w") as f:
        json.dump(train_data, f, indent=4)
    print(f"\nSaved {len(train_data)} training samples to {train_output_path}")

    # Save test data
    test_output_path = os.path.join(output_dir, "test.json")
    with open(test_output_path, "w") as f:
        json.dump(test_data, f, indent=4)
    print(f"Saved {len(test_data)} test samples to {test_output_path}")


if __name__ == "__main__":
    create_training_and_test_data()
