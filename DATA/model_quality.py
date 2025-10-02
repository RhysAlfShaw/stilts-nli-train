import json
import os


def analyze_model_quality(data_dir="DATA/model_gen_eval"):
    """
    Analyzes JSON files in a specified directory to calculate the percentage
    of failed commands ('chat.jar' == 'failed') for each file.

    Args:
        data_dir (str): The directory containing the JSON files to analyze.
    """
    print(f"Analyzing model quality in: {data_dir}")
    results = {}

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                total_commands = len(data)
                failed_commands = 0

                for item in data:
                    if item.get("chat.jar") == "failed":
                        failed_commands += 1

                if total_commands > 0:
                    failure_percentage = (failed_commands / total_commands) * 100
                else:
                    failure_percentage = 0.0

                results[filename] = {
                    "total_commands": total_commands,
                    "failed_commands": failed_commands,
                    "failure_percentage": f"{failure_percentage:.2f}%",
                }
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {filename}. Skipping.")
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

    print("\n--- Model Quality Analysis Results ---")
    for filename, stats in results.items():
        print(f"\nFile: {filename}")
        print(f"  Total Commands: {stats['total_commands']}")
        print(f"  Failed Commands: {stats['failed_commands']}")
        print(f"  Failure Percentage: {stats['failure_percentage']}")
    print("--------------------------------------")


if __name__ == "__main__":
    analyze_model_quality()
