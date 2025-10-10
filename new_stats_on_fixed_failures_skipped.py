import json
import os
from collections import defaultdict


def analyze_notebooklm_stats(base_dir="DATA/NOTEBOOKLM"):
    """
    Analyzes JSON files in the specified base directory and its 'fixed_failed_data'
    subdirectory to count successful, failed, and fixed 'chat.jar' executions,
    grouped by their 'cmd' key.
    """
    # Use defaultdict to automatically handle new cmd keys without causing KeyErrors.
    cmd_success_counts = defaultdict(int)
    cmd_initial_failure_counts = defaultdict(int)
    cmd_fixed_counts = defaultdict(int)
    cmd_removed_counts = defaultdict(int)
    list_of_all_chatjar_values = []
    # --- Process the main directory ---
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return

    print(f"\n--- Analyzing files in {base_dir} ---")
    for filename in os.listdir(base_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(base_dir, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    continue

                for item in data:
                    # Normalize status string to handle potential typos or whitespace
                    status = str(item.get("chat.jar", "")).lower().strip()
                    list_of_all_chatjar_values.append([status])
                    cmd_key = item.get("cmd")

                    if not cmd_key:
                        continue

                    # Define success conditions for the main directory
                    if status in {
                        "executed successfully",
                        "skipped - no 'stilts' prefix",
                    }:
                        cmd_success_counts[cmd_key] += 1

                    elif status in {"failed"}:
                        cmd_initial_failure_counts[cmd_key] += 1

            except (json.JSONDecodeError, Exception) as e:
                print(f"  Could not process {filename}: {e}")

    # --- Process the fixed_failed_data subdirectory ---
    fixed_data_dir = os.path.join(base_dir, "fixed_failed_data")
    if not os.path.isdir(fixed_data_dir):
        print(f"Warning: Subdirectory '{fixed_data_dir}' not found. Skipping.")
    else:
        print(f"\n--- Analyzing files in {fixed_data_dir} ---")
        for filename in os.listdir(fixed_data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(fixed_data_dir, filename)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    if not isinstance(data, list):
                        continue

                    for item in data:
                        status = str(item.get("chat.jar", "")).lower().strip()
                        list_of_all_chatjar_values.append([status])
                        cmd_key = item.get("cmd")

                        if not cmd_key:
                            continue

                        # A "fixed" command is both a fix and a success.
                        if status == "fixed":
                            cmd_fixed_counts[cmd_key] += 1
                            # cmd_success_counts[cmd_key] += 1
                        # Handle other success statuses, including common typos
                        elif status in {
                            "executed successfully",
                            "skipped",
                            "skipped - no 'stilts' prefix",
                            "skipped-plot",
                            "skipped-operaton in param",
                        }:
                            cmd_fixed_counts[cmd_key] += 1

                        elif status == "failed":
                            cmd_removed_counts[cmd_key] += 1

                except (json.JSONDecodeError, Exception) as e:
                    print(f"  Could not process {filename}: {e}")

    # --- Print Combined Summary Table ---
    print("\n--- Combined Execution Statistics by Command ---")

    # Get a unique, sorted list of all commands found.
    all_cmds = sorted(
        list(
            set(cmd_success_counts.keys())
            | set(cmd_initial_failure_counts.keys())
            | set(cmd_fixed_counts.keys())
            | set(cmd_removed_counts.keys())
        )
    )

    if not all_cmds:
        print("No command statistics to display.")
        return

    # Print table header
    print(
        f"{'Command':<15} {'total':<10} {'Failure':<10} {'Successful':<12} {'Removed':<10} {'Fixed':<8} {'Final':<10} {'Success Rate (%)':>18}"
    )
    print(
        f"{'-'*15:<15} {'-'*10:>10} {'-'*10:>10} {'-'*12:>12} {'-'*10:>10} {'-'*8:>8} {'-'*10:>10} {'-'*18:>18}"
    )

    # Print table rows for each command
    for cmd in all_cmds:
        success = cmd_success_counts[cmd]
        failure = cmd_initial_failure_counts[cmd]
        fixed = cmd_fixed_counts[cmd]
        removed = cmd_removed_counts[cmd] + (failure - fixed)
        final_total = success + fixed

        total_attempts = success + failure

        success_rate = (
            ((final_total) / total_attempts * 100) if total_attempts > 0 else 0
        )

        print(
            f"{cmd:<15} {total_attempts:<10} {failure:>10} {success:>12} {removed:>10} {fixed:>8} {final_total:>10} {success_rate:>17.2f}%"
        )

    total_all_data = 0
    total_usable_data = 0
    total_fixed_data = 0
    total_removed = 0
    for cmd in all_cmds:
        success = cmd_success_counts[cmd]
        failure = cmd_initial_failure_counts[cmd]
        fixed = cmd_fixed_counts[cmd]
        removed = cmd_removed_counts[cmd]
        total_removed += failure - fixed
        total_all_data += success + failure
        total_usable_data += success + fixed
        total_fixed_data += fixed

    print(f"Total generated data: {total_all_data}")
    print(f"Total training data: {total_usable_data}")
    print(f"Total removed data: {total_removed}")
    print(f"Total fixed data: {total_fixed_data}")
    print(f"all Unique chat.jar values")
    unique_chatjar_values = set(tuple(x) for x in list_of_all_chatjar_values)
    for value in unique_chatjar_values:
        print(value)


if __name__ == "__main__":
    # The main directory containing the JSON files.
    data_directory = "DATA/NOTEBOOKLM"
    analyze_notebooklm_stats(data_directory)
