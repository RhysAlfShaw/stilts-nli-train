import json
import re

file_path = "/home/rhys/stilts-ai-train/DATA/failed_data/_Hint__arguments_containing_spaces_must_be_quoted_ copy.json"

try:
    with open(file_path, "r") as f:
        data = json.load(f)
except (IOError, json.JSONDecodeError) as e:
    print(f"Error reading or parsing JSON file: {e}")
    exit(1)


def fix_command_value(value):
    value = value.strip()
    if not value or " " not in value or '"' in value:
        return value

    parts = value.split(" ", 1)
    command = parts[0]

    if len(parts) == 1:
        return value

    rest = parts[1]

    if command in ("addcol", "replacecol"):
        arg_parts = rest.split(" ", 1)
        if len(arg_parts) == 2:
            return f'{command} {arg_parts[0]} "{arg_parts[1]}"'

    return f'{command} "{rest}"'


def process_response(response):
    pattern = re.compile(r"((?:o|i|u)?cmd\d*)= '([^\"]*)'")

    def replacer(match):
        arg_name = match.group(1)
        full_value = match.group(2)

        commands = full_value.split(";")
        fixed_commands = [fix_command_value(c.strip()) for c in commands]

        return f"{arg_name}='{'; '.join(fixed_commands)}'"

    return pattern.sub(replacer, response)


for item in data:
    if "response" in item and isinstance(item["response"], str):
        if "cmd='select" not in item["response"]:
            item["response"] = process_response(item["response"])

try:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print("File fixed successfully.")
except IOError as e:
    print(f"Error writing to file: {e}")
