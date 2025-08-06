import json
import os
import sys

import subprocess

data_dir = "DATA/"
docker_cmd = "docker run stilts-chat-stilts-dev-env java -jar chat.jar"
GREEN = "\033[92m"
RESET = "\033[0m"
RED = "\033[91m"


# read the JSON file and get all the responces
def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


# for each response, check if it contains the docker command
def check_docker_command_in_responses(data):
    for item in data:
        responce = item["response"]
        print("Item:", item)
        print("\n\n")
        # remove stilts from the response
        # remove java -jar stilts or stilts from the response
        responce = (
            responce.replace("java -jar stilts.jar", "").replace("stilts", "").strip()
        )

        # test if the command is valid by running it in the docker container.

        cmd = f"{docker_cmd} {responce}"

        # run it with subprocess

        try:
            result = subprocess.run(
                cmd, shell=True, check=True, capture_output=True, text=True
            )

            print(f"Command executed {GREEN}successfully{RESET}:\n\n {cmd}")
            # if sucessful add element to the item
            item["chat.jar"] = "executed successfully"

        except subprocess.CalledProcessError as e:
            print(f"Command {RED}failed{RESET}:\n\n {cmd}\nError: {e.stderr}")
            # if failed add element to the item
            item["chat.jar"] = "failed"
            item["error"] = e.stderr
        print("\n\n-----------------------------------\n\n")


# main function to read the file and check the commands
def main():

    # get a list of all files in the data directory
    data_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    # iterate over each file and check the commands
    for data_file in data_files:
        data_path = os.path.join(data_dir, data_file)
        print(f"Processing file: {data_path}")
        data = read_json_file(data_path)
        check_docker_command_in_responses(data)
        # save the modified data back to the file
        with open(data_path, "w") as file:
            json.dump(data, file, indent=4)

        print("Data saved back to the file.")


if __name__ == "__main__":
    main()
