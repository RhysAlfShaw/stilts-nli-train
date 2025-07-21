import xml.etree.ElementTree as ET
import json
import os


def convert_xml_to_json(xml_file_path, json_file_path):
    """
    Parses an XML file and converts the content of <example> tags
    into a JSON file.

    The XML is expected to have a structure like:
    <output>
      <example>
        <task>...</task>
        <prompt>...</prompt>
        <response>...</response>
      </example>
      ...
    </output>

    The output JSON will be a list of objects, where each object has
    a "prompt" and a "response" key.
    e.g., [{"prompt": "...", "response": "..."}, ...]

    Args:
        xml_file_path (str): The path to the input XML file.
        json_file_path (str): The path where the output JSON file will be saved.
    """
    # --- 1. Error handling for file paths ---
    if not os.path.exists(xml_file_path):
        print(f"Error: The file '{xml_file_path}' was not found.")
        return

    # --- 2. Data extraction and parsing ---
    data_list = []
    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Find all 'example' elements
        for example in root.findall("example"):
            # Find the 'prompt' and 'response' elements within each 'example'
            prompt_element = example.find("prompt")
            response_element = example.find("response")

            # Extract text content, handling cases where tags might be missing
            prompt_text = (
                prompt_element.text.strip()
                if prompt_element is not None and prompt_element.text
                else ""
            )
            response_text = (
                response_element.text.strip()
                if response_element is not None and response_element.text
                else ""
            )

            # Create a dictionary and append it to our list
            if prompt_text or response_text:  # Only add if there's some content
                data_list.append({"prompt": prompt_text, "response": response_text})

    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # --- 3. JSON file creation ---
    try:
        # Write the list of dictionaries to a JSON file
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            # Use indent=4 for pretty-printing the JSON
            json.dump(data_list, json_file, ensure_ascii=False, indent=4)
        print(f"Successfully converted '{xml_file_path}' to '{json_file_path}'")

    except IOError as e:
        print(f"Error writing to JSON file: {e}")


# --- Example Usage ---
# 2. Create a dummy XML file to run the script on
input_xml_file = "DATA/examples.xml"

# 3. Define the desired output JSON file name
output_json_file = "DATA/doc-examples-formatted.json"

# 4. Run the conversion function
convert_xml_to_json(input_xml_file, output_json_file)

# 5. (Optional) Print the content of the created JSON file to verify
print("\n--- Content of generated JSON file ---")
with open(output_json_file, "r", encoding="utf-8") as f:
    print(f.read())
