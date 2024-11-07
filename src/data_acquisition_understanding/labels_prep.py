import json
import os
import sys

def convert_txt_annotations_to_json(markdown_file, json_file):
    # Read the content of the markdown file
    with open(markdown_file, 'r') as file:
        markdown_content = file.read()

    # Initialize structures for the JSON
    data = {}
    image_id = []
    labels = []

    # Separate content into image_id and labels
    # we suppose that there is no header in the txt file
    for line in markdown_content.strip().split('\n')[:]:
        parts = line.split(maxsplit=1)
        image_id.append(parts[0])
        if len(parts) > 1:
            labels.append(parts[1])
            data[parts[0]] = parts[1]

    # Create the JSON structure following the COCO format
    info = {
        "description": "Mapping quality labels to images",
        "year": 2024,
        "version": "1.0",
        "contributor": "mfournigault",
        "url": "https://github.com/mfournigault/astro_iqa",
    }
    coco = {
        "info": info,
        "images": image_id,
        "categories": ["GOOD", "B_SEEING", "BGP", "BT", "RBT"],
        "annotations": data
    }
    coco_data = json.dumps(coco, indent=4)

    # Write the JSON data to a file
    with open(json_file, 'w') as json_file:
        json_file.write(coco_data)


if __name__ == '__main__':
    # get data_directory and file_type from command line
    txt_file = sys.argv[1]
    json_file = sys.argv[2]
    convert_txt_annotations_to_json(txt_file, json_file)