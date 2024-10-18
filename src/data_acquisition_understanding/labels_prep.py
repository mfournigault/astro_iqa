import json

# Contenu du fichier Markdown
markdown_content = """
| Image_id | Label |
718683  GOOD
731965  B_SEEING, BT
1013974    B_SEEING
1021182    GOOD
1110042    B_SEEING
1143261    B_SEEING
1625580    BT
1625581   BT
1625582   BT
1625583   GOOD
1625584   GOOD
1625586   GOOD
1625588   BGP, B_SEEING
1625589   BGP, B_SEEING
1625627   GOOD
1625632  GOOD
1625633  GOOD
1635753    RBT
1671968 BGP, B_SEEING
1736833   RBT
1778985   RBT
1850900     BT
1851894 BGP, B_SEEING
1853401 B_SEEING, BT
2120820    B_SEEING
"""

# Initialise structures of the json
data = {}
image_id = []
labels = []

# Separating content into image_id and labels
for line in markdown_content.strip().split('\n')[1:]:
    parts = line.split(maxsplit=1)
    image_id.append(parts[0])
    labels.append(parts[1])
    data[parts[0]] = parts[1]

# Creating the JSON structure by following the COCO format
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

# Écrire les données JSON dans un fichier
with open('map_images_labels.json', 'w') as json_file:
    json_file.write(coco_data)