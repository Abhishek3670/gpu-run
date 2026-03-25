import glob
import json

classes = set()
for file_path in glob.glob("*.json"):
    with open(file_path) as json_file:
        data = json.load(json_file)
        for shape in data.get("shapes", []):
            classes.add(shape.get("label"))

print(classes)
