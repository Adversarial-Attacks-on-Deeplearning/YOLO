#!/bin/bash

# Create directory structure
mkdir -p coco/{annotations,val2017}

# Download validation set (1GB)
wget http://images.cocodataset.org/zips/val2017.zip -P coco/

# Download annotations (241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P coco/

# Unzip files
unzip coco/val2017.zip -d coco/val2017/
unzip coco/annotations_trainval2017.zip -d coco/annotations/

# Cleanup zip files
rm coco/val2017.zip coco/annotations_trainval2017.zip

# Verify structure
echo "Validation images: $(find coco/val2017 -type f | wc -l)"
echo "Annotations: $(find coco/annotations -type f -name '*.json')"