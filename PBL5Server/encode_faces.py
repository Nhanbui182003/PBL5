"""
A script to encode faces from a dataset and save the encodings to a file.
"""

from imutils import paths
import pickle
import cv2
import os
import face_recognition

# Set default values directly in the script
dataset_path = "face_recognition_dataset"  # Path to the directory of faces and images
encodings_path = "models/encodings.pickle"  # Path to the serialized db of facial encodings
detection_method = "cnn"  # face detector to use: cnn or hog

# Get paths of images in dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(dataset_path))

# Initialize list to hold known encodings and known names
knownEncodings = []
knownNames = []

# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # Extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # Load the image and convert it from BGR to RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model=detection_method)

    # Compute the facial embeddings for each face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over the encodings
    for encoding in encodings:
        # Add each encoding and name to the list of known names and encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# Dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}

with open(encodings_path, "wb") as f:
    f.write(pickle.dumps(data))
