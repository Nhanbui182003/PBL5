"""
A script to build a dataset by capturing images from the webcam.
"""

import cv2
import os

# Set the output directory directly
output_dir = "face_recognition_dataset/"
DRIVER_NAME = input("Enter driver name: ").strip()
output_dir += DRIVER_NAME

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize webcam video capture
video = cv2.VideoCapture(0)
total = 0

# Start capturing frames from the webcam
while True:
    ret, frame = video.read()

    if not ret:
        print("[ERROR] Failed to capture image from webcam.")
        break

    # Display the captured frame
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF

    # Save the frame when 'k' key is pressed
    if key == ord("k"):
        filename = "{}.png".format(str(total).zfill(5))  # Zero-pad the file name to 5 digits
        filepath = os.path.sep.join([output_dir, filename])
        cv2.imwrite(filepath, frame)
        total += 1
        print(f"[INFO] Saved {filepath}")

    # Exit the loop when 'q' key is pressed
    elif key == ord("q"):
        break

# Print the total number of images saved
print(f"[INFO] {total} face images stored")

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
