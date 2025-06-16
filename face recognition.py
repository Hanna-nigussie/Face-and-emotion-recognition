import cv2
import face_recognition
import numpy as np
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  

# Initialize known encodings and names
known_encodings = []
known_names = []

# Load and encode images 
image_files = [
    (r"C:\Users\user\OneDrive\Desktop\facerecognition\hanna.jpg", "Hanna Nigussie"),
    (r"C:\Users\user\OneDrive\Desktop\facerecognition\hailey.jpg", "Hailey Bieber"),
    (r"C:\Users\user\OneDrive\Desktop\facerecognition\kylie.jpg", "Kylie Jenner"),
    (r"C:\Users\user\OneDrive\Desktop\facerecognition\john.jpg", "Yohannes dessie"),
]

for img_path, name in image_files:
    image = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(name)
    else:
        print(f"Warning: No face found in {img_path}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Variables to track mouth movement over time
previous_mouth_height = None
mouth_moving = False
movement_threshold = 1

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect face locations
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(frame)

    for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
        name = "Unknown"  

        # Compare faces & get closest match
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if any(matches):
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]

        # Get mouth landmarks
        top_lip = landmarks["top_lip"]
        bottom_lip = landmarks["bottom_lip"]

        # Visualize the landmarks (for debugging)
        for point in top_lip:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)  # Green points for top lip
        for point in bottom_lip:
            cv2.circle(frame, tuple(point), 2, (0, 0, 255), -1)  # Red points for bottom lip

        # Calculate the vertical distance between top and bottom lips (mouth opening)
        top_lip_center = np.mean(top_lip, axis=0)  # Center of the top lip
        bottom_lip_center = np.mean(bottom_lip, axis=0)  # Center of the bottom lip
        mouth_height = np.linalg.norm(top_lip_center - bottom_lip_center)

        if previous_mouth_height is not None:
            # Check if the mouth height has changed (even the slightest difference)
            mouth_moving = abs(mouth_height - previous_mouth_height) > movement_threshold

        previous_mouth_height = mouth_height

        # Display the mouth height value for debugging (helps to visualize movements)
        #cv2.putText(frame, f"Mouth Height: {mouth_height:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # If mouth is moving (person is talking), display the message
        if mouth_moving:
            cv2.putText(frame, "Mouth Moving - Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Mouth Not Moving", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw rectangle and name label for face recognition
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Face and Mouth Movement Detection", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
