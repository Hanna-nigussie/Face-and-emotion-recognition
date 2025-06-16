import cv2
import face_recognition
import numpy as np
from deepface import DeepFace

# Load known face encodings and names
known_encodings = []
known_names = []

# Load and encode images
image_files = [
    (r"C:\Users\user\OneDrive\Desktop\facerecognition\hanna.jpg", "Hanna Nigussie"),
    (r"C:\Users\user\OneDrive\Desktop\facerecognition\hailey.jpg", "Hailey Bieber"),
    (r"C:\Users\user\OneDrive\Desktop\facerecognition\kylie.jpg", "Kylie Jenner"),
]

for img_path, name in image_files:
    image = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(name)
    else:
        print(f"Warning: No face found in {img_path}")

# Initialize face cascade for emotion detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert to grayscale for emotion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)  # Convert back to RGB for DeepFace

    # Face detection for recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Face detection for emotion analysis
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"  # Default name
        emotion = "N/A"   # Default emotion

        # Face recognition
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if any(matches):
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]

        # Extract face ROI for emotion detection
        face_roi = rgb_frame[top:bottom, left:right]

        try:
            # Perform emotion analysis
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except Exception as e:
            emotion = "Error"

        # Draw rectangle and labels
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} | {emotion}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Face Recognition & Emotion Detection", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
