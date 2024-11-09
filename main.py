import face_recognition  # noqa
import cv2  # noqa
import numpy as np  # noqa
import csv  # noqa
from datetime import datetime  # noqa

# For the first webcam
video_capture = cv2.VideoCapture(0)

# Load Known Faces using OpenCV, then convert to RGB
shahrukh_image = cv2.imread("faces/shahrukh.jpg")
shahrukh_image = cv2.cvtColor(shahrukh_image, cv2.COLOR_BGR2RGB)  # Ensure RGB format

shahrukh_encoding = face_recognition.face_encodings(shahrukh_image)[0]

haseeb_image = cv2.imread("faces/haseeb.jpg")
haseeb_image = cv2.cvtColor(haseeb_image, cv2.COLOR_BGR2RGB)  # Ensure RGB format

haseeb_encoding = face_recognition.face_encodings(haseeb_image)[0]

salman_image = cv2.imread("faces/salman.jpg")
salman_image = cv2.cvtColor(salman_image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
salman_encoding = face_recognition.face_encodings(salman_image)[0]

# Store known face encodings and names
known_face_encodings = [shahrukh_encoding, salman_encoding, haseeb_encoding]
known_face_names = ["Shahrukh", "Salman", "Haseeb"]

# Set to track recognized students
recognized_students = set()

# Get current date for the CSV file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Use 'with' to ensure file is properly closed
with open(f"{current_date}.csv", "w+", newline="") as f:
    lnwriter = csv.writer(f)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame")
            continue  # Skip this iteration and try to capture again

        # Resize frame for faster processing and convert it to RGB
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Recognize faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Write to the CSV file if the student is recognized and not already recorded
                if name not in recognized_students:
                    recognized_students.add(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])

                # Display name on the video frame
                cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with recognized names
        cv2.imshow("Attendance", frame)

        # Break out of the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
