import face_recognition 
import cv2 
import numpy as np  
import csv 
from datetime import datetime  

video_capture = cv2.VideoCapture(0)

shahrukh_image = cv2.imread("faces/shahrukh.jpg")
shahrukh_image = cv2.cvtColor(shahrukh_image, cv2.COLOR_BGR2RGB)  

shahrukh_encoding = face_recognition.face_encodings(shahrukh_image)[0]

haseeb_image = cv2.imread("faces/haseeb.jpg")
haseeb_image = cv2.cvtColor(haseeb_image, cv2.COLOR_BGR2RGB)  

haseeb_encoding = face_recognition.face_encodings(haseeb_image)[0]

salman_image = cv2.imread("faces/salman.jpg")
salman_image = cv2.cvtColor(salman_image, cv2.COLOR_BGR2RGB)  
salman_encoding = face_recognition.face_encodings(salman_image)[0]


known_face_encodings = [shahrukh_encoding, salman_encoding, haseeb_encoding]
known_face_names = ["Shahrukh", "Salman", "Haseeb"]

recognized_students = set()

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

with open(f"{current_date}.csv", "w+", newline="") as f:
    lnwriter = csv.writer(f)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame")
            continue  

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                if name not in recognized_students:
                    recognized_students.add(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])

                cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
