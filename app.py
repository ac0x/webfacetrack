from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os
import pickle

app = Flask(__name__)
camera = cv2.VideoCapture(1)



print("Loading encoded file")
file = open("EncodeFile.p", 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("File loaded")

def recognize_faces(frame):
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            print("Known face detected")
            print(f"Student ID: {studentIds[matchIndex]}")

            print(f"FaceLocation: {faceLoc}")

            top, right, bottom, left = faceLoc
            bbox = (left * 4 + 55, top * 4 + 162, (right - left) * 4, (bottom - top) * 4)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

    return frame

def generate_frames():
    while True:
        success, img = camera.read()
        if not success:
            break
        else:
            img = recognize_faces(img)
            _, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
