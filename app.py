from flask import Flask, render_template, Response, url_for, request, redirect
from flask_socketio import SocketIO, emit
import cv2
import face_recognition
import numpy as np
import os
import pickle
from firebase_admin import credentials, db, storage
import firebase_admin
from werkzeug.utils import secure_filename


app = Flask(__name__)
socketio = SocketIO(app)
camera = cv2.VideoCapture(0)

recognized_faces = set()
present_students_count = 0

print("Loading encoded file")
file = open("EncodeFile.p", 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("File loaded")

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facetrack-80ed9-default-rtdb.europe-west1.firebasedatabase.app/",
    'storageBucket': "facetrack-80ed9.appspot.com"
})

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_images_and_ids(folderpath):
    pathList = os.listdir(folderpath)
    imageList = []
    studentIds = []

    for path in pathList:
        imageList.append(cv2.imread(os.path.join(folderpath, path)))
        studentIds.append(os.path.splitext(path)[0])

    return imageList, studentIds

def find_encodings(image_list):
    encode_list = []
    for img in image_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    
    return encode_list

def save_encodings(encode_list, student_ids, filename="EncodeFile.p"):
    encode_list_with_ids = [encode_list, student_ids]
    with open(filename, "wb") as file:
        pickle.dump(encode_list_with_ids, file)

folderpath = 'images'

# loading photos and ids
imageList, studentIds = load_images_and_ids(folderpath)

print("Start Encoding")
# searching for encode faces
encodeListKnown = find_encodings(imageList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

# saving encode data to file
save_encodings(encodeListKnown, studentIds)
print("File Saved")

def recognize_faces(frame):
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex] and studentIds[matchIndex] not in recognized_faces:
            print("Known face detected")
            print(f"Student ID: {studentIds[matchIndex]}")

            recognized_faces.add(studentIds[matchIndex])

            global present_students_count
            present_students_count += 1

            print(f"FaceLocation: {faceLoc}")

            top, right, bottom, left = faceLoc
            bbox = (left * 4, top * 4, (right - left) * 4, (bottom - top) * 4)

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

            socketio.emit('update_present_students_count', present_students_count, namespace='/')

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@socketio.on('reset_present_students_count', namespace='/')
def handle_reset_present_students_count():
    global present_students_count
    present_students_count = 0
    recognized_faces.clear()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add_student', methods=['POST', 'GET'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        surname = request.form['surname']
        index = request.form['index']
        smjer = request.form['smjer']
        email = request.form['email']
        photo = request.files['photo']

        if photo and photo.filename != '':
            filename = secure_filename(photo.filename)
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            photo.save(photo_path)

            # Koristite filename (bez ekstenzije) kao ključ
            student_id = os.path.splitext(filename)[0]

            bucket = storage.bucket()
            blob = bucket.blob(f'images/{filename}')
            blob.upload_from_filename(photo_path)

            photo_url = blob.public_url

            ref = db.reference('Students')
            student_data = {
                'name': name,
                'surname': surname,
                'index': index,
                'smjer': smjer,
                'email': email,
                'total_attendance': 0,
                'last_attendance_time': '',
                'photo_path': f'{UPLOAD_FOLDER}/{filename}', 
                'photo_url': photo_url
            }
            ref.child(student_id).set(student_data)


        return redirect(url_for('index'))

@app.route('/add_student_form')
def add_student_form():
    return render_template('addStudent.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
