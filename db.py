import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facetrack-80ed9-default-rtdb.europe-west1.firebasedatabase.app/"
})

ref = db.reference('Students')

data = {
    "1111":
        {
            "name": "Aleksandar Vesovic",
            "smjer": "FIST",
            "index": "21/100",
            "total_attendance": 6,
            "standing": "6",
            "year": 4,
            "last_attendance_time": "2023-11-29 01:53:30"
        }
}

for key, value in data.items():
    ref.child(key).set(value)


