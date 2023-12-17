import cv2
import face_recognition
import pickle
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, db, storage



cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facetrack-80ed9-default-rtdb.europe-west1.firebasedatabase.app/",
    'storageBucket': "facetrack-80ed9.appspot.com"
})

folderpath = 'images'
pathList = os.listdir(folderpath)
imageList = []
studentIds = []

for path in pathList:
    imageList.append(cv2.imread(os.path.join(folderpath, path)))
    studentIds.append(os.path.splitext(path)[0])
    
    #add to firebase database
    fileName = f'{folderpath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)
#print(studentIds)
    
def findEncodings(imageList):
    encodeList = []
    for img in imageList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    
    return encodeList

print("Start Encoding")
encodeListKnown = findEncodings(imageList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

file = open("EncodeFile.p", "wb")
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")