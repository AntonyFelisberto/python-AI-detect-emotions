import cv2
import numpy as np
import pandas as pd
import zipfile
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import random

"""
Extrair arquivos
path = "projects\\arquivos\\Material.zip"
zip_object = zipfile.ZipFile(path, mode="r")
zip_object.extractall("projects\\arquivos")
zip_object.close()
"""

imagem = cv2.imread("projects\\images\\image.png")

cascade_faces = "projects\\arquivos\Material\\haarcascade_frontalface_default.xml"
caminho_modelos = "projects\\arquivos\\Material\\modelo_01_expressoes.h5"
face_detection = cv2.CascadeClassifier(cascade_faces)
classificador_emocoes = load_model(caminho_modelos,compile=False)
exprecoes = ["Raiva","Nojo","Medo","Feliz","Triste","Surpreso","Neutro"]

print(imagem.shape)

original = imagem.copy()
faces = face_detection.detectMultiScale(original,scaleFactor=1.1,minNeighbors=3,minSize=(20,20))

print(faces)
print(len(faces))
print(faces.shape)

cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
roi = cinza[faces[0][1]:faces[0][1] + 128, faces[0][0]:faces[0][0] + 128] #posicao X:X + 128, Y:Y + 128
print(roi)
print(roi.shape)

roi = cv2.resize(roi,(48,48))
print(roi.dtype)
roi = roi.astype("float")
roi = roi / 255
roi = img_to_array(roi)
print(roi.shape)
roi = np.expand_dims(roi, axis=0) #passando imaem pro formato do tensorflow
print(roi.shape)

preds = classificador_emocoes.predict(roi)[0]
print(preds)
emotion_probability = np.max(preds)
print(emotion_probability)

print(preds.argmax())
label = exprecoes[preds.argmax()]
print(label)

cv2.putText(original, label, (faces[0][0],faces[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255),2,cv2.LINE_AA)
cv2.rectangle(original,(faces[0][0],faces[0][1]),(faces[0][0]+128,faces[0][1]+128),(0,0,255),2)
cv2.imwrite("ima.jpg",original)

probabilidades = np.ones((250,300,3),dtype="uint8") * 255
print(probabilidades)
print(probabilidades.shape)
if len(faces) == 1:
    for (i,(emotion,prob)) in enumerate(zip(exprecoes,preds)):
        text = "{}:{:.2f}".format(emotion,prob * 100)
        w = int(prob * 300)
        cv2.rectangle(probabilidades,(7,(i * 35)+5),(w,(i * 35)),(200,250,20),-1)
        cv2.putText(probabilidades,text,(10,(i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0),1,cv2.LINE_AA)