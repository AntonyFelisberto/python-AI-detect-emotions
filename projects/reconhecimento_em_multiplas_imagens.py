import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import img_to_array

imagem = cv2.imread('projects\\images\\multiplas_imagens.png')

cascade_faces = "projects\\arquivos\Material\\haarcascade_frontalface_default.xml"
caminho_modelos = "projects\\arquivos\\Material\\modelo_01_expressoes.h5"

face_detection = cv2.CascadeClassifier(cascade_faces)
classificador_emocoes = load_model(caminho_modelos, compile = False)
expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]


faces = face_detection.detectMultiScale(imagem, scaleFactor = 1.2,
                                        minNeighbors = 5, minSize = (20,20))

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

original = imagem.copy()

for (x, y, w, h) in faces:
  # Extração do ROI (region of interest)  
  roi = cinza[y:y + h, x:x + w] # utilizamos as coordenadas (onde inicia a face) e a largura e altura para extrair a região de interesse

  # Redimensiona imagem
  roi = cv2.resize(roi, (48, 48))

  # Normalização
  roi = roi.astype("float") / 255
  roi = img_to_array(roi)
  roi = np.expand_dims(roi, axis = 0)

  # Previsões
  preds = classificador_emocoes.predict(roi)[0]
  print(preds)

  # Emoção detectada
  emotion_probability = np.max(preds)
  print(emotion_probability)

  print(preds.argmax())
  label = expressoes[preds.argmax()]


  # Mostra resultado na tela para o rosto
  cv2.putText(original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 0, 255), 2, cv2.LINE_AA)
  cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imwrite("asd.jpg",original)