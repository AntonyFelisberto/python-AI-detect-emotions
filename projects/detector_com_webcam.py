import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import img_to_array

from base64 import b64decode
import numpy as np
import io
from PIL import Image

cascade_faces = "projects\\arquivos\Material\\haarcascade_frontalface_default.xml"
caminho_modelos = "projects\\arquivos\\Material\\modelo_01_expressoes.h5"

def capture_photo():
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Error: Could not open webcam")
        return

    ret, frame = webcam.read()

    if not ret:
        print("Error: Could not read frame")
        return

    webcam.release()

    cv2.imwrite("captured_photo.jpg", frame)
    return np.asarray(Image.open("captured_photo.jpg"))

# Clique na imagem da webcam para tirar uma foto
imagem = capture_photo() 
# Inverte a ordem dos canais (utilizar caso a imagem capturada fique com cores invertidas)
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB) 
cv2.imwrite("testecaptura.jpg",imagem)

# Carrega o modelo
face_detection = cv2.CascadeClassifier(cascade_faces)
classificador_emocoes = load_model(caminho_modelos, compile=False)

expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]
 
original = imagem.copy()
faces = face_detection.detectMultiScale(original,scaleFactor=1.1,minNeighbors=3,minSize=(20,20))
cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
 
if len(faces) > 0:
    for (fX, fY, fW, fH) in faces:
      roi = cinza[fY:fY + fH, fX:fX + fW]
      roi = cv2.resize(roi, (48, 48))
      roi = roi.astype("float") / 255.0
      roi = img_to_array(roi)
      roi = np.expand_dims(roi, axis=0)
      preds = classificador_emocoes.predict(roi)[0]
      print(preds)
      emotion_probability = np.max(preds)
      label = expressoes[preds.argmax()]
      cv2.putText(original, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.rectangle(original, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
else:
    print('Nenhuma face detectada')

probabilidades = np.ones((250, 300, 3), dtype="uint8") * 255
# Mostra gráfico apenas se detectou uma face
if len(faces) == 1:
  for (i, (emotion, prob)) in enumerate(zip(expressoes, preds)):
      # Nome das emoções
      text = "{}: {:.2f}%".format(emotion, prob * 100)
      w = int(prob * 300)
      cv2.rectangle(probabilidades, (7, (i * 35) + 5),
      (w, (i * 35) + 35), (200, 250, 20), -1)
      cv2.putText(probabilidades, text, (10, (i * 35) + 23),
      cv2.FONT_HERSHEY_SIMPLEX, 0.45,
      (0, 0, 0), 1, cv2.LINE_AA)

cv2.imwrite("captura.jpg",original)
cv2.destroyAllWindows()