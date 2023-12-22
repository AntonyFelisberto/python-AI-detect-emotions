import cv2
import numpy as np
import pandas as pd
import zipfile
from keras.models import load_model
import operator
from keras.models import load_model
from keras.preprocessing.image import img_to_array

path = "projects\\arquivos\\Material.zip"
zip_object = zipfile.ZipFile(file=path, mode="r")
zip_object.extractall("./")
zip_object.close

arquivos_modelos = ["modelo_01_expressoes.h5", "modelo_02_expressoes.h5", "modelo_03_expressoes.h5", "modelo_04_expressoes.h5", "modelo_05_expressoes.h5"]

modelos = {}

x_test = np.load('mod_xtest.npy')
y_test = np.load('mod_ytest.npy')

for modelo in arquivos_modelos:
  model = load_model('projects\\arquivos\\Material\\' + modelo)

  scores = model.evaluate(np.array(x_test), np.array(y_test), batch_size=64)
  print("---"+ str(modelo) +"---")
  print("Perda/Loss: " + str(scores[0]))
  print("Acurácia: " + str(scores[1]))
  modelos[modelo] = str(scores[1])
  print("\n")

order_modelos = sorted(modelos.items(), key=operator.itemgetter(1), reverse=True)
print(order_modelos)

imagem = cv2.imread("projects\\arquivos\\Material\\testes\\teste_gabriel.png")
cv2.imshow(imagem)

cascade_faces = 'projects\\arquivos\\Material\\haarcascade_frontalface_default.xml'
caminho_modelo = 'projects\\arquivos\\Material' + str(order_modelos[0][0])
face_detection = cv2.CascadeClassifier(cascade_faces)
classificador_emocoes = load_model(caminho_modelo, compile=False)
expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]

face_detection = cv2.CascadeClassifier(cascade_faces)
classificador_emocoes = load_model(caminho_modelo, compile=False)

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

cv2.imshow(original)

probabilidades = np.ones((250, 300, 3), dtype="uint8") * 255
if len(faces) == 1:
  for (i, (emotion, prob)) in enumerate(zip(expressoes, preds)):
      text = "{}: {:.2f}%".format(emotion, prob * 100)
      w = int(prob * 300)
      cv2.rectangle(probabilidades, (7, (i * 35) + 5),
      (w, (i * 35) + 35), (200, 250, 20), -1)
      cv2.putText(probabilidades, text, (10, (i * 35) + 23),
      cv2.FONT_HERSHEY_SIMPLEX, 0.45,
      (0, 0, 0), 1, cv2.LINE_AA)

  cv2.imshow(probabilidades)

cv2.imwrite("captura.jpg",original)
cv2.destroyAllWindows()