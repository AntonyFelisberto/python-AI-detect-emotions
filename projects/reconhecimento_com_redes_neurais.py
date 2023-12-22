import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import tensorflow
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.models import model_from_json
from keras.applications import VGG16
from sklearn.metrics import confusion_matrix
import itertools

def normalizar(x):
    x = x.astype("float32")
    x = x / 255.0
    return faces

def plota_historico_modelo(historico_modelo):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(range(1,len(historico_modelo.history["accuracy"])+1),historico_modelo.history["accuracy"],'r')
    axs[0].plot(range(1,len(historico_modelo.history["val_accuracy"])+1),historico_modelo.history["val_accuracy"],'b')
    axs[0].set_title("Acuracia do modelo")
    axs[0].set_ylabel("Acuracia")
    axs[0].set_xlabel("Epoch")
    axs[0].set_xticks(np.arange(1,len(historico_modelo.history["accuracy"])+1),len(historico_modelo.history["accuracy"])/10)
    axs[0].legend(["training accuracy","validation_accuracy"],loc="best")

    axs[1].plot(range(1,len(historico_modelo.history["loss"])+1),historico_modelo.history["loss"],'r')
    axs[1].plot(range(1,len(historico_modelo.history["val_loss"])+1),historico_modelo.history["val_loss"],'b')
    axs[1].set_title("Loss do modelo")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_xticks(np.arange(1,len(historico_modelo.history["loss"])+1),len(historico_modelo.history["loss"])/10)
    axs[1].legend(["training loss","validation_loss"],loc="best")
    fig.savefig("historico_modelo.png")


data = pd.read_csv("projects\\arquivos\\fer2013\\fer2013.csv")
data.tail()

plt.figure(figsize=(12,6))
plt.hist(data['emotion'],bins=30)
plt.title("imagens x emocoes")
plt.plot()
plt.show()

pixels = data["pixels"].tolist()
largura , altura = 48,48
faces = []
num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48,48
amostras = 0

for pixel_sequencia in pixels:
    face = [int(pixel) for pixel in pixel_sequencia.split(' ')]
    face = np.asarray(face).reshape(largura,altura)
    faces.append(face)

    if amostras < 10:
        cv2.imshow('', face.astype(np.uint8))
        cv2.waitKey(0)
    amostras+=1

print("numero de imagens no dataset ",str(len(faces)))
faces = np.asarray(faces)
print(faces.shape)
faces = np.expand_dims(faces,-1)
print(faces.shape)

emocoes = pd.get_dummies(data["emotion"]).values

x_train, x_test, y_train, y_test = train_test_split(faces,emocoes,test_size=0.1,random_state=42)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=41)

print("numero de imagens no conjunto de trinamento: ",len(x_train))
print("numero de imagens no conjunto de teste: ",len(x_test))
print("numero de imagens no conjunto de valicação: ",len(x_val))

np.save("mod_xtest",x_test)
np.save("mod_ytest",y_test)

# activation PODE SER activation=elu
# PODERIA COLOCAR kernel_initializer = "he_normal"

model = Sequential()
model.add(Conv2D(num_features,kernel_size=(3,3),activation="relu",input_shape=(width,height,1),data_format = "channels_last",kernel_regularizer=l2(0.01))) 
model.add(Conv2D(num_features,kernel_size=(2,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(2*2*2*num_features,activation="relu")) 
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(2*num_features,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(num_labels,activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
arquivo_modelo = "projects\\arquivos\\Material\\modelo_05_expressoes.h5"
arquivo_modelo_json = "projects\\arquivos\\Material\\modelo_01_expressoes.json"

lr_reducer = ReduceLROnPlateau(monitor="val_loss",factor=0.9,patience=3,verbose=1)
early_stopper = EarlyStopping(monitor="val_loss",min_delta=0,patience=8,verbose=1,mode="auto")
checkpointer = ModelCheckpoint(arquivo_modelo,monitor="val_loss",verbose=1,save_best_only=True)

model_json = model.to_json()
with open(arquivo_modelo_json,"w") as json_file:
    json_file.write(model_json)

history = model.fit(np.array(x_train),np.array(y_train),batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(np.array(x_val),np.array(y_val)),shuffle=True,callbacks=[lr_reducer,early_stopper,checkpointer])

print(history.history)
plota_historico_modelo(history)

scores = model.evaluate(np.array(x_test),np.array(y_test),batch_size=batch_size)
print("acuracia ",str(scores[0]))
print("erros ",str(scores[1]))

true_y = []
pred_y = []
x = np.load("mod_xtest.npy")
y = np.load("mod_ytest.npy")

print(y[0],x[0])

json_file = open(arquivo_modelo_json,"r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(arquivo_modelo)

y_pred = loaded_model.predict(x)
yp = y_pred.tolist()
yt = y.tolist()
count = 0

for i in range(len(y)):
    yy = max(yp[i])
    yyt = max(yt[i])
    pred_y.append(yp[i].index(yy))
    true_y.append(yt[i].index(yyt))
    if yp[i].index(yy) == yt[i].index(yyt):
        count += 1

acc = (count / len(y)) * 100

print("acuracia nos testes ", acc)

np.save("truey_mod01",true_y)
np.save("predy_mod01",pred_y)

y_true = np.load("truey_mod01.npy")
y_pred = np.load("predy_mod01.npy")
cm = confusion_matrix(y_true,y_pred)
expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]
titulo = "matriz de confusão"
print(cm)

plt.imshow(cm,interpolation="nearest",cmap=plt.cm.Blues)
plt.title(titulo)
plt.colorbar()
tick_marks = np.arange(len(expressoes))
plt.xticks(tick_marks,expressoes,rotation=45)
plt.yticks(tick_marks,expressoes)
fmt = "d"
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j,i,format(cm[i,j],fmt),horizontalaligment="center",color="white" if cm[i,j] > thresh else "black")

plt.ylabel("classificação correta")
plt.xlabel("predição modelo")
plt.savefig("matriz_confusao_mod01.png")

imagem = cv2.imread("projects\\arquivos\\Material\\testes\\teste02.jpg")
cv2.imshow(imagem)

original = imagem.copy()
gray = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
cv2.imshow(gray)

face_cascade = cv2.CascadeClassifier("projects\\arquivos\\Material\\haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray,1.1,3)

for (x,y,w,h) in faces:
    cv2.rectangle(original,(x,y),(x +w,y+h),(0,255,0),1)
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = roi_gray.astype("float")/255.0
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray,(48,48)),-1),0)
    prediction = loaded_model(cropped_img)[0]
    cv2.putText(original,expressoes[int(np.argmax(prediction))],(x,y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2,cv2.LINE_AA)

cv2.imshow(original)