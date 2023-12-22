import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image
import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.models import model_from_json
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools

diretorio = 'projects\\arquivos'

diretorio_treinamento = diretorio + '\\fer2013\\train'

# aqui vamos adicionar todas as emoções, que correspondem a cada uma das pastas dentro do diretorio_treinamento
classes = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] 
# (antes tinha a pasta Disgust mas nessa demonstração resolvemos remover pois tinha poucas imagens)
# Obs: deixar os items da lista na ordem desejada (exemplo: Angry será a emoção com id 0)

total = []        # variavel que irá conter o diretório de todas as imagens da pasta
emotions = []     # lista das emoções
nome_img = []     # lista com o nome das imagens (apenas para visualizar na tabela)
nome_emotion = [] # lista das emoções mas nome ao invés de id (apenas para visualizar na tabela) 


maximo_fotos = 3000 # (opcional) define o numero máximo de fotos que serão carregadas do diretório para cada classe

for i, classe in enumerate(classes):
  imgs_treinamento = glob.glob(diretorio_treinamento + classe + "/*") # verifica todos os arquivos dentro da pasta 

  a = 0   # variavel auxiliar para controlar dentro do for quantas fotos serão carregadas para cada emoção
  ### isso é opcional. nesse exemplos limitamos para 2000 fotos (total 12000) pois se fossem todas ia levar muito tempo
  ### se for carregar todas as fotos deixe a = 99999 por exemplo ou remova o (if a >= maximo_fotos: break)

  for img in imgs_treinamento:
    total.append(img);
    nome_img.append(str(img.split("/")[-1]))
    emotions.append(i)
    nome_emotion.append(classe)
    a = a+1
    if a >= maximo_fotos:
      break

  #imgs_treinamento.remove(diretorio_treinamento + classe + '\\Thumbs.db')
  print("Numero de imagens com a emoção "+ classe +" = "+str(len(imgs_treinamento)))


dataset = pd.DataFrame()

dataset["img"] = nome_img
dataset["emotion"] = emotions
dataset["class"] = nome_emotion

print(dataset)

largura, altura = 48, 48

faces = []
amostras = 0 
t = time.time()
t_total = time.time()

for face in total:
  #imagem = Image.open(face).convert('L')
  imagem = cv2.imread(face, 0)
  imagem = np.asarray(imagem).reshape(largura, altura) 
  #imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
  imagem = cv2.resize(imagem, (largura, altura))

  amostras = amostras + 1
  faces.append(imagem)

  if (amostras % 200 == 0): # pra mostrar o progresso
    print(str(amostras) + " fotos carregadas [+ " + str(round(time.time() - t, 2)) + "s]")
    t = time.time()
  
  if (amostras <= 5):
    cv2.imshow(imagem) # exibe as primeiras 5 imagens só, pra não sobrecarregar o colab

print("Tempo total: " + str(time.time() - t_total))

len(faces)

dataset.head()
dataset.tail()

plt.figure(figsize=(12,6))
plt.hist(dataset['emotion'], bins=11)
plt.title("Imagens x emoção")
plt.show()

faces = np.asarray(faces) 
faces = np.expand_dims(faces, -1)

def normalizar(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

faces = normalizar(faces)

#print(emotions)
#print(faces)
#print(emocoes)

print("Número total de imagens no dataset: "+str(len(faces)))

emocoes = pd.get_dummies(dataset['emotion']).as_matrix() 

x_train, x_test, y_train, y_test = train_test_split(faces, emocoes, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=41)

print("Número de imagens no conjunto de treinamento:", len(x_train))
print("Número de imagens no conjunto de testes:", len(x_test))
print("Número de imagens no conjunto de validação:", len(y_val))

np.save(diretorio + 'mod_xtest_alt', x_test)
np.save(diretorio + 'mod_ytest_alt', y_test)

num_features = 32
num_classes = 6
width, height = 48, 48
batch_size = 16
epochs = 70

model = Sequential()

model.add(Conv2D(num_features, (3, 3), padding = 'same', kernel_initializer="he_normal",
                 input_shape = (width, height, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(num_features, (3, 3), padding = "same", kernel_initializer="he_normal", 
                 input_shape = (width, height, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(2*num_features, kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(2*num_features, kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes, kernel_initializer="he_normal"))
model.add(Activation("softmax"))

print(model.summary())

datagen = ImageDataGenerator(
      rotation_range=10,
      shear_range=0.1,
      zoom_range=0.1,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
      fill_mode='nearest')

datagen = ImageDataGenerator() 

# A classe ImageDataGenerator do Keras retornará apenas os dados transformados aleatoriamente.
# Ou seja, as imagens originais não serão adicionadas, ou somadas junto às imagens geradas por transformações.

# É feito assim pois o objetivo da técnica é garantir que a rede neural veja "novas" imagens que nunca foram vistas antes em cada epoch.
# Se incluíssemos as imagens originais de treinamento junto com as imagens geradas em cada lote (batch), 
#    a rede "veria" os dados originais de treinamento várias vezes, o que não é o objetivo. 
# O objetivo geral do data augmentation é aumentar a generalização do modelo.
# Usando o data augmentation somos capazes de diminuir ou até mesmo previnir o overfitting (sobreajuste)

# Se o batch_size for 32 por exemplo, ImageDataGenerator () retorna 32 imagens aplicando transformações aleatórias usadas para treinar.
# Obs: as operações de aumento de dados são feitas na memória, então as imagens geradas são descartadas logo em seguida.

print(len(datagen.flow(x_train, y_train)))

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
arquivo_modelo = diretorio + "modelo_02_expressoes_alt.h5" # arquivo do modelo
arquivo_modelo_json = diretorio + "modelo_02_expressoes_alt.json" # arquivo do json, para salvar a arquitetura
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(arquivo_modelo, monitor='val_loss', verbose=1, save_best_only=True)

model_json = model.to_json()
with open(arquivo_modelo_json, "w") as json_file:
    json_file.write(model_json)

np.array(x_train).shape # set de treinamento

history = model.fit(np.array(x_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(x_val), np.array(y_val)),
          shuffle=True,
          callbacks=[lr_reducer, early_stopper, checkpointer])

# usarmos fit_generator() ao invés de fit() porque os dados de treinamento vieram de um gerador
history = model.fit_generator(
          datagen.flow(x_train, y_train, batch_size=batch_size),
          epochs=epochs,
          verbose=1,
          validation_data= (x_val, y_val),
          validation_steps = len(x_val) // batch_size, 
          steps_per_epoch = len(x_train) // batch_size,
          callbacks=[lr_reducer, early_stopper, checkpointer])

def plota_historico_modelo(historico_modelo):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(range(1,len(historico_modelo.history['accuracy'])+1),
                historico_modelo.history['accuracy'],'r')
    axs[0].plot(range(1,len(historico_modelo.history['val_accuracy'])+1),
                historico_modelo.history['val_accuracy'],'b')
    axs[0].set_title('Acurácia do Modelo')
    axs[0].set_ylabel('Acuracia')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(historico_modelo.history['accuracy'])+1),
                      len(historico_modelo.history['accuracy'])/10)
    axs[0].legend(['training accuracy', 'validation accuracy'], loc='best')

    axs[1].plot(range(1,len(historico_modelo.history['loss'])+1),
                historico_modelo.history['loss'],'r')
    axs[1].plot(range(1,len(historico_modelo.history['val_loss'])+1),
                historico_modelo.history['val_loss'],'b')
    axs[1].set_title('Perda/Loss do Modelo')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(historico_modelo.history['loss'])+1),
                      len(historico_modelo.history['loss'])/10)
    axs[1].legend(['training loss', 'validation Loss'], loc='best')
    fig.savefig('historico_modelo_mod02_alt.png')
    plt.show()

plota_historico_modelo(history)

scores = model.evaluate(np.array(x_test), np.array(y_test), batch_size=batch_size)
print("Acurácia: " + str(scores[1]))
print("Perda/Loss: " + str(scores[0]))

true_y=[]
pred_y=[]
x = np.load(diretorio + 'mod_xtest_alt.npy')
y = np.load(diretorio + 'mod_ytest_alt.npy')
json_file = open(arquivo_modelo_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(arquivo_modelo)
y_pred= loaded_model.predict(x)
yp = y_pred.tolist()
yt = y.tolist()
count = 0
for i in range(len(y)):
    yy = max(yp[i])
    yyt = max(yt[i])
    pred_y.append(yp[i].index(yy))
    true_y.append(yt[i].index(yyt))
    if(yp[i].index(yy)== yt[i].index(yyt)):
        count+=1
acc = (count/len(y))*100
np.save(diretorio + 'truey__mod_alt', true_y)
np.save(diretorio + 'predy__mod_alt', pred_y)
print("Acurácia no conjunto de testes: "+str(acc)+"%")

y_true = np.load(diretorio + 'truey__mod_alt.npy')
y_pred = np.load(diretorio + 'predy__mod_alt.npy')
cm = confusion_matrix(y_true, y_pred)
expressoes = ["Raiva", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]
titulo='Matriz de Confusão'
print(cm)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(titulo)
plt.colorbar()
tick_marks = np.arange(len(expressoes))
plt.xticks(tick_marks, expressoes, rotation=45)
plt.yticks(tick_marks, expressoes)
fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Classificação Correta')
plt.xlabel('Predição')
plt.savefig('matriz_confusao_mod_alt.png')
plt.show()

imagem = cv2.imread(diretorio + "testes/teste02.jpg")
cv2.imshow(imagem)

model = load_model(diretorio + "modelo_02_expressoes_alt.h5")
scores = model.evaluate(np.array(x_test), np.array(y_test), batch_size=batch_size)
print("Perda/Loss: " + str(scores[0]))
print("Acurácia: " + str(scores[1]))

expressoes = ["Raiva", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"] #"Nojo", 
original = imagem.copy()
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(diretorio + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.2, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 1)
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = roi_gray.astype("float") / 255.0
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, 
                  norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
    prediction = model.predict(cropped_img)[0]
    cv2.putText(original, expressoes[int(np.argmax(prediction))], (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow(original)