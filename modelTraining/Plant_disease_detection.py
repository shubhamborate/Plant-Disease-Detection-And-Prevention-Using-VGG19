import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

len(os.listdir(r"F:\SHUBHAM BORATE\Capstone\Plant-disease-detector-master final\Plant-disease-detector-master\Dataset\train"))


train_datagen = ImageDataGenerator(zoom_range=0.5, shear_range=0.3, horizontal_flip=True, preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train = train_datagen.flow_from_directory(directory=r"F:\SHUBHAM BORATE\Capstone\Plant-disease-detector-master final\Plant-disease-detector-master\Dataset\train",target_size=(256,256), batch_size=32)

val = val_datagen.flow_from_directory(directory=r"F:\SHUBHAM BORATE\Capstone\Plant-disease-detector-master final\Plant-disease-detector-master\Dataset\val", target_size=(256,256), batch_size=32)


from keras.layers import Dense, Flatten
from keras.models import Model
import keras
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

base_model = VGG19(input_shape=(256, 256, 3), include_top=False)

for layer in base_model.layers:
  layer.trainable=False

X = Flatten()(base_model.output)

X = Dense(units=4, activation='softmax')(X)

model = Model(base_model.input, X)


model.summary()

model.compile(optimizer='adam', loss = keras.losses.categorical_crossentropy, metrics=['accuracy'])




from keras.callbacks import ModelCheckpoint, EarlyStopping


es = EarlyStopping(monitor = 'val_accuracy', min_delta = 0.01, patience = 3, verbose = 1)


mc = ModelCheckpoint(filepath = "best_model.h5", min_delta = 0.01, patience = 3, verbose = 1, save_best_onlu = True)

cb = [es, mc]


his = model.fit_generator(train, steps_per_epoch=2, epochs = 2, verbose = 1, callbacks = cb, validation_data=val, validation_steps=16)
 
h = his.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c = "red")
plt.title("acc vs val_acc")
plt.show()


plt.plot(h['loss'])
plt.plot(h['val_loss'], c = "red")
plt.title("loss vs val_loss")
plt.show()




from keras.models import load_model

model = load_model(r"F:/SHUBHAM BORATE/Capstone/Plant-disease-detector-master final/Plant-disease-detector-master/modelTraining/best_model.h5")



acc = model.evaluate(val)[1]

print(f"The accuracy of your model is = {acc*100} %")


ref = dict(zip( list(train.class_indices.values()), list(train.class_indices.keys())) )


def prediction(path):

  img = load_img(path, target_size = (256,256))

  i = img_to_array(img)

  im = preprocess_input(i)

  img = np.expand_dims(im, axis = 0)

  pred = np.argmax(model.predict(img))

  print(f" the image belongs to { ref[pred] } ")

'''
path = r"C:/Users/VISHAL/Desktop/Plant-disease-detector-master/Plant-disease-detector-master/Dataset/test/black_measles100.jpg"

prediction(path)
'''
  

















