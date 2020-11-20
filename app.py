from flask import Flask, render_template, request, send_from_directory
import keras
from keras.preprocessing import image
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
import numpy as np
import os

import sys
from PIL import Image

sys.modules['Image'] = Image

model = Sequential()


model.add(Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))


model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
basedir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(basedir, 'static/covidFinal.h5')
model.load_weights(model_path)




COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save(os.path.join(basedir, 'static/{}.jpg').format(COUNT))

    img_arr = cv2.imread(os.path.join(basedir, 'static/{}.jpg').format(COUNT))
    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224,224,3)

    prediction = model.predict(img_arr, batch_size=10)

    value = round(prediction[0,0],2)
    print(value)
    pred = np.array([value, 1.0-value])
    print(pred)
    COUNT += 1
    return render_template('prediction.html', data=pred)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT - 1))


if __name__ == '__main__':
    app.run(debug=True)
