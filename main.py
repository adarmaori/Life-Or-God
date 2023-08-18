import tensorflow as tf
import cv2
import numpy as np
import imageio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

SIZE = 64

def load_video(file_name):
    reader = imageio.get_reader(filename)
    frames = np.array([np.array(frame) for frame in reader])

    return frames

filename = 'output.mov'
frames = load_video(filename)
frames = frames[:, :, :, 0]
X_train = frames[0:-1, :SIZE, :SIZE] / 255
Y_train = frames[1::, :SIZE, :SIZE] / 255
print(X_train.shape, Y_train.shape)
model = Sequential([
    Flatten(input_shape=(SIZE, SIZE)),
    Dense(SIZE*SIZE, activation='relu'),
    Dense(SIZE*SIZE, activation='sigmoid'),
    Reshape((SIZE, SIZE))
])
print("Created model")

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print("Model compiled")

model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_split=0.2)
print("Model Trained")

model.save('model.h5')
print("Model Saved")
