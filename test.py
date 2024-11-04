import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os

def extract_frames(video_path, frame_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frame = cv2.resize(frame, (299, 299))
            frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames)

data_gen = ImageDataGenerator(rescale=1./255)
train_data = data_gen.flow_from_directory(
    'dataset/train', target_size=(299, 299), batch_size=32, class_mode='binary')
validation_data = data_gen.flow_from_directory(
    'dataset/validation', target_size=(299, 299), batch_size=32, class_mode='binary')

base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // 32,
    validation_data=validation_data,
    validation_steps=validation_data.samples // 32,
    epochs=5)

for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // 32,
    validation_data=validation_data,
    validation_steps=validation_data.samples // 32,
    epochs=1000)

def predict_video(video_path, model, frame_rate=5):
    frames = extract_frames(video_path, frame_rate)
    frames = frames / 255.0
    predictions = model.predict(frames)
    avg_prediction = np.mean(predictions)
    return "Fake" if avg_prediction > 0.5 else "Real"

model.save('deepfake_detector.h5')
