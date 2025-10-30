import os
from random import randint

import keras
import librosa
import numpy as np
from keras import layers
import tensorflow as tf
from playsound import playsound

# Задание 3 - свёрточные сети, распознавание типа звуков
# https://www.slingacademy.com/article/audio-classification-using-tensorflows-audio-module/

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    og_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return og_mfcc.T

def pad_mfcc_sequences(mfcc_list):
    max_len = max(m.shape[0] for m in mfcc_list)
    n_features = mfcc_list[0].shape[1]
    padded = np.zeros((len(mfcc_list), max_len, n_features), dtype=np.float32)
    for i, m in enumerate(mfcc_list):
        padded[i, :m.shape[0], :] = m
    return padded

def extract_label(file_name):
    # 0 = air_conditioner
    # 1 = car_horn
    # 2 = children_playing
    # 3 = dog_bark
    # 4 = drilling
    # 5 = engine_idling
    # 6 = gun_shot
    # 7 = jackhammer
    # 8 = siren
    # 9 = street_music
    return str(file_name.split("-")[1])


def get_main_dataset(
        audio_dataset_path: str = "/media/roman/ROMA_S USB/UrbanSound8K/audio",
        start_folder: int = 1,
        end_folder: int = 1
) -> list[list]:
    # mfcc, labels, full path
    dataset = [[], [], []]
    for i in range(start_folder, end_folder + 1):
        fold_name = f"fold{i}"
        for file_name in os.listdir(f"{audio_dataset_path}/{fold_name}"):
            print(f"DIR: {fold_name}, FILE: {file_name}")
            if file_name.endswith(".wav"):
                full_path = f"{audio_dataset_path}/{fold_name}/{file_name}"
                dataset[0].append(extract_mfcc(full_path))
                dataset[1].append(int(extract_label(file_name)))
                dataset[2].append(full_path)
    return dataset


if __name__ == "__main__":
    mfcc_general, labels_general, filepaths_general = get_main_dataset()

    mfcc_general = pad_mfcc_sequences(mfcc_general)

    mfcc_train = np.array(mfcc_general[len(mfcc_general) // 2:])
    labels_train = np.array(labels_general[len(labels_general) // 2:])
    filepaths_train = np.array(filepaths_general[len(filepaths_general) // 2:])

    mfcc_test = np.array(mfcc_general[:len(mfcc_general) // 2])
    labels_test = np.array(labels_general[:len(labels_general) // 2])
    filepaths_test = np.array(filepaths_general[:len(filepaths_general) // 2])

    model = keras.Sequential([
        keras.layers.Input(mfcc_train.shape[1:]),
        keras.layers.Conv1D(16, kernel_size=4, padding="same", activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    model.fit(mfcc_train, labels_train, epochs=100, batch_size=32, validation_split=0.2)

    test_index = randint(0, len(filepaths_test))
    playsound(filepaths_test[test_index])
    print(f"Нейросеть оценила как: {model.predict(np.expand_dims(mfcc_test[test_index], axis=0))[0].argmax()}, правильно: {labels_test[test_index]}")
