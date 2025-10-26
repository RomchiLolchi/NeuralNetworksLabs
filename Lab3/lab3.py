import os
import keras
import librosa
import numpy as np
from keras import layers
import tensorflow as tf

# Задание 3 - свёрточные сети, распознавание типа звуков
# https://www.slingacademy.com/article/audio-classification-using-tensorflows-audio-module/
# Текущая проблема - подготовка единого датасета! Почему-то не совпадают размерности, есть mfcc_general
# записи, где просто какая-то хрень происходит, надо бы подкрутить посмотреть!!!!

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    og_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    trimmed_mfcc = [[], []]
    for i in og_mfcc:
        if len(i) < 300:
            continue
        trimmed_mfcc.append(i[:300])
    return trimmed_mfcc


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
        audio_dataset_path: str = "D:\\UrbanSound8K\\audio",
        start_folder: int = 1,
        end_folder: int = 1
) -> list[list]:
    # mfcc, labels, full path
    dataset = [[], [], []]
    for i in range(start_folder, end_folder + 1):
        fold_name = f"fold{i}"
        for file_name in os.listdir(f"{audio_dataset_path}\\{fold_name}"):
            print(f"DIR: {fold_name}, FILE: {file_name}")
            if file_name.endswith(".wav"):
                full_path = f"{audio_dataset_path}\\{fold_name}\\{file_name}"
                dataset[0].append(extract_mfcc(full_path))
                dataset[1].append(extract_label(file_name))
                dataset[2].append(full_path)
    return dataset


if __name__ == "__main__":
    mfcc_general, labels_general, filepaths_general = get_main_dataset()
    # mfcc_general = []

    # max_second_shape = -1
    # for i in range(len(old_mfcc_general)):
    #     try:
    #         second_shape = np.array(i).shape[1]
    #         if second_shape > max_second_shape:
    #             max_second_shape = second_shape
    #     except Exception:
    #         continue
    #
    # for i in range(len(old_mfcc_general)):
    #     try:
    #         mfcc_general.append(np.array(i).reshape((40, max_second_shape)))
    #     except ValueError:
    #         continue

    mfcc_train = tf.convert_to_tensor(mfcc_general[len(mfcc_general) // 2:], dtype=tf.float32)
    labels_train = tf.convert_to_tensor(labels_general[len(labels_general) // 2:], dtype=tf.float32)
    filepaths_train = tf.convert_to_tensor(filepaths_general[len(filepaths_general) // 2:], dtype=tf.float32)

    mfcc_test = mfcc_general[:len(mfcc_general) // 2]
    labels_test = labels_general[:len(labels_general) // 2]
    filepaths_test = filepaths_general[:len(filepaths_general) // 2]

    model = keras.Sequential([
        keras.layers.Conv1D(16, kernel_size=3, activation='relu', input_shape=np.array(mfcc_train).shape[1:]),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
        # keras.layers.Conv1D(32, 3, activation='relu', input_shape=np.array(mfcc_train).shape),
        # keras.layers.MaxPooling1D(2, 2),
        # keras.layers.Dropout(0.25),
        #
        # keras.layers.Conv1D(64, 3, activation='relu'),
        # keras.layers.MaxPooling1D(2, 2),
        # keras.layers.Dropout(0.25),
        #
        # keras.layers.Flatten(),
        # keras.layers.Dense(128, activation='relu'),
        # keras.layers.Dropout(0.5),
        #
        # keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(mfcc_train, labels_train, epochs=10, batch_size=32, validation_split=0.2)
