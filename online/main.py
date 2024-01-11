import tensorflow as tf
from mne_realtime import LSLClient
from dataclasses import dataclass
import mne
import cv2
import numpy as np
import threading
import logging
logging.basicConfig(level=logging.WARNING)
mne.set_log_level('WARNING')


@dataclass
class DataWrapper:
    data: np.array


def read_data(data_wrapper: DataWrapper):
    filename = '../data/Artem/для тренировки.edf'
    raw = mne.io.read_raw_edf(filename, preload=True)
    with LSLClient(info=raw.info, host='NBEEG16_1075_Data', wait_max=100) as client:
        client_info = client.get_measurement_info()
        sfreq = int(client_info['sfreq'])

        while True:
            epoch = client.get_data_as_epoch(n_samples=sfreq)
            try:
                data_wrapper.data = np.concatenate((data_wrapper.data, epoch.get_data()[0]), axis=1)
            except AttributeError:
                break


if __name__ == '__main__':
    data_wrapper = DataWrapper(data=np.empty((16, 0)))
    thread = threading.Thread(target=read_data, args=(data_wrapper,))
    thread.start()

    full_movement_cycle_hz = round(9.5 * 250)  # sec * sfreq
    model = tf.keras.models.load_model('../saved_model/')
    wait_img = cv2.imread('../images/wait.png')
    up_img = cv2.imread('../images/imagined_up.png')
    down_img = cv2.imread('../images/imagined_down.png')
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    while True:
        user_input = input("Enter s to make move and e to exit: ")
        if user_input == 's':
            cv2.imshow('Display', wait_img)
            cv2.waitKey(10000)
            cv2.imshow('Display', up_img)
            cv2.waitKey(2500)
            cv2.imshow('Display', wait_img)
            cv2.waitKey(1500)
            cv2.imshow('Display', down_img)
            cv2.waitKey(2500)
            cv2.imshow('Display', wait_img)
            cv2.waitKey(1500)

            data = data_wrapper.data[:, -full_movement_cycle_hz:]
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            data = (data - mean) / std

            data = data.reshape(1, data.shape[0], data.shape[1])
            probs = model.predict(data)
            preds = probs.argmax(axis=-1)
            print(f'Your movement type: {preds[0]}')

            cv2.destroyWindow('Display')

        elif user_input == 'e':
            break
