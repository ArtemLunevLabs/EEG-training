from marked_data_loader import MarkedDataLoader, MarkedRaw
from model import train_cnn_tf

if __name__ == '__main__':
    marked_raw_list = [
        MarkedRaw(raw_filename='drive/MyDrive/eeg_data/data/Artem/для тренировки.edf',
                  mark_filename='drive/MyDrive/eeg_data/markups/mark for all 2 (Олеся, Саша, Влад, Миша).txt',
                  need_augmentation=True),
        MarkedRaw(raw_filename='drive/MyDrive/eeg_data/data/Artem/для тренировки2.edf',
                  mark_filename='drive/MyDrive/eeg_data/markups/mark for all 2 (Олеся, Саша, Влад, Миша).txt',
                  need_augmentation=True),
        MarkedRaw(raw_filename='drive/MyDrive/eeg_data/data/Artem/15-3-5a9a464e-890a-4608-8ab4-b1c02415e563.edf',
                  mark_filename='drive/MyDrive/eeg_data/markups/mark 15-3-5a9a464e-890a-4608-8ab4-b1c02415e563 (Олеся, Влад, Миша).txt',
                  need_augmentation=False),
    ]
    marked_data_loader = MarkedDataLoader(
        marked_raw_list=marked_raw_list,
        with_reference_process=False,
        with_filter=False,
        lower_freq=1,
        upper_freq=50,
        invert_fp_signal=True,
        to_remove_channels=None,
        use_ica=False,
        ica_components=10,
        ica_exclude=[0],
        normalization_method=None
    )
    print(marked_data_loader.X.shape)

    X_train, X_test, y_train, y_test = marked_data_loader.X[:800], marked_data_loader.X[800:], marked_data_loader.y[:800], marked_data_loader.y[800:]
    train_cnn_tf(X_train, X_test, y_train, y_test, enable_logging=True)
