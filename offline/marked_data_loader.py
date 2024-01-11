from typing import Optional, List
from dataclasses import dataclass
import numpy as np
import mne


@dataclass
class MarkedRaw:
    raw_filename: str
    mark_filename: str
    need_augmentation: bool


class MarkedDataLoader:
    def __init__(self, marked_raw_list: List[MarkedRaw],
                 with_reference_process: bool = False,
                 with_filter: bool = True, lower_freq: float = 0, upper_freq: float = 100,
                 invert_fp_signal: bool = False, to_remove_channels: Optional[list] = None,
                 use_ica: bool = True, ica_components: int = 8, ica_exclude: Optional[list] = None,
                 normalization_method: Optional[str] = None):
        self.raw_list = []
        for marked_raw in marked_raw_list:
            raw = mne.io.read_raw_edf(marked_raw.raw_filename, preload=True)

            if to_remove_channels:
                raw.drop_channels(to_remove_channels)
            data = raw.get_data()
            if invert_fp_signal:
                data[:2] = -data[:2]
            if with_reference_process:
                data_len = 15 - (0 if not to_remove_channels else len(to_remove_channels))
                data[:data_len] = data[:data_len] - data[data_len]

            info = mne.create_info(ch_names=raw.ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
            raw_new = mne.io.RawArray(data, info)
            raw_new = raw_new.set_eeg_reference(ref_channels="average")

            if with_filter:
                raw_new.filter(lower_freq, upper_freq)

            if use_ica:
                ica = mne.preprocessing.ICA(n_components=ica_components, random_state=0)
                ica.fit(raw_new)
                ica.exclude = ica_exclude
                ica.apply(raw_new)

            self.raw_list.append(raw_new)

        self.X = []
        self.y = []
        for i, raw in enumerate(self.raw_list):
            self.process_raw(raw, marked_raw_list[i].mark_filename, marked_raw_list[i].need_augmentation)

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def normalize_trial(self, trial: np.array) -> np.array:
        mean = np.mean(trial, axis=1, keepdims=True)
        std = np.std(trial, axis=1, keepdims=True)
        return (trial - mean) / std

    def process_raw(self, raw, mark_filename: str, with_augmentation: bool):
        info = raw.info

        data = raw.get_data(picks=info.ch_names)
        hz = 250
        video_start = 15

        with open(mark_filename, 'r') as f:
            for movement_info in f.readlines():
                movement_info = movement_info.split(', ')
                movement_start_sec, movement_end_sec, label = \
                    float(movement_info[0]) + video_start, float(movement_info[1]) + video_start, int(movement_info[2])
                if label in [4, 5]:
                    continue

                self.X.append(self.normalize_trial(data[:, round(movement_start_sec * hz):round(movement_end_sec * hz)]))
                self.y.append(label)

                if with_augmentation:
                    self.X.append(self.normalize_trial(data[:, round((movement_start_sec - 0.5) * hz):round((movement_end_sec - 0.5) * hz)]))
                    self.y.append(label)

                    self.X.append(self.normalize_trial(data[:, round((movement_start_sec + 0.5) * hz):round((movement_end_sec + 0.5) * hz)]))
                    self.y.append(label)

                    self.X.append(self.normalize_trial(data[:, round((movement_start_sec - 1) * hz):round((movement_end_sec - 1) * hz)]))
                    self.y.append(label)

                    self.X.append(self.normalize_trial(data[:, round((movement_start_sec + 1) * hz):round((movement_end_sec + 1) * hz)]))
                    self.y.append(label)
