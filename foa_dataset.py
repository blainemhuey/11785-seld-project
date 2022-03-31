import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchaudio

SOUND_EVENT_CLASSES = [
    "Female speech, woman speaking",
    "Male speech, man speaking",
    "Clapping",
    "Telephone",
    "Laughter",
    "Domestic sounds",
    "Walk, footsteps",
    "Door, open or close",
    "Music",
    "Musical instrument",
    "Water tap, faucet",
    "Bell",
    "Knock"
]


class FOADataset(Dataset):
    """
    Custom PyTorch Dataset for DCASE FOA Datsets

    TODO: CURRENTLY IMPLEMENTS MEL SPECTRA AND INTENSITY VECTORS (i.e. SELDNet Inputs)
    """

    def __init__(self, data_path, folds=None, train=True):
        """
        Init Function for FOADataset
        :param data_path: String path to root folder containing 'foa_dev' and 'metadata_dev'
        :param folds: List of fold integers to use in this dataset
        :param train: Bool indicating whether to use train dataset of val dataset
        """

        # Calculate Directory Names
        foa_directory_sony = osp.join(data_path, "foa_dev", "dev-train-sony" if train else "dev-test-sony")
        meta_directory_sony = osp.join(data_path, "metadata_dev", "dev-train-sony" if train else "dev-test-sony")
        foa_directory_tau = osp.join(data_path, "foa_dev", "dev-train-tau" if train else "dev-test-tau")
        meta_directory_tau = osp.join(data_path, "metadata_dev", "dev-train-tau" if train else "dev-test-tau")

        all_foa_files = [osp.join(foa_directory_tau, file) for file in os.listdir(foa_directory_tau)]
        all_foa_files.extend([osp.join(foa_directory_sony, file) for file in os.listdir(foa_directory_sony)])
        all_meta_files = [osp.join(meta_directory_tau, file) for file in os.listdir(meta_directory_tau)]
        all_meta_files.extend([osp.join(meta_directory_sony, file) for file in os.listdir(meta_directory_sony)])

        # Parse File Names
        foa_file_data = [self.parse_foa_file_name(file) for file in all_foa_files]
        meta_file_data = [self.parse_foa_file_name(file) for file in all_meta_files]

        # Create Lists of All Valid File Paths in Given Folds
        self.folds = folds
        self.foa_files = [
            file for file, data in zip(all_foa_files, foa_file_data)
            if (folds is None or data["fold"] in folds)
        ]
        self.foa_files.sort()
        self.meta_files = [
            file for file, data in zip(all_meta_files, meta_file_data)
            if (folds is None or data["fold"] in folds)
        ]
        self.meta_files.sort()
        assert len(self.foa_files) == len(self.meta_files)

        # TODO: Add Options for Other Input Features
        # Load SELDNet Input Features and ACCDOA Output
        features = []
        multi_accdoas = []
        for foa_file, meta_file in zip(self.foa_files, self.meta_files):
            feature = self.audio_to_foa_features(foa_file)
            multi_accdoa = self.metadata_to_multi_accdoa(self.load_metadata(meta_file),
                                                         total_frames=feature.shape[2])
            features.append(feature)
            multi_accdoas.append(multi_accdoa)
        self.features = torch.concat(features, dim=-1)
        self.multi_accdoa = np.concatenate(multi_accdoas, axis=-1)

    @staticmethod
    def parse_foa_file_name(file):
        """
        Parses filenames of the following format:
        "fold[fold number]_room[room number per fold]_mix[recording number per room per split].wav"
        :param file: filename
        :return: metadata dictionary
        """

        name, extension = osp.splitext(osp.basename(file))
        fold_text, room_text, mix_text = name.split("_")
        fold = int(fold_text.replace("fold", ""))
        room = int(room_text.replace("room", ""))
        mix = int(mix_text.replace("mix", ""))
        return {"fold": fold, "room": room, "mix": mix}

    @staticmethod
    def audio_to_foa_features(file, fft_size=1024, hop_length=20):
        """
        Generates the SELDNet Input Features
        :param file: Filepath to Audio File to Load
        :param fft_size: Size of FFT calculation to perform
        :param hop_length: Stride of FFT in ms
        :return: torch.Tensor of Shape 7x64xT
        """
        waveform, sample_rate = torchaudio.load(file, normalize=True)

        spec_trans = torchaudio.transforms.Spectrogram(n_fft=fft_size, hop_length=sample_rate // (1000 // hop_length),
                                                       pad=0, power=None)
        mel_trans = torchaudio.transforms.MelScale(n_mels=64, sample_rate=sample_rate, n_stft=fft_size // 2 + 1)

        with torch.no_grad():
            spectrogram = spec_trans(waveform)
            mel_spec = mel_trans(torch.real(torch.pow(spectrogram, 2)))

            intensity = torch.real(torch.conj(spectrogram[0]) * spectrogram[1:])
            intensity = intensity / torch.norm(intensity, dim=0)  # TODO: Check shape
            mel_intensity = mel_trans(intensity)
        return torch.concat((mel_spec, mel_intensity), dim=0)

    @staticmethod
    def load_metadata(file):
        """
        Reads in the CSV Label File of the Format
        '[frame number (int)], [active class index (int)], [source number index (int)], [azimuth (int)], [elevation (int)]'

        :param file: Filepath to CSV File to Load
        :return: List of Metadata Dictionaries
        """
        metadata = []
        with open(file, 'r') as f:
            for line in f.readlines():
                frame_number, active_class, source_number, azimuth, elevation = line.split(",")
                metadata.append({
                    "frame_number": int(frame_number),
                    "active_class": int(active_class),
                    "source_number": int(source_number),
                    "azimuth": int(azimuth),
                    "elevation": int(elevation)
                })
        return metadata

    @staticmethod
    def metadata_to_multi_accdoa(metadata, total_frames, n=3, c=len(SOUND_EVENT_CLASSES)):
        multi_accdoa = np.zeros((3, n, c, total_frames))
        event_count_per_frame = np.zeros((c, total_frames), dtype=np.int)
        for metadata_i in metadata:
            f, a, s, az, el = (metadata_i["frame_number"], metadata_i["active_class"], metadata_i["source_number"],
                               metadata_i["azimuth"], metadata_i["elevation"])
            norm_az_el = np.array([[np.cos(np.deg2rad(az))], [np.sin(np.deg2rad(az))], [np.sin(np.deg2rad(el))]])
            multi_accdoa[:, event_count_per_frame[a, f]:, a, f] = norm_az_el
            event_count_per_frame[a, f] += 1
        return multi_accdoa

    def __len__(self):
        return self.features.shape[-1]

    def __getitem__(self, item):
        return self.features[:, :, item], self.multi_accdoa[:, :, :, item]
