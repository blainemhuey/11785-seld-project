import os
import os.path as osp

import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from torch.nn.functional import pad

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
    """

    implemented_model_features = ["seldnet", "rd3net"]

    def __init__(self, data_path, folds=None, train=True, model="seldnet", hop_length=20, context=0):
        """
        Init Function for FOADataset
        :param data_path: String path to root folder containing 'foa_dev' and 'metadata_dev'
        :param folds: List of fold integers to use in this dataset
        :param train: Bool indicating whether to use train dataset of val dataset
        """

        # Assert that requested features are currently implemented
        assert model in FOADataset.implemented_model_features
        feat_size = 250
        hop_length = 20
        self.model = model
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

        # Load SELDNet Input Features and ACCDOA Output
        features = []
        multi_accdoas = []
        self.feature_width = 100 // hop_length
        for foa_file, meta_file in zip(self.foa_files, self.meta_files):
            if model == "seldnet":
                feature = self.audio_to_seldnet_features(foa_file, hop_length=hop_length)[:,:,:-1]
                multi_accdoa = self.metadata_to_multi_accdoa(self.load_metadata(meta_file),
                                                         total_frames=feature.shape[2] // (100 // 20))[:,:,:-1]
                feature_chunked = self.chunk_seldnet_feature(feature, feat_size)
                multi_accdoa_chunked = self.chunk_seldnet_multiaccdoa(multi_accdoa, feat_size, hop_length )
                assert(len(feature_chunked) == len(multi_accdoa_chunked))
                features.extend(feature_chunked)
                multi_accdoas.extend(multi_accdoa_chunked)

            else:
                feature = self.audio_to_rd3net_features(foa_file, hop_length=hop_length)
                total_frames = feature.shape[2] // (100 // hop_length)
                feature = feature[:, :, :total_frames * (100 // hop_length)]
                multi_accdoa = self.metadata_to_multi_accdoa(self.load_metadata(meta_file),
                                                             total_frames=total_frames)
                features.append(feature)
                multi_accdoas.append(multi_accdoa)

        if model=="seldnet":
            self.features = np.stack(features)
            self.multi_accdoa = np.stack(multi_accdoas)
        else:
            self.features = pad(torch.concat(features, dim=-1), (context, context))
            self.multi_accdoa = np.concatenate(multi_accdoas, axis=-1)
        self.context = context

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
    def audio_to_seldnet_features(file, fft_size=1024, hop_length=20, eps=1e-8):
        """
        Generates the SELDNet Input Features
        :param file: Filepath to Audio File to Load
        :param fft_size: Size of FFT calculation to perform
        :param hop_length: Stride of FFT in ms
        :param eps: Division eps to prevent NaN outputs
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
            intensity = intensity / (torch.pow(torch.abs(spectrogram[0]), 2) +
                                     torch.mean(torch.pow(torch.abs(spectrogram[1:]), 2), dim=0) + eps)
            mel_intensity = mel_trans(intensity)
        return torch.concat((mel_spec, mel_intensity), dim=0)

    @staticmethod
    def audio_to_rd3net_features(file, fft_size=1024, hop_length=20):
        """
        Generates the RD3Net Input Features
        :param file: Filepath to Audio File to Load
        :param fft_size: Size of FFT calculation to perform
        :param hop_length: Stride of FFT in ms
        :return: torch.Tensor of Shape 7x(fft/2+1)xT
        """
        waveform, sample_rate = torchaudio.load(file, normalize=True)

        spec_trans = torchaudio.transforms.Spectrogram(n_fft=fft_size, hop_length=sample_rate // (1000 // hop_length),
                                                       pad=0, power=None)

        with torch.no_grad():
            spectrogram = spec_trans(waveform)

            amplitude = torch.abs(spectrogram)
            ipd = torch.angle(spectrogram[0]) - torch.angle(spectrogram[1:])

        return torch.concat((amplitude, ipd), dim=0)

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
        """
        Turns a List of Python Dictionaries with SELD Labels Into A Multi-ACCDOA Truth Vector
        :param metadata: List of Python Dictionaries (from 'load_metadata')
        :param total_frames: Total number of 100ms frames in source audio
        :param n: Maximum number of repetitions
        :param c: Number of classes
        :return: N x 3 x C x Total Frames Numpy Ndarray
        """
        multi_accdoa = np.zeros((n, 3, c, total_frames))
        event_count_per_frame = np.zeros((c, total_frames), dtype=np.int)
        for metadata_i in metadata:
            f, a, s, az, el = (metadata_i["frame_number"], metadata_i["active_class"], metadata_i["source_number"],
                               metadata_i["azimuth"], metadata_i["elevation"])
            f -= 1
            norm_az_el = np.array([np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), np.sin(np.deg2rad(el))])
            multi_accdoa[event_count_per_frame[a, f]:, :, a, f] = norm_az_el
            event_count_per_frame[a, f] += 1
        return multi_accdoa

    @staticmethod
    def chunk_seldnet_feature(feature, feat_size=250):
      
      s0,s1,s2 = feature.shape
      # print(feature.shape)
      news2 = int(np.ceil(s2/feat_size)*feat_size)
      # print("padded length  ", news2)
      feature = np.pad(feature, ((0,0), (0,0), (0,news2-s2)))
      # print(feature.shape, "  new feature shape")
      feature = np.reshape(feature, (7,news2,64))
      return np.split(feature, news2/feat_size, axis=1 )
      # return feature

    @staticmethod
    def chunk_seldnet_multiaccdoa(multi_accdoa,feat_size, hop_length):
      split_size = feat_size//(100//hop_length)
      # print(multi_accdoa.shape, "  multi accdoa shape")
      # print(split_size, " split size")
      split_count = multi_accdoa.shape[-1]/split_size
      toPad = int(np.ceil(split_count)*split_size) - multi_accdoa.shape[-1]

      multi_accdoa = np.pad(multi_accdoa, ((0,0), (0,0),(0,0), (0,toPad)))
      # print(multi_accdoa.shape, "  multi accdoa shape")
      split_count = multi_accdoa.shape[-1]/split_size
      # print(split_count)


      return np.split(multi_accdoa, split_count, axis=-1)


    def __len__(self):
        if self.model=="seldnet":
            return self.features.shape[0]
        return self.multi_accdoa.shape[-1]

    def __getitem__(self, item):
        if self.model =="seldnet":
            return torch.from_numpy(self.features[item]), torch.from_numpy(self.multi_accdoa[item])
        return self.features[:, :, item*self.feature_width:(item+1)*self.feature_width+self.context*2], \
               self.multi_accdoa[:, :, :, item]
