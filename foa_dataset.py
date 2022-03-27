import os
import os.path as osp
from torch.utils.data.dataset import Dataset


class FOADataset(Dataset):
    """
    Custom PyTorch Dataset for DCASE FOA Datsets

    TODO: CURRENTLY ONLY GETS FILENAMES, DOES NO PREPROCESSING
    """

    def __init__(self, data_path, folds, train=True):
        """
        Init Function for FOADataset
        :param data_path: String path to root folder containing 'foa_dev' and 'metadata_dev'
        :param folds: List of fold integers to use in this dataset
        :param train: Bool indicating whether to use train dataset of val dataset
        """

        # Calculate Directory Names
        foa_directory = osp.join(data_path, "foa_dev", "dev-train" if train else "dev-val")
        meta_directory = osp.join(data_path, "metadata_dev", "dev-train" if train else "dev-val")
        all_foa_files = os.listdir(foa_directory)
        all_meta_files = os.listdir(meta_directory)

        # Parse File Names
        foa_file_data = [self.parse_foa_file_name(file) for file in all_foa_files]
        meta_file_data = [self.parse_foa_file_name(file) for file in all_meta_files]

        # Create Lists of All Valid File Paths in Given Folds
        self.folds = folds
        self.foa_files = [
            osp.join(foa_directory, file) for file, data in zip(all_foa_files, foa_file_data)
            if (data["fold"] in folds)
        ]
        self.meta_files = [
            osp.join(meta_directory, file) for file, data in zip(all_meta_files, meta_file_data)
            if (data["fold"] in folds)
        ]
        assert len(self.foa_files) == len(self.meta_files)

    @staticmethod
    def parse_foa_file_name(file):
        """
        Parses filenames of the following format:
        "fold[fold number]_room[room number per fold]_mix[recording number per room per split].wav"
        :param file: filename
        :return: metadata dictionary
        """

        name, extension = osp.splitext(file)
        fold_text, room_text, mix_text = name.split("_")
        fold = int(fold_text.replace("fold", ""))
        room = int(room_text.replace("room", ""))
        mix = int(mix_text.replace("mix", ""))
        return {"fold": fold, "room": room, "mix": mix}

    def __len__(self):
        return len(self.foa_files)

    def __getitem__(self, item):
        return self.foa_files[item]
