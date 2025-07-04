import os
import pickle
import torch
import pandas


class DataBase():

    def __init__(self, config):
        self._config       = config
        self._train_data   = None
        self._val_data     = None
        self._test_data    = None
        self._label_scaler = None


    def set_up(self):
        self._train_data = self._load_data(
            file        = self._config["input_train_data_file"],
            data_points = self._config["train_data_points"]
        )
        self._val_data   = self._load_data(
            file        = self._config["input_val_data_file"],
            data_points = self._config["val_data_points"]
        )
        self._test_data  = self._load_data(
            file        = self._config["input_test_data_file"],
            data_points = self._config["test_data_points"]
        )
        self._label_scaler = self._load_scaler()


    def _load_data(self, file, data_points):
        path = os.path.join(self._config["partitioning_dir"], file)
        data = pandas.read_csv(
            filepath_or_buffer = path,
            delimiter          = ",",
            header             = 0,
            index_col          = False,
            dtype              = "float32",
        )
        if data_points is not None:
            data = data.sample(
                n            = data_points,
                replace      = False,
                random_state = 42
            )
            data = data.reset_index(drop=True)
        return data


    def _load_scaler(self):
        path = os.path.join(self._config["partitioning_dir"], self._config["scaling_file"])
        with open(path, "rb") as file:
            return pickle.load(file)


    def get_features(self):
        return len(self._train_data.filter(like="x_").columns)


    def get_labels(self):
        return len(self._train_data.filter(like="y_").columns)


    def get_train_data(self):
        return self._train_data


    def get_val_data(self):
        return self._val_data


    def get_test_data(self):
        return self._test_data


    def rescale_labels(self, labels):
        return torch.from_numpy(self._label_scaler.inverse_transform(labels.cpu().numpy()))
