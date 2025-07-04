import os
import re
import pickle
import torch
import pandas
import sklearn.model_selection

from database import data_set


class DataBase():

    def __init__(self, config):
        self._config = config
        self._data   = None
        self._scaler = None


    def set_up(self):
        self._data   = self._load_data()
        self._scaler = self._load_scaler()


    def _load_data(self):
        path = os.path.join(self._config["input_dir"], self._config["input_data_file"])
        data = pandas.read_csv(
            filepath_or_buffer = path,
            delimiter          = ",",
            header             = 0,
            index_col          = False,
            dtype              = "float32",
        )
        data = data.sample(
            n            = self._config["data_points"],
            replace      = False,
            random_state = 42
        )
        data = data.reset_index(drop=True)
        return data


    def _load_scaler(self):
        path = os.path.join(self._config["input_dir"], self._config["scaling_file"])
        with open(path, "rb") as file:
            return pickle.load(file)


    def submit_predictions(self, predictions):
        for net_id, net_predictions in predictions.items():
            for lb in range(self.get_labels()):
                self._data[f"net_{net_id}_prediction_{lb}"] = net_predictions["predictions"][:,lb]
            self._data[    f"net_{net_id}_loss"]            = net_predictions["losses"]
        self._detect_best_predictions()


    def _detect_best_predictions(self):
        loss_columns = [col for col in self._data.columns if re.match(r"net_\d+_loss", col)]
        self._data["minimal_loss"]     = self._data[loss_columns].min(axis=1)
        self._data["minimal_loss_net"] = self._data[loss_columns].idxmin(axis=1)


    def drop(self, net_id):
        net_prediction_cls = [f"net_{net_id}_prediction_{lb}" for lb in range(self.get_labels())]
        self._data = self._data.drop(columns = net_prediction_cls + [f"net_{net_id}_loss"])
        self._detect_best_predictions()


    def rescale_labels(self, labels):
        return torch.from_numpy(self._scaler.inverse_transform(labels.cpu().numpy()))


    def get_features(self):
        return len(self._data.filter(like="x_").columns)


    def get_labels(self):
        return len(self._data.filter(like="y_").columns)


    def get_all_data(self):
        features = self._data.filter(like="x_")
        labels   = self._data.filter(like="y_")
        return data_set.DataSet(pandas.concat([features, labels], axis=1))


    def get_mapped_data(self, net_id):
        mapped_data = self._data[self._data["minimal_loss_net"] == f"net_{net_id}_loss"]
        features    = mapped_data.filter(like="x_")
        labels      = mapped_data.filter(like="y_")
        return data_set.DataSet(pandas.concat([features, labels], axis=1))


    def get_poorly_predicted_data(self, std_devs):
        loss_mean            = self._data["minimal_loss"].mean()
        loss_std             = self._data["minimal_loss"].std()
        loss_threshold       = loss_mean + std_devs * loss_std
        poor_data            = self._data[self._data["minimal_loss"] > loss_threshold]
        poor_data_len        = len(poor_data)
        min_poor_predictions = self._config["algorithm_adding_min_poor_predictions"]
        val_share            = 0.2
        sufficient_data = poor_data_len >= min_poor_predictions and poor_data_len * val_share >= 1

        if sufficient_data:
            poor_loss      = poor_data["minimal_loss"].mean()
            features       = poor_data.filter(like="x_")
            labels         = poor_data.filter(like="y_")
            poor_data      = pandas.concat([features, labels], axis=1)
            poor_train_data, poor_val_data = sklearn.model_selection.train_test_split(
                poor_data,
                test_size    = val_share,
                random_state = 42
            )
            poor_train_data = data_set.DataSet(poor_train_data)
            poor_val_data   = data_set.DataSet(poor_val_data)

            return sufficient_data, poor_train_data, poor_val_data, poor_loss

        return sufficient_data, None, None, None


    def get_replacability(self, net_id):
        loss_columns = [col for col in self._data.columns if re.match(r"net_\d+_loss", col)]

        # In case there is only one net we consider it unreplacable
        if len(loss_columns) == 1:
            return float("inf")

        loss_columns_without_net = loss_columns
        loss_columns_without_net.remove(f"net_{net_id}_loss")

        loss             = self._data["minimal_loss"].mean()
        loss_without_net = self._data[loss_columns_without_net].min(axis=1).mean()

        return loss_without_net/loss


    def get_partitioned_data(self):
        features = self._data.filter(like="x_")
        labels   = self._data["minimal_loss_net"]

        # Extract the net number from the label
        labels   = labels.str.extract(r"net_(\d+)_loss").astype(int)

        # Convert labels to count from zero
        unique_labels = sorted(labels[0].unique())
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels        = labels[0].map(label_mapping)

        return features, labels, len(unique_labels)


    def export_data(self):
        return self._data
