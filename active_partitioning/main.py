import os
import torch

import algorithm
import data_exporter
import data_classifier
from database import data_base
from networks import network_manager


def main():
    config = _set_config()
    config = _select_device(config)

    base = data_base.DataBase(config)
    base.set_up()

    manager = network_manager.NetworkManager(config, base)
    manager.set_up()

    classifier = data_classifier.DataClassifier(config, base)
    classifier.set_up()

    exporter = data_exporter.DataExporter(config, base, manager, classifier)
    exporter.set_up()

    algorithm.run(config, manager, exporter)

    classifier.fit()

    exporter.export()


def _set_config():
    return {
        "device":                                     "gpu",
        "input_dir":                                  os.path.join("..", "input"),
        "output_dir":                                 os.path.join("..", "output"),
        "relative_paths":                             True,
        "input_data_file":                            "scaled_training_data.csv",
        "scaling_file":                               "scaler_labels.pickle",
        "scaling_compute_test_loss_rescaled":         True,
        "output_config_file":                         "config_partitioning.txt",
        "output_hyperparameters_file":                "hyperparameters.txt",
        "output_classifier_file":                     "classifier.pickle",
        "output_classifier_fit_file":                 "classifier_fit.txt",
        "output_drop_log_file":                       "drop_log.csv",
        "output_add_log_file":                        "add_log.csv",
        "data_points":                                50,
        "training_batch_size":                        8,
        "test_batch_size":                            128,
        "network_layers_bounds":                      [2, 6],
        "network_neurons_bounds":                     [4, 10],
        "network_learning_rate_bounds":               [0.0001, 0.005],
        "network_momentum_bounds":                    [0.4, 0.6],
        "network_activation":                         "tanh",
        "network_loss":                               "mse",
        "network_optimizer":                          "adam",
        "network_optimization_mode":                  "min",
        "network_learning_rate_decay_factor":         0.5,
        "network_learning_rate_decay_patience":       10,
        "algorithm_epochs":                           5,
        "algorithm_initial_nets":                     10,
        "algorithm_dropping_adding_start":            0,
        "algorithm_dropping_active":                  True,
        "algorithm_dropping_interval":                1,
        "algorithm_dropping_replacability":           1.8,
        "algorithm_adding_active":                    True,
        "algorithm_adding_interval":                  1,
        "algorithm_adding_catch_up_train_min_epochs": 30,
        "algorithm_adding_catch_up_train_patience":   10,
        "algorithm_adding_poor_prediction_devs":      1,
        "algorithm_adding_min_poor_predictions":      5,
    }

def _select_device(config):
    if   config["device"] == "gpu":
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif config["device"] == "cpu":
        config["device"] = torch.device("cpu")
    else:
        raise ValueError("No valid device option selected!")
    return config


if __name__ == '__main__':
    main()
    