import os
import ast
import json
import torch

import data_base
import single_model
import modular_model
import data_exporter


def main():
    config = _set_config()
    config = _select_device(           config)
    config = _set_partitioning_dir(    config)
    config = _read_partitioning_config(config)

    exporter = data_exporter.DataExporter(config)
    exporter.set_up()

    base = data_base.DataBase(config)
    base.set_up()

    sm = single_model.SingleModel(config, base)
    sm.set_up()
    sm.train(train_data=base.get_train_data(), val_data=base.get_val_data())

    mm = modular_model.ModularModel(config, base)
    mm.set_up()
    mm.train(train_data=base.get_train_data(), val_data=base.get_val_data())

    exporter.export(base=base, sm=sm, mm=mm)


def _set_config():
    return {
        "device":                               "gpu",
        "partitioning_dir":                     os.path.join("..", "..", "active_partitioning", "output"),
        "specific_partitioning_dir":            False,
        "input_config_file":                    "config_partitioning.txt",
        "input_classifier_file":                "classifier.pickle",
        "input_classifier_fit_file":            "classifier_fit.txt",
        "input_train_data_file":                "scaled_training_data.csv",
        "input_val_data_file":                  "scaled_validation_data.csv",
        "input_test_data_file":                 "scaled_test_data.csv",
        "scaling_file":                         "scaler_labels.pickle",
        "scaling_compute_test_loss_rescaled":   True,
        "output_train_data_file":               "scaled_training_data.csv",
        "output_val_data_file":                 "scaled_validation_data.csv",
        "output_test_data_file":                "scaled_test_data.csv",
        "output_train_loss_file":               "train_loss.txt",
        "output_val_loss_file":                 "val_loss.txt",
        "output_test_loss_file":                "test_loss.txt",
        "output_config_file":                   "config_modular_model.txt",
        "output_hyperparameters_file":          "hyperparameters.txt",
        "train_epochs":                         5,
        "train_early_stop_patience":            5,
        "train_early_stop_min_epochs":          5,
        "hyperperameter_sweeps":                10,
        "train_data_points":                    50,
        "val_data_points":                      10,
        "test_data_points":                     10,
        "train_batch_size":                     None, # Loaded from partitioning config
        "test_batch_size":                      None, # Loaded from partitioning config
        "network_layers_bounds":                None, # Loaded from partitioning config
        "network_neurons_bounds":               None, # Loaded from partitioning config
        "network_learning_rate_bounds":         None, # Loaded from partitioning config
        "network_momentum_bounds":              None, # Loaded from partitioning config
        "network_activation":                   None, # Loaded from partitioning config
        "network_loss":                         None, # Loaded from partitioning config
        "network_optimizer":                    None, # Loaded from partitioning config
        "network_optimization_mode":            None, # Loaded from partitioning config
        "network_learning_rate_decay_factor":   None, # Loaded from partitioning config
        "network_learning_rate_decay_patience": None, # Loaded from partitioning config
    }


def _select_device(config):
    match config["device"]:
        case "gpu":
            config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        case "cpu":
            config["device"] = torch.device("cpu")
        case _:
            raise ValueError("No valid device option selected!")
    return config


def _set_partitioning_dir(config):
    if not config["specific_partitioning_dir"]:
        part_dir_top = config["partitioning_dir"]
        entries      = os.listdir(config["partitioning_dir"])
        entries      = [entry for entry in entries if not entry.endswith("_modular_model")]
        part_dirs    = [e for e in entries if os.path.isdir(os.path.join(part_dir_top, e))]
        config["partitioning_dir"] = os.path.join(part_dir_top, sorted(part_dirs)[-1])
    return config


def _read_partitioning_config(config):
    path = os.path.join(config["partitioning_dir"], config["input_config_file"])
    with open(path, mode="r", encoding="utf-8") as file:
        input_config = json.load(file)

    config["train_batch_size"]                     = int(input_config["training_batch_size"])
    config["test_batch_size"]                      = int(input_config["test_batch_size"])
    config["network_layers_bounds"]                = ast.literal_eval(input_config["network_layers_bounds"])
    config["network_neurons_bounds"]               = ast.literal_eval(input_config["network_neurons_bounds"])
    config["network_learning_rate_bounds"]         = ast.literal_eval(input_config["network_learning_rate_bounds"])
    config["network_momentum_bounds"]              = ast.literal_eval(input_config["network_momentum_bounds"])
    config["network_activation"]                   = input_config["network_activation"]
    config["network_loss"]                         = input_config["network_loss"]
    config["network_optimizer"]                    = input_config["network_optimizer"]
    config["network_optimization_mode"]            = input_config["network_optimization_mode"]
    config["network_learning_rate_decay_factor"]   = float(input_config["network_learning_rate_decay_factor"])
    config["network_learning_rate_decay_patience"] = int(input_config["network_learning_rate_decay_patience"])

    return config


if __name__ == "__main__":
    main()
