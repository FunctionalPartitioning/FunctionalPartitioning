import copy
import torch
import numpy

import data_set


class Network(torch.nn.Module):

    def __init__(self, config, hyperparameters, base):
        super().__init__()
        self._config          = config
        self._hyperparameters = hyperparameters
        self._base            = base
        self._layer_stack     = None
        self._loss            = None
        self._optimizer       = None
        self._scheduler       = None


    def set_up(self):
        self._stack_layers()
        self._set_loss()
        self._set_optimizer()
        self._set_learning_rate_scheduler()


    def forward(self, x):
        logits = self._layer_stack(x)
        return logits


    def train_net(self, train_data, val_data):
        train_data = torch.utils.data.DataLoader(
            dataset    = data_set.DataSet(train_data),
            batch_size = self._config["train_batch_size"]
        )

        best_model_state = None
        best_val_loss    = float("inf")
        best_epoch       = 0

        for epoch in range(self._config["train_epochs"]):

            # Set to training mode
            self._layer_stack.train()

            for batch_x, batch_y in train_data:
                batch_x = batch_x.to(self._config["device"])
                batch_y = batch_y.to(self._config["device"])

                # Reset all gradients
                self._optimizer.zero_grad()

                # Forward-, backward-propagation and optimization
                prediction_y = self._layer_stack(batch_x)
                train_loss   = self._loss(prediction_y, batch_y)
                train_loss.backward()
                self._optimizer.step()

            val_loss = self.test_net(val_data)["loss"]
            self._scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_model_state = copy.deepcopy(self.state_dict())
                best_val_loss    = val_loss
                best_epoch       = epoch

            epochs_since_improve = epoch - best_epoch
            if (epochs_since_improve > self._config["train_early_stop_patience"] and
                epoch                > self._config["train_early_stop_min_epochs"]):
                break

        self.load_state_dict(best_model_state)
        return {"epoch": epoch, "val_loss": best_val_loss}


    def test_net(self, data):
        data = torch.utils.data.DataLoader(
            dataset    = data_set.DataSet(data),
            batch_size = self._config["test_batch_size"]
        )

        # Set to test mode
        self._layer_stack.eval()

        predictions = []
        loss        = 0
        with torch.no_grad():

            for batch_x, batch_y in data:
                batch_x = batch_x.to(self._config["device"])
                batch_y = batch_y.to(self._config["device"])

                prediction_y = self._layer_stack(batch_x)

                predictions += [prediction_y.cpu().numpy()]

                if self._config["scaling_compute_test_loss_rescaled"]:
                    batch_y      = self._base.rescale_labels(batch_y)
                    prediction_y = self._base.rescale_labels(prediction_y)

                loss        += self._loss(prediction_y, batch_y).item()

        return {"predictions": numpy.concatenate(predictions, axis=0), "loss": loss / len(data)}


    def get_hyperparameters(self):
        return self._hyperparameters


    def _stack_layers(self):
        features = self._base.get_features()
        labels   = self._base.get_labels()
        layers   = self._hyperparameters["layers"]
        neurons  = self._hyperparameters["neurons"]

        self._layer_stack = torch.nn.Sequential()

        if layers == 1:
            self._layer_stack.append(torch.nn.Linear(features, labels))

        else: # parameters["layers"] >= 2
            # First layer
            self._layer_stack.append(torch.nn.Linear(features, neurons))
            self._append_activation_to_layer_stack()

            # Layers in between
            for _ in range(layers-2):
                self._layer_stack.append(torch.nn.Linear(neurons, neurons))
                self._append_activation_to_layer_stack()

            # Last layer
            self._layer_stack.append(torch.nn.Linear(neurons, labels))

        self._layer_stack.to(self._config["device"])


    def _append_activation_to_layer_stack(self):
        if   self._config["network_activation"] == "relu":
            self._layer_stack.append(torch.nn.ReLU())
        elif self._config["network_activation"] == "leaky_relu":
            self._layer_stack.append(torch.nn.LeakyReLU())
        elif self._config["network_activation"] == "tanh":
            self._layer_stack.append(torch.nn.Tanh())
        elif self._config["network_activation"] == "sigmoid":
            self._layer_stack.append(torch.nn.Sigmoid())
        else:
            raise ValueError(
                f"The set activation function {self._config['network_activation']} is unvalid!"
            )


    def _set_loss(self):
        if   self._config["network_loss"] == "mse":
            self._loss = torch.nn.MSELoss()
        elif self._config["network_loss"] == "mae":
            self._loss = torch.nn.L1Loss()
        else:
            raise ValueError(f"The set loss function {self._config['network_loss']} is unvalid!")


    def _set_optimizer(self):
        if   self._config["network_optimizer"] == "adam":
            self._optimizer = torch.optim.Adam(
                params   = self._layer_stack.parameters(),
                lr       = self._hyperparameters["learning_rate"]
            )
        elif self._config["network_optimizer"] == "adagrad":
            self._optimizer = torch.optim.Adagrad(
                params   = self._layer_stack.parameters(),
                lr       = self._hyperparameters["learning_rate"]
            )
        elif self._config["network_optimizer"] == "rmsprop":
            self._optimizer = torch.optim.RMSprop(
                params   = self._layer_stack.parameters(),
                lr       = self._hyperparameters["learning_rate"],
                momentum = self._hyperparameters["momentum"]
            )
        elif self._config["network_optimizer"] == "sgd":
            self._optimizer = torch.optim.SGD(
                params   = self._layer_stack.parameters(),
                lr       = self._hyperparameters["learning_rate"],
                momentum = self._hyperparameters["momentum"]
            )
        else:
            raise ValueError(f"The set optimizer {self._config['network_optimizer']} is unvalid!")


    def _set_learning_rate_scheduler(self):
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = self._optimizer,
            mode      = self._config["network_optimization_mode"],
            factor    = self._config["network_learning_rate_decay_factor"],
            patience  = self._config["network_learning_rate_decay_patience"],
        )
