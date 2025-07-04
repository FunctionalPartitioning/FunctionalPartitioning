import copy
import torch
import numpy


class Network(torch.nn.Module):

    def __init__(self, config, hyperparameters, base):
        super().__init__()
        self._config            = config
        self._hyperparameters   = hyperparameters
        self._base              = base
        self._layer_stack       = None
        self._optimization_loss = None
        self._ranking_loss      = None
        self._optimizer         = None
        self._scheduler         = None


    def set_up(self):
        self._stack_layers()
        self._set_optimization_loss()
        self._set_ranking_loss()
        self._set_optimizer()
        self._set_learning_rate_scheduler()


    def forward(self, x):
        logits = self._layer_stack(x)
        return logits


    def predict(self, data):
        data = torch.utils.data.DataLoader(
            dataset    = data,
            batch_size = self._config["test_batch_size"]
        )

        # Set the model to test mode
        self._layer_stack.eval()

        predictions = []
        losses      = []
        with torch.no_grad():

            for batch_x, batch_y in data:
                batch_x = batch_x.to(self._config["device"])
                batch_y = batch_y.to(self._config["device"])

                prediction_y = self._layer_stack(batch_x)
                loss         = self._ranking_loss(prediction_y, batch_y)

                predictions.append(prediction_y.cpu().numpy())
                losses.append(loss.cpu().numpy())

        predictions = numpy.concatenate(predictions, axis=0)
        losses      = numpy.concatenate(losses,      axis=0)
        losses      = numpy.mean(       losses,      axis=1)
        return {"predictions": predictions, "losses": losses}


    def train_after_addition(self, train_data, val_data, max_epochs):
        best_model_state = None
        best_val_loss    = float("inf")
        best_epoch       = 0

        for epoch in range(max_epochs):
            self.train_for_one_epoch(train_data)
            val_loss   = self.test(val_data)
            self._scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_model_state = copy.deepcopy(self.state_dict())
                best_val_loss    = val_loss
                best_epoch       = epoch

            epochs_since_improve = epoch - best_epoch
            if (epochs_since_improve > self._config["algorithm_adding_catch_up_train_patience"] and
                epoch                > self._config["algorithm_adding_catch_up_train_min_epochs"]):
                break

        self.load_state_dict(best_model_state)
        return best_val_loss, epoch


    def train_for_one_epoch(self, data):
        data = torch.utils.data.DataLoader(
            dataset    = data,
            batch_size = self._config["training_batch_size"]
        )

        # Set to training mode
        self._layer_stack.train()

        for batch_x, batch_y in data:
            batch_x = batch_x.to(self._config["device"])
            batch_y = batch_y.to(self._config["device"])

            # Reset all gradients
            self._optimizer.zero_grad()

            # Forward-, backward-propagation and optimization
            prediction_y  = self._layer_stack(batch_x)
            training_loss = self._optimization_loss(prediction_y, batch_y)
            training_loss.backward()
            self._optimizer.step()


    def test(self, data):
        data = torch.utils.data.DataLoader(
            dataset    = data,
            batch_size = self._config["test_batch_size"]
        )

        # Set to test mode
        self._layer_stack.eval()

        loss = 0
        with torch.no_grad():

            for batch_x, batch_y in data:
                batch_x = batch_x.to(self._config["device"])
                batch_y = batch_y.to(self._config["device"])

                prediction_y = self._layer_stack(batch_x)

                if self._config["scaling_compute_test_loss_rescaled"]:
                    batch_y      = self._base.rescale_labels(batch_y)
                    prediction_y = self._base.rescale_labels(prediction_y)

                loss += self._optimization_loss(prediction_y, batch_y).item()

        return loss / len(data)


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


    def _set_optimization_loss(self):
        if   self._config["network_loss"] == "mse":
            self._optimization_loss = torch.nn.MSELoss()
        elif self._config["network_loss"] == "mae":
            self._optimization_loss = torch.nn.L1Loss()
        else:
            raise ValueError(f"The set loss function {self._config['network_loss']} is unvalid!")


    def _set_ranking_loss(self):
        if   self._config["network_loss"] == "mse":
            self._ranking_loss = torch.nn.MSELoss(reduction="none")
        elif self._config["network_loss"] == "mae":
            self._ranking_loss = torch.nn.L1Loss(reduction="none")
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
