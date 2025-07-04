import abc
import copy
import skopt

import network


class AbstractModel(abc.ABC):

    def __init__(self, config, base):
        self._config                   = config
        self._base                     = base
        self._current_best_model       = None
        self._current_best_model_loss  = float("inf")
        self._current_best_model_epoch = 0


    @abc.abstractmethod
    def train(self, train_data, val_data):
        """Train the model."""


    @abc.abstractmethod
    def test(self, data):
        """Test the model."""


    def _create_optimized_trained_model(self, train_data, val_data):
        hyperparameter_search_space = [
            skopt.space.Integer(
                low  = self._config["network_layers_bounds"][0],
                high = self._config["network_layers_bounds"][1],
                name = "layers"),
            skopt.space.Integer(
                low  = self._config["network_neurons_bounds"][0],
                high = self._config["network_neurons_bounds"][1],
                name = "neurons"),
            skopt.space.Real(
                low   = self._config["network_learning_rate_bounds"][0],
                high  = self._config["network_learning_rate_bounds"][1],
                prior = "log-uniform",
                name  = "learning_rate"),
            skopt.space.Real(
                low   = self._config["network_momentum_bounds"][0],
                high  = self._config["network_momentum_bounds"][1],
                prior = "uniform",
                name  = "momentum"),
        ]

        self._current_best_model        = None
        self._current_best_model_loss   = float("inf")
        self._current_best_model_epoch = 0

        @skopt.utils.use_named_args(hyperparameter_search_space)
        def objective(**params):
            hyperparameters = {
                "layers":        int(params["layers"]),
                "neurons":       int(params["neurons"]),
                "learning_rate": params["learning_rate"],
                "momentum":      params["momentum"]
            }

            model = network.Network(self._config, hyperparameters, self._base)
            model.set_up()
            train_results = model.train_net(train_data, val_data)
            val_loss = train_results["val_loss"]
            epoch    = train_results["epoch"]

            if val_loss < self._current_best_model_loss:
                self._current_best_model       = copy.deepcopy(model)
                self._current_best_model_loss  = val_loss
                self._current_best_model_epoch = epoch

            return val_loss

        skopt.gp_minimize(
            func         = objective,
            dimensions   = hyperparameter_search_space,
            n_calls      = self._config["hyperperameter_sweeps"],
            random_state = 42
        )

        return self._current_best_model, self._current_best_model_epoch
