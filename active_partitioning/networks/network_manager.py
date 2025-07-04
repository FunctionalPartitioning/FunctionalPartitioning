import random
import pandas

from networks import network


class NetworkManager():

    def __init__(self, config, base):
        self._config   = config
        self._base     = base
        self._nets     = {}
        self._drop_log = pandas.DataFrame()
        self._add_log  = pandas.DataFrame()


    def set_up(self):
        for _ in range(self._config["algorithm_initial_nets"]):
            net = self._create_net()
            self._add_net(net)


    def predict(self):
        predictions = {}
        data        = self._base.get_all_data()
        for net_id, net in self._nets.items():
            predictions[net_id] = net.predict(data)
        self._base.submit_predictions(predictions)


    def train(self):
        for net_id, net in self._nets.items():
            data = self._base.get_mapped_data(net_id)
            net.train_for_one_epoch(data)


    def drop(self, epoch):
        if not self._config["algorithm_dropping_active"]:
            return
        if epoch < self._config["algorithm_dropping_adding_start"]:
            return
        if epoch % self._config["algorithm_dropping_interval"] != 0:
            return

        for net_id in list(self._nets.keys()):
            net_replacability = self._base.get_replacability(net_id)
            net_drop          = net_replacability < self._config["algorithm_dropping_replacability"]
            if net_drop:
                self._drop_net(net_id)
            self._log_drop(epoch, net_id, net_replacability, net_drop)


    def add(self, epoch):
        if not self._config["algorithm_adding_active"]:
            return
        if epoch < self._config["algorithm_dropping_adding_start"]:
            return
        if epoch % self._config["algorithm_adding_interval"] != 0:
            return

        std_devs = self._config["algorithm_adding_poor_prediction_devs"]
        (
            sufficient_data,
            train_data,
            val_data,
            old_train_loss
        ) = self._base.get_poorly_predicted_data(std_devs)

        if not sufficient_data:
            return

        net                          = self._create_net()
        new_val_loss, epochs_trained = net.train_after_addition(train_data, val_data, epoch+1)

        net_add = new_val_loss < old_train_loss
        if net_add:
            self._add_net(net)

        self._log_add(epoch, old_train_loss, new_val_loss, epochs_trained, net_add)


    def _create_net(self):
        hyperparameters = {
            "layers": random.randint(
                a = self._config["network_layers_bounds"][0],
                b = self._config["network_layers_bounds"][1]),
            "neurons": random.randint(
                a = self._config["network_neurons_bounds"][0],
                b = self._config["network_neurons_bounds"][1]),
            "learning_rate": random.uniform(
                a = self._config["network_learning_rate_bounds"][0],
                b = self._config["network_learning_rate_bounds"][1]),
            "momentum": random.uniform(
                a = self._config["network_momentum_bounds"][0],
                b = self._config["network_momentum_bounds"][1]),
        }
        net = network.Network(self._config, hyperparameters, self._base)
        net.set_up()
        return net


    def _drop_net(self, net_id):
        if len(self._nets) > 1:
            self._base.drop(net_id)
            del self._nets[net_id]


    def _add_net(self, net):
        net_id = max(self._nets.keys(), default=-1) + 1
        self._nets[net_id] = net


    def _log_drop(self, epoch, net_id, net_replacability, net_drop):
        new_row = pandas.DataFrame([{
            "epoch":             epoch,
            "net_id":            net_id,
            "net_replacability": net_replacability,
            "net_drop":          net_drop
        }])
        if self._drop_log.empty:
            self._drop_log = new_row
        else:
            self._drop_log = pandas.concat([self._drop_log, new_row], ignore_index=True)


    def _log_add(self, epoch, old_train_loss, new_val_loss, epochs_trained, net_add):
        new_row = pandas.DataFrame([{
            "epoch":          epoch,
            "old_train_loss": old_train_loss,
            "new_val_loss":   new_val_loss,
            "epochs_trained": epochs_trained,
            "net_add":        net_add
        }])
        if self._add_log.empty:
            self._add_log = new_row
        else:
            self._add_log = pandas.concat([self._add_log, new_row], ignore_index=True)


    def export_drop_log(self):
        return self._drop_log


    def export_add_log(self):
        return self._add_log


    def export_hyperparameters(self):
        hyperparameters = {}
        for net_id, net in self._nets.items():
            hyperparameters[net_id] = net.get_hyperparameters()
        return hyperparameters
