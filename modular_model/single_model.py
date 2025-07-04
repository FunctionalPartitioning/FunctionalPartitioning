import abstract_model


class SingleModel(abstract_model.AbstractModel):

    def __init__(self, config, base):
        super().__init__(config, base)
        self._model  = None
        self._epoch  = None


    def set_up(self):
        pass


    def train(self, train_data, val_data):
        self._model, self._epoch = self._create_optimized_trained_model(
            train_data = train_data,
            val_data   =   val_data,
        )


    def test(self, data):
        data_           = data.reset_index(drop=True)
        results         = self._model.test_net(data=data_)
        results["data"] =                           data_
        return {0: results}


    def export_hyperparameters(self):
        hyperparameters = self._model.get_hyperparameters()
        return {0: hyperparameters}
