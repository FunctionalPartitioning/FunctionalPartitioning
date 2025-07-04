import os
import re
import pickle
import numpy

import abstract_model


class ModularModel(abstract_model.AbstractModel):

    def __init__(self, config, base):
        super().__init__(config, base)
        self._models            = {}
        self._epochs            = {}
        self._classifier        = None
        self._classifier_fitted = False


    def set_up(self):
        file = self._config["input_classifier_fit_file"]
        path  = os.path.join(self._config["partitioning_dir"], file)
        with open(path, mode="r", encoding="utf-8") as file:
            fitted = re.search(r"Classifier fitted:\s*(True|False)", file.read()).group(1)

        if fitted == "True":
            self._classifier_fitted = True

            file = self._config["input_classifier_file"]
            path = os.path.join(self._config["partitioning_dir"], file)
            with open(path, "rb") as file:
                self._classifier = pickle.load(file)


    def train(self, train_data, val_data):
        if self._classifier_fitted:

            for idx, class_ in enumerate(self._classifier.classes_):
                train_data_classes = self._classifier.predict(train_data.filter(like="x_"))
                val_data_classes   = self._classifier.predict(  val_data.filter(like="x_"))

                if (numpy.any([train_data_classes==class_]) and
                    numpy.any([  val_data_classes==class_])):

                    self._models[idx], self._epochs[idx] = self._create_optimized_trained_model(
                        train_data = train_data[train_data_classes==class_],
                        val_data   =   val_data[  val_data_classes==class_],
                    )


    def test(self, data):
        results = {}

        if self._classifier_fitted:

            for idx, class_ in enumerate(self._classifier.classes_):
                data_classes = self._classifier.predict(data.filter(like="x_"))

                if idx in self._models and numpy.any([data_classes==class_]):
                    data_        = data[data_classes==class_].reset_index(drop=True)
                    results[idx] = self._models[idx].test_net(
                        data             = data_
                    )
                    results[idx]["data"] = data_
                else:
                    results[idx] = {
                        "predictions": numpy.array([]),
                        "loss":        numpy.nan,
                        "data":        data[data_classes==class_]
                    }

        return results


    def export_hyperparameters(self):
        hyperparameters = {}

        if self._classifier_fitted:

            for idx, _ in enumerate(self._classifier.classes_):
                if idx in self._models:
                    hyperparameters[idx] = self._models[idx].get_hyperparameters()
                else:
                    hyperparameters[idx] = None

        return hyperparameters
