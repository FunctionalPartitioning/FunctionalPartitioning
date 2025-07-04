import os
import json
import copy
import pickle
import shutil
import datetime


class DataExporter():

    def __init__(self, config, base, manager, classifier):
        self._config             = config
        self._base               = base
        self._manager            = manager
        self._classifier         = classifier
        self._full_output_dir    = None
        self._process_output_dir = None


    def set_up(self):
        current_dir = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        if self._config["relative_paths"]:
            cwd = os.getcwd()
            self._full_output_dir = os.path.join(cwd, self._config["output_dir"], current_dir)
        else:
            self._full_output_dir = os.path.join(     self._config["output_dir"], current_dir)

        self._process_output_dir = os.path.join(self._full_output_dir, "process")

        os.mkdir(self._full_output_dir)
        os.mkdir(self._process_output_dir)


    def export_data(self, epoch):
        data = self._base.export_data()
        path = os.path.join(self._process_output_dir, f"epoch_{epoch}.csv")
        data.to_csv(path_or_buf=path, index=False)


    def export(self):
        self.export_data(self._config["algorithm_epochs"])
        self._copy_input_dir()
        self._export_config()
        self.export_hyperparameters()
        self.export_process_parameters()
        self.export_classifier()


    def _copy_input_dir(self):
        src_path = self._config["input_dir"]
        dst_path = self._full_output_dir
        for path in os.listdir(src_path):
            path = os.path.join(src_path, path)
            if os.path.isfile(path):
                shutil.copy(src = path, dst = dst_path)


    def _export_config(self):
        config = copy.deepcopy(self._config)
        config = {key: str(value) for key, value in config.items()}

        path = os.path.join(self._full_output_dir, self._config["output_config_file"])
        with open(path, mode="w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)


    def export_hyperparameters(self):
        hyperparameters = self._manager.export_hyperparameters()

        path = os.path.join(self._full_output_dir, self._config["output_hyperparameters_file"])
        with open(path, mode="w", encoding="utf-8") as file:
            json.dump(hyperparameters, file, indent=4)


    def export_process_parameters(self):
        classifier_score, classifier_fitted = self._classifier.get_score()
        drop_log                            = self._manager.export_drop_log()
        add_log                             = self._manager.export_add_log()

        path = os.path.join(self._full_output_dir, self._config["output_classifier_fit_file"])
        with open(path, mode="w", encoding="utf-8") as file:
            file.write(f"Classifier fitted: {classifier_fitted}\n")
            file.write(f"Classifier score: { classifier_score}")

        path = os.path.join(self._full_output_dir, self._config["output_drop_log_file"])
        drop_log.to_csv(path_or_buf=path, index=False)
        path = os.path.join(self._full_output_dir, self._config["output_add_log_file"])
        add_log.to_csv(path_or_buf=path, index=False)


    def export_classifier(self):
        classifier = self._classifier.get_fitted_classifier()
        path = os.path.join(self._full_output_dir, self._config["output_classifier_file"])
        with open(path,"wb") as file:
            pickle.dump(classifier, file, protocol=pickle.HIGHEST_PROTOCOL)
