import os
import copy
import json
import shutil
import pandas


class DataExporter():

    def __init__(self, config):
        self._config     = config
        self._output_dir = None


    def set_up(self):
        self._create_output_dir()


    def export(self, base, sm, mm):
        self._export_data(base, sm, mm)
        self._export_config()
        self._export_hyperparameters(sm, mm)


    def _create_output_dir(self):
        self._output_dir = self._config["partitioning_dir"] + "_modular_model"

        if os.path.exists(self._output_dir):
            shutil.rmtree(self._output_dir)

        os.mkdir(self._output_dir)


    def _export_data(self, base, sm, mm):
        self._export_one_model_data(
            base   = base,
            m      = sm,
            m_text = "single_model",
        )
        self._export_one_model_data(
            base   = base,
            m      = mm,
            m_text = "modular_model",
        )


    def _export_one_model_data(self, base, m, m_text):
        self._export_one_data_set(
            m         = m,
            m_text    = m_text,
            data      = base.get_train_data(),
            data_text = self._config["output_train_data_file"],
            loss_text = self._config["output_train_loss_file"],
        )
        self._export_one_data_set(
            m         = m,
            m_text    = m_text,
            data      = base.get_val_data(),
            data_text = self._config["output_val_data_file"],
            loss_text = self._config["output_val_loss_file"],
        )
        self._export_one_data_set(
            m         = m,
            m_text    = m_text,
            data      = base.get_test_data(),
            data_text = self._config["output_test_data_file"],
            loss_text = self._config["output_test_loss_file"],
        )


    def _export_one_data_set(self, m, m_text, data, data_text, loss_text):
        results = m.test(data)

        for idx, result in results.items():
            export_data = result["data"]
            if result["predictions"].size > 0:
                cols        = [f"prediction_{idx}" for idx in range(result["predictions"].shape[1])]
                predictions = pandas.DataFrame(result["predictions"], columns=cols)
                export_data = pandas.concat([export_data, predictions], axis=1)

            path = os.path.join(self._output_dir, f"{m_text}_{idx}_{data_text}")
            export_data.to_csv(path_or_buf=path, index=False)

            path = os.path.join(self._output_dir, f"{m_text}_{idx}_{loss_text}")
            with open(path, mode="w", encoding="utf-8") as file:
                file.write(f"{result['loss']:.6f}\n")


    def _export_config(self):
        config = copy.deepcopy(self._config)
        config = {key: str(value) for key, value in config.items()}

        path = os.path.join(self._output_dir, self._config["output_config_file"])
        with open(path, mode="w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)


    def _export_hyperparameters(self, sm, mm):
        self._export_one_model_hyperparameters(m=sm, m_text="single_model")
        self._export_one_model_hyperparameters(m=mm, m_text="modular_model")


    def _export_one_model_hyperparameters(self, m, m_text):
        hyperparameters = m.export_hyperparameters()

        for idx, hyperparameter in hyperparameters.items():
            path = os.path.join(
                self._output_dir, f"{m_text}_{idx}_{self._config['output_hyperparameters_file']}"
            )
            with open(path, mode="w", encoding="utf-8") as file:
                json.dump(hyperparameter, file, indent=4)
