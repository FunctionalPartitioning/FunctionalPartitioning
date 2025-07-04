import sklearn.svm


class DataClassifier():

    def __init__(self, config, base):
        self._config     = config
        self._base       = base
        self._svm        = None
        self._svm_fitted = False


    def set_up(self):
        self._svm = sklearn.svm.SVC(kernel="rbf", C=0.1, random_state=42)


    def fit(self):
        features, net_labels, unique_net_labels = self._base.get_partitioned_data()
        if unique_net_labels > 1:
            self._svm.fit(features, net_labels)
            self._svm_fitted = True


    def get_fitted_classifier(self):
        return self._svm


    def get_score(self):
        features, net_labels, _ = self._base.get_partitioned_data()
        if self._svm_fitted:
            score = self._svm.score(features, net_labels)
        else:
            score = float("nan")
        return score, self._svm_fitted
