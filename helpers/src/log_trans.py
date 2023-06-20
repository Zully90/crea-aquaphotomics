import numpy as np
from sklearn.base import (BaseEstimator, MultiOutputMixin, RegressorMixin,
                          TransformerMixin)


class LogTrans(BaseEstimator, TransformerMixin, RegressorMixin, MultiOutputMixin):
    def __init__(self):
        pass

    def fit(self, X, Y=None):
        X = np.log10(X + 1)
        return self

    def transform(self, X, Y=None):
        X = np.log10(X + 1)
        return X

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X, Y)

    def predict(self, X=None):
        return np.log10(X + 1)
