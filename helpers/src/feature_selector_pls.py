#
# QualiCtrl è un progetto di Qualitade S.r.l.
# (c) 2019 e anni successivi Qualitade S.r.l. - Ogni diritto è riservato.
# (c) 2019 and following years by Qualitade S.r.l. -  All right reserved.
#
# Autore: Marcello Vanzulli (marcello.vanzulli@qualitade.com)
# Autore: Mirko Morello (mirko.morello@it-impresa.it)
#
import numpy as np
import pandas as pd
from sklearn.base import (BaseEstimator, MultiOutputMixin, RegressorMixin,
                          TransformerMixin)
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (LeaveOneOut, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline


class FeatureSelectorPLS(
    BaseEstimator, TransformerMixin, RegressorMixin, MultiOutputMixin
):

    """Classe per selezionare le WL significative, sulla base di un numero arbitrario di componenti principali.
    Nasce per essere inserito in una pipeline di Scikit-learn.
    N.B. soffre di un FutureWarning particolarmente importante: per un uso stand alone non soffre di niente, usandolo in una pipeline di Grid Search possono capitare le combinazioni in cui n_features < n_components.
    Questo è un grosso problema per due motivi:
    1. di base è scorretto da un punto di vista formale;
    2. sklearn 1.1. raiserà errore; la speranza è che dia un alternativa (come fa ora abbassando le componenti fino al rango di X) altrimenti bisognerà gestire anche questo.
    """

    # Occhio Zio: controlla che le mediocentrature siano al loro posto.

    def __init__(self, components=20):
        self.components = components

    def fit(self, X, Y=None):

        # Define MSE array to be populated
        mse = np.zeros((self.components, X.shape[1]))

        # Regression with specified number of components, using full spectrum
        pls = PLSRegression(n_components=self.components, scale=True)
        pls.fit(X, Y)

        # Indices of sort spectra according to ascending absolute value of PLS coefficients
        sorted_indexes = np.argsort(np.abs(pls.coef_[:, 0]))

        # Sort spectra accordingly
        Xsorted = X[:, sorted_indexes]

        # Discard one wavelength at a time of the sorted spectra,
        # regress, and calculate the MSE cross-validation
        for wl in range(X.shape[1] - (self.components + 1)):
            pls_wl = PLSRegression(n_components=self.components, scale=True)

            pls_wl.fit(Xsorted[:, wl:], Y)

            Y_cv = cross_val_predict(pls_wl, Xsorted[:, wl:], Y, cv=5)

            mse[self.components - 1, wl] = mean_squared_error(Y, Y_cv)

        # Calculate and print the position of minimum in MSE
        mseminx, mseminy = np.where(mse == np.min(mse[np.nonzero(mse)]))
        optimized_pls_components = mseminx[0] + 1
        num_discarded_wl = mseminy[0]

        # Calculate PLS with optimal components and export values
        # pls = PLSRegression(n_components=self.components, scale=True)
        # pls.fit(X, Y)

        sorted_indexes = np.argsort(np.abs(pls.coef_[:, 0]))
        selected_wl_indexes = np.sort(sorted_indexes[num_discarded_wl:])

        # Qui sotto metto tutto quello che mi serve da salvare
        self.selected_WL = selected_wl_indexes
        self.modello = pls
        self.sorted_indexes = sorted_indexes
        self.mse = mse

        return self

    def transform(self, X, Y=None):
        X = pd.DataFrame(X).iloc[:, self.selected_WL]
        return X

    def fit_transform(
        self, X, Y
    ):  # Ci sarebbe da gestire il caso  in cui Y non viene fornito
        return self.fit(X, Y).transform(X, Y)

    def predict(self, X=None):
        y_pred = self.modello.predict(X)
        return y_pred
