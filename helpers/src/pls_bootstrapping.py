#
# QualiCtrl è un progetto di Qualitade S.r.l.
# (c) 2019 e anni successivi Qualitade S.r.l. - Ogni diritto è riservato.
# (c) 2019 and following years by Qualitade S.r.l. -  All right reserved.
#
# Autore: Marcello Vanzulli (marcello.vanzulli@qualitade.com)
# Autore: Nicholas Fiorentini (nicholas.fiorentini@qualitade.com)
#

"""Modulo per il bootstrapping dei residui basato su PLS."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve
from sklearn.utils import resample


class PLSBootstrapping:
    """Classe per gestire il bootstrapping dei residui per trovare gli outliers tra i campioni."""

    def __init__(self, X, Y):
        """Costruttore"""
        # ATTRIBUTI
        self.X = X
        self.Y = Y.values
        self.df_selected_samples = None
        self.components = 3

    def run(self, *, components=3, iterations=500, ptrain=0.75, alpha=0.05):
        """
        Esegue il bootstrapping dei residui per trovare gli outliers (tra i campioni).

        ptrain
            La percentuale di campioni da dedicare al training.
            Default: 0.75
        """
        # pylint: disable=too-many-locals

        self.components = max(1, components)

        # parametri del bootstrapping
        num_samples = len(self.Y)
        indexes = np.arange(num_samples)
        fractioning = int(num_samples * ptrain)

        # il dataframe finale con tutti i risultati di prediction nelle varie iterazioni
        df_final = pd.DataFrame(index=indexes, columns=range(iterations))

        # the Bootstrapping
        for i in range(iterations):
            # scelgo un insieme di campioni random per il training
            training_indexes = np.random.choice(indexes, fractioning, replace=False)

            # fitting PLS
            X_train = self.X.iloc[training_indexes, :]
            Y_train = self.Y[training_indexes]

            pls = PLSRegression(
                n_components=self.components
            )  # FIXME: manca un scale=False ?
            pls.fit(X_train, Y_train)

            # Predictions con i valori non usati per il training
            validation_indexes = np.setdiff1d(indexes, training_indexes)
            X_validation = self.X.iloc[validation_indexes]
            Y_validation = pls.predict(X_validation)

            # salvare le Y di predizione nel nostro dataframe (un df composto dalle predizioni)
            df_final.iloc[
                validation_indexes, i
            ] = Y_validation.flatten()  # flatten trasforma l'array in 1D

        # Leverage e studentizzazione
        """
        Reference:
        https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/regression/how-to/partial-least-squares/methods-and-formulas/model-information/#leverages
        https://www.wiki.eigenvector.com/index.php?title=Pls
        """
        pls = PLSRegression(n_components=self.components, scale=False)
        pls.fit(self.X, self.Y)

        scores_x = np.matrix(pls.x_scores_)
        leverage = (
            scores_x @ np.linalg.inv(scores_x.T @ scores_x) @ scores_x.T + 1 / num_samples
        )  # nota: @ è np.dot
        diag = np.diagonal(leverage)

        residui = df_final.subtract(self.Y, axis=0)
        media = residui.mean(axis=1)
        MSEr = np.power(media, 2).sum() / (
            len(media - self.components)
        )  # MSE = sum((res).^2)./(m-ncomp); PRESO DA EIGENVECTOR (PAGINA PLS)
        student = media.values / np.sqrt(
            MSEr * (1 - diag)
        )  # syres = res./sqrt(MSE.*(1-L)); ; preso sempre da eigenvector

        # Limite del leveraggio
        limit = 2 * self.components / num_samples

        # Limite dei residui
        limit_alpha = sp.stats.t.ppf(1 - alpha, len(diag) - 1)  # 1 coda

        # plots
        _, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(40, 10), squeeze=True)

        # BarPlot dei leverage
        sns.barplot(x=indexes, y=diag, ax=ax1)

        # scatterplot
        sns.scatterplot(x=diag, y=student, size=self.Y, sizes=(25, 500), ax=ax2)

        # plot dei limiti
        plt.axhline(y=limit_alpha, linewidth=2, color="r")
        plt.axhline(y=-limit_alpha, linewidth=2, color="r")
        plt.axvline(x=limit, linewidth=2, color="r")
        plt.show()

        # output
        self.df_selected_samples = self.X.copy()
        self.df_selected_samples["Leverage"] = diag
        self.df_selected_samples["Student"] = student
        self.df_selected_samples["Y"] = self.Y

        self.df_selected_samples = self.df_selected_samples[
            (self.df_selected_samples.Student < abs(limit_alpha))
            & (self.df_selected_samples.Leverage < limit)
        ]

    def plot_overfitting(
        self, *, cv=5, max_components=30, scoring="neg_root_mean_squared_error"
    ):
        """
        Analisi di overfitting del modello.

        Notare che questa strategia di CV ti mostra quanto oscilla il tuo parametro qualitativo sulla base dello split.
        Questo suggerisce un "grado di completezza" del dataset.
        """
        # setup
        leverageIdx = self.df_selected_samples.columns.get_loc(
            "Leverage"
        )  # per sapere dove finiscono le colonne degli spettri
        param_range = np.arange(
            1, min(max_components, leverageIdx)
        )  # nel caso ci fossero meno componenti disponibili

        train_scores, test_scores = validation_curve(
            PLSRegression(scale=False),
            self.df_selected_samples.iloc[:, 0:leverageIdx],
            self.df_selected_samples.Y.values,
            param_name="n_components",
            cv=cv,
            param_range=param_range,
            scoring=scoring,
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        ymax = np.max(np.abs(test_scores_mean) + np.abs(test_scores_std))

        plt.figure(figsize=(15, 10))
        plt.title("Validation Curve with PLS")
        plt.xlabel("Components")
        plt.ylabel("RMSE")
        plt.xlim(1, param_range[-1])
        plt.ylim(0.0, ymax)

        plt.plot(
            param_range,
            np.abs(train_scores_mean),
            label="Training score",
            color="darkorange",
        )

        plt.fill_between(
            param_range,
            np.abs(train_scores_mean) - train_scores_std,
            np.abs(train_scores_mean) + train_scores_std,
            alpha=0.2,
            color="darkorange",
        )

        plt.plot(
            param_range,
            np.abs(test_scores_mean),
            label="Cross-validation score",
            color="navy",
        )

        plt.fill_between(
            param_range,
            np.abs(test_scores_mean) - test_scores_std,
            np.abs(test_scores_mean) + test_scores_std,
            alpha=0.2,
            color="navy",
        )

        plt.legend(loc="best")
        plt.show()

    def performance(self, *, iterations=1000, ptrain=0.75, alpha=0.95):
        """
        Analizza le performance del bootstrapping.

        ptrain
            La percentuale di campioni da dedicare al training.
            Default: 0.75
        """
        # pylint: disable=too-many-locals

        # setup bootstrap
        training_size = int(len(self.df_selected_samples) * ptrain)

        df = self.df_selected_samples.copy()
        del df["Leverage"]
        del df["Student"]

        # run bootstrapping
        stats = list()
        for _ in range(iterations):
            # prepare train and test sets
            train = resample(df, n_samples=training_size, replace=False)

            # fit model
            model = PLSRegression(n_components=self.components, scale=False)
            model.fit(train.iloc[:, :-1], train.iloc[:, -1])

            # evaluate model
            test_index = df.index.difference(
                train.index
            )  # sono gli indici "rimasti fuori" dal train
            test_set = df.drop(test_index)
            predictions = model.predict(test_set.iloc[:, :-1])
            score = mean_squared_error(test_set.iloc[:, -1], predictions)
            stats.append(score)

        # plot scores
        plt.hist(np.sqrt(stats))
        plt.show()

        # confidence intervals & RMSE
        lower = max(0, np.percentile(stats, ((1.0 - alpha) / 2.0) * 100))
        upper = min(1, np.percentile(stats, (alpha + ((1.0 - alpha) / 2.0)) * 100))

        RMSE = np.sqrt(np.mean(stats))

        print(f"RMSE: {RMSE.round(2)}")
        print(
            f"{alpha}% confidence interval is between {np.sqrt(lower).round(2)} and {np.sqrt(upper).round(2)}"
        )
