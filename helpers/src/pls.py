#
# QualiCtrl è un progetto di Qualitade S.r.l.
# (c) 2019 e anni successivi Qualitade S.r.l. - Ogni diritto è riservato.
# (c) 2019 and following years by Qualitade S.r.l. -  All right reserved.
#
# Autore: Marcello Vanzulli (marcello.vanzulli@qualitade.com)
# Autore: Nicholas Fiorentini (nicholas.fiorentini@qualitade.com)
#

"""Modulo per le utility di esplorazione dei modelli basati su PLS."""

from sys import stdout

import matplotlib.collections as collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.model_selection import (LeaveOneOut, cross_val_predict,
                                     cross_val_score, validation_curve)


def pls_variable_selection(*, X, Y, max_components=5):
    """
    Esegue una PLS per la selezione delle variabili.

    Ritorna i seguenti DataFrame
        indici delle WL selezionate
        optimized_pls_components,
        num_WL_scartate
    """
    # pylint: disable=too-many-locals

    max_components = max(1, max_components)

    # serve array numpy
    X = X.values

    # Define MSE array to be populated
    mse = np.zeros((max_components, X.shape[1]))

    # Loop over the number of PLS components
    # TODO: ragionare su come ottimizzare questo doppio ciclo for
    for i in range(max_components):
        # Regression with specified number of components, using full spectrum
        pls = PLSRegression(n_components=i + 1, scale=False)
        pls.fit(X, Y)

        # Indices of sort spectra according to ascending absolute value of PLS coefficients
        sorted_indexes = np.argsort(np.abs(pls.coef_[:, 0]))

        # Sort spectra accordingly
        Xsorted = X[:, sorted_indexes]

        # Discard one wavelength at a time of the sorted spectra,
        # regress, and calculate the MSE cross-validation
        for j in range(X.shape[1] - (i + 1)):
            plsj = PLSRegression(n_components=i + 1)
            # FIXME: manca un scale=False ?

            plsj.fit(Xsorted[:, j:], Y)

            Y_cv = cross_val_predict(plsj, Xsorted[:, j:], Y, cv=5)

            mse[i, j] = MSE(Y, Y_cv)

        # progress status
        comp = 100 * (i + 1) / (max_components)
        stdout.write(f"\r{comp:.0f}% completed")
        stdout.flush()
    print("")  # stampa una riga vuota

    # Calculate and print the position of minimum in MSE
    mseminx, mseminy = np.where(mse == np.min(mse[np.nonzero(mse)]))
    optimized_pls_components = mseminx[0] + 1
    num_discarded_wl = mseminy[0]

    print(f"Optimised number of PLS components: {optimized_pls_components}")
    print(f"Wavelengths to be discarded {num_discarded_wl}")
    print(f"Optimised RMSEP {np.sqrt(mse[mseminx, mseminy][0])}")

    # Calculate PLS with optimal components and export values
    pls = PLSRegression(n_components=optimized_pls_components, scale=False)
    pls.fit(X, Y)

    sorted_indexes = np.argsort(np.abs(pls.coef_[:, 0]))
    selected_wl_indexes = np.sort(sorted_indexes[num_discarded_wl:])
    # ovvero taglio le prime num_discarded_wl colonnee riordino gli indici

    return sorted_indexes, selected_wl_indexes, optimized_pls_components, num_discarded_wl


def feature_plot(*, X, selected_wl_indexes):
    """Feature plot della PLS."""
    # serve array numpy
    X = X.values

    # Get an array of WL
    wl_list = np.linspace(908.1, 1676.2, X.shape[1])  # WL range
    # TODO: circa simile a quello del MicroNIR --> FIXARE

    # Get a boolean array according to the indices that are being discarded
    wl_idx = np.in1d(wl_list, wl_list[selected_wl_indexes])

    # Plot spectra with superimpose selected bands
    _, ax = plt.subplots(figsize=(20, 10))
    ax.plot(wl_list, X.T, "C0", alpha=0.3)
    plt.ylabel("Spectra")
    plt.xlabel("Wavelength (nm)")

    ymin = np.min(X)
    ymax = np.max(X)

    collection = collections.BrokenBarHCollection.span_where(
        wl_list,
        ymin=ymin,
        ymax=ymax,
        where=(wl_idx == True),  # noqa
        facecolor="C1",
        alpha=0.3,
    )
    ax.add_collection(collection)
    plt.show()


def pls_cv(*, X, Y, max_components=5):
    """
    Esegue la PLS in cross validation senza la selezione delle variabili. Plotta il risultato della PLS.

    Ritorna il DataFrame con i risultati di cross validation.
    """
    # setup
    max_components = max(1, max_components)
    full_components = max(1, X.shape[1] - 1)

    # dataset per i risultati
    df_results = pd.DataFrame(
        columns=["components", "RMSEcv"],
        index=np.arange(max_components),
    )

    # taglio le componenti e uso array numpy
    X = X.values[:, 0:full_components]

    # itero su tutte le componenti e tutte le combinazioni di cross-validation LOO
    for components in range(
        1, max_components + 1
    ):  # nota che partiamo da 1 perché 0 componenti non ha senso
        # preparo il dataset per le performance di cross-validation
        df_cv = pd.DataFrame(columns=["Yreal", "Ycv"])
        df_cv["Yreal"] = Y
        df_cv["Ycv"] = Y

        # PLS sulla cv (split del set in training e validation)
        for train_index, test_index in LeaveOneOut().split(X):
            pls = PLSRegression(n_components=components, scale=False, max_iter=500).fit(
                X=X[train_index], Y=Y.iloc[train_index]
            )

            # Attenzione che ritorna gli INDICI, non i valori
            df_cv["Ycv"].iloc[test_index] = pls.predict(X[test_index])[
                0
            ]  # calcolo la Y sul campione rimasto in test

        # estrarre RMSE e i parametri che salvo in un dataframe Pandas
        df_results["components"].iloc[components - 1] = components
        df_results["RMSEcv"].iloc[components - 1] = np.std(df_cv["Yreal"] - df_cv["Ycv"])

        # TODO: eventualmente interrompere il ciclo in anticipo se i parametri sono fuori soglia
        # TODO      1. usare un epsilon delta tra due rmsecv
        # TODO      2. interrompere sse il delta < epsilon per n ripetizioni

    # plot della cross-validation
    plt.subplots(figsize=(20, 8))
    plt.style.use("seaborn")
    plt.plot(df_results["components"].values, df_results["RMSEcv"].values)

    return df_results


def sk_crossvalidscore(*, X, Y, cv=20, scoring="neg_root_mean_squared_error"):
    """Esegue lo scoring di cross validation usando la PLS."""
    model = PLSRegression(n_components=5, scale=False)
    scores = cross_val_score(
        model, X, Y, cv=cv, scoring=scoring
    )  # la documentazone riporta che viene restitutio il valore negativo.

    print(f"RMSE di ogni gruppo: {list(abs(scores.round(2)))}")
    print(f"Numero di campioni per gruppo ≈ {np.round(len(Y) / cv, 0)}")
    print(f"RMSE: {abs(scores.mean()):.2f} (± {scores.std() * 2:.2f})")


def pls_cv20(*, X, Y, max_components=5, cv=10, scoring="neg_root_mean_squared_error"):
    """
    Esegue la PLS in cross validation senza la selezione delle variabili. Plotta il risultato della PLS.
    Ritorna il DataFrame con i risultati di cross validation.
    """
    # setup
    max_components = max(1, max_components)
    full_components = max(1, X.shape[1] - 1)

    # dataset per i risultati
    df_results = pd.DataFrame(
        columns=["components", "RMSEcv", "R2", "abs mean of coeff", "mean of coeff"],
        index=np.arange(max_components),
    )

    # taglio le componenti e uso array numpy
    X = X.values[:, 0:full_components]

    # itero su tutte le componenti e tutte le combinazioni di cross-validation LOO
    for components in range(
        1, max_components + 1
    ):  # nota che partiamo da 1 perché 0 componenti non ha senso
        # preparo il dataset per le performance di cross-validation
        df_cv = pd.DataFrame(columns=["Yreal", "Ycv"])
        df_cv["Yreal"] = Y
        df_cv["Ycv"] = Y

        # PLS sulla cv (split del set in training e validation)
        for train_index, test_index in LeaveOneOut().split(X):
            pls = PLSRegression(n_components=components, scale=False, max_iter=500).fit(
                X=X[train_index], Y=Y.iloc[train_index]
            )

            # Attenzione che ritorna gli INDICI, non i valori
            df_cv["Ycv"].iloc[test_index] = pls.predict(X[test_index])[
                0
            ]  # calcolo la Y sul campione rimasto in test

        # estrarre RMSE e i parametri che salvo in un dataframe Pandas
        df_results["components"].iloc[components - 1] = components
        df_results["RMSEcv"].iloc[components - 1] = (
            MSE(df_cv["Yreal"], df_cv["Ycv"]) ** 0.5
        )
        df_results["R2"].iloc[components - 1] = r2_score(df_cv["Yreal"], df_cv["Ycv"])

        # Estraggo e salvo la media dei coefficienti
        df_results["abs mean of coeff"].iloc[components - 1] = (
            abs(pls.coef_).mean().round(2)
        )
        df_results["mean of coeff"].iloc[components - 1] = pls.coef_.mean().round(4)

    ### Varianza Spiegata ###

    # Total Variance
    total_variance_in_x = sum(np.var(X, axis=0))

    # Variance in transformed X data for each latent vector:
    variance_in_x = np.var(pls.x_scores_, axis=0)

    # Normalize variance by total variance:
    fractions_of_explained_variance = variance_in_x / total_variance_in_x

    # Cumulative Explained Variance
    cum_expl_var = pd.DataFrame(1 - fractions_of_explained_variance)

    # Aggiungo ai risultati
    df_results["Explained Variance (Perc)"] = 1 - fractions_of_explained_variance

    ### Errori in cv ###
    param_range = np.arange(1, max_components + 1)

    train_scores, test_scores = validation_curve(
        PLSRegression(scale=False),
        X=X,
        y=Y,
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

    ### PLOTS ###

    # Main settings
    fig, axs = plt.subplots(figsize=(25, 30), nrows=5, ncols=1, sharex=False)
    plt.style.use("seaborn")
    fig.subplots_adjust(hspace=0.2)

    # Plot della cross-validation
    axs[0].set_title("RMSE vs Components")
    axs[0].set_xlabel("Components")
    axs[0].set_ylabel("RMSE")
    axs[0].plot(df_results["components"].values, df_results["RMSEcv"].values)

    axs[1].set_title("Explained Variance vs Components")
    axs[1].set_xlabel("Components")
    axs[1].set_ylabel("Explained Variance (Percentage)")
    axs[1].plot(
        df_results["components"].values, df_results["Explained Variance (Perc)"], c="blue"
    )

    # Plot dell'andamento
    axs[2].set_title("Validation Curve with PLS")
    axs[2].set_xlabel("Components")
    axs[2].set_ylabel("RMSE")
    # axs[2].xlim(1, param_range[-1])
    # axs[2].ylim(0.0, ymax)

    axs[2].plot(
        param_range, np.abs(train_scores_mean), label="Training score", color="darkorange"
    )
    axs[2].fill_between(
        param_range,
        np.abs(train_scores_mean) - train_scores_std,
        np.abs(train_scores_mean) + train_scores_std,
        alpha=0.2,
        color="darkorange",
    )
    axs[2].plot(
        param_range,
        np.abs(test_scores_mean),
        label="Cross-validation score",
        color="navy",
    )
    axs[2].fill_between(
        param_range,
        np.abs(test_scores_mean) - test_scores_std,
        np.abs(test_scores_mean) + test_scores_std,
        alpha=0.2,
        color="navy",
    )
    axs[2].legend(loc="best")

    # Plot dei coefficienti
    axs[3].set_title("Absolute Coefficient vs components")
    axs[3].set_xlabel("Components")
    axs[3].set_ylabel("Mean of abs values of pls coefficient")

    axs[3].plot(df_results["components"].values, df_results["abs mean of coeff"].values)

    # Plot dei coefficienti
    axs[4].set_title("Coefficient vs components")
    axs[4].set_xlabel("Components")
    axs[4].set_ylabel("Mean values of pls coefficient")

    axs[4].plot(df_results["components"].values, df_results["mean of coeff"].values)

    ### PRINTS ###

    # Controllo dei poveri --> man, sarà sempre "huge" se lo fai con un numero di componenti alto.
    # if (abs(pls.coef_).mean().round(2)) > (Y.mean().round(2) * 2):
    #    print(
    #        f"PAY ATTENTION: for {max_components} components, the mean of the coefficients of PLS is huge, probable overfitting"
    #    )
    #    print(f"abs mean of coefficients: {abs(pls.coef_).mean().round(2)}")
    # else:
    #    print("Nothing to say about PLS Coefficients")
        
    plt.close()

    return df_results, pls, fig

    ### Una volta lanciato questo ci sarebbe un altro metodo che impacchetta il modello con il numero di componenti prescelto ###
    ### Sempre dentro questo metodo ci starebbe mettere anche il bootstrapping dell'errore in CV ###
