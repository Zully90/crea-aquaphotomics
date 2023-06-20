#
# QualiCtrl è un progetto di Qualitade S.r.l.
# (c) 2019 e anni successivi Qualitade S.r.l. - Ogni diritto è riservato.
# (c) 2019 and following years by Qualitade S.r.l. -  All right reserved.
#
# Autore: Marcello Vanzulli (marcello.vanzulli@qualitade.com)
# Autore: Nicholas Fiorentini (nicholas.fiorentini@qualitade.com)
#

"""Modulo per la classe che gestisce una PCA."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from statsmodels.multivariate.pca import PCA as smPCA  # noqa


class PCA:
    """Calcola la PCA e permette di analizzarne i risultati."""

    # CONSTRUCTOR
    def __init__(self, *, values, components=3):
        """
        Costruttore.

        values
            Array numpy con i samples per righe e le features per colonne.

        components
            Il numero intero positivo di componenti della PCA.
            Default: 3
        """
        # ATTRIBUTES
        self.values = values
        self.components = max(1, components)
        self.model = None
        self.loadings = None
        self.scores = None
        self.E = None  # la matrice ricostruita
        self.varianza_spiegata = None
        self.num_total_components = len(self.values)
        self.T_Q = pd.DataFrame(
            columns=["T_Hotelling", "Q"], index=np.arange(self.num_total_components)
        )

        # initialization
        self._compute()

    # METHODS
    def _compute(self):
        """Crea il modello della PCA."""
        # Modello
        self.model = smPCA(
            self.values, standardize=False, method="nipals", demean=False, normalize=False
        )

        # Calcola la varianza spiegata
        self.varianza_spiegata = (self.model.eigenvals / self.model.eigenvals.sum()) * 100

        # Ottieni Loadings & Scores
        self.loadings = self.model.loadings[:, : self.components].copy()
        self.scores = self.model.scores[:, : self.components].copy()

        # Matrice ricostruita
        hat = self.scores.dot(self.loadings.T)  # Dovrebbe essere la autoscalata
        self.E = np.array(np.matrix(self.values) - np.matrix(hat))

        # per ultimo verifico che sia tutto ok
        check_varianza = np.round(self.varianza_spiegata.sum(), decimals=2)
        assert (  # nosec
            check_varianza == 100.0
        ), "ERRORE: la varianza spiegata non è a somma 100"

    def show_varianza_spiegata(self, *, epsilon=0.001, max_components=None):
        """
        Mostra le informazioni sul modello PCA generato mostrando la varianza spiegata.

        epsilon
            Taglia tutta la varianza spiegata minore di epsilon nelle stampe informative.
            Default: 0.001

        max_components
            Numero massimo di componenti da plottare nel grafico della varianza spiegata.
            Default: None (tutti)
        """
        # info generiche sul modello
        print(self.model)

        # stampa le varianze più significative
        with np.printoptions(precision=3, suppress=True):
            print(self.varianza_spiegata[self.varianza_spiegata >= epsilon])

        # plot di tutta la varianza spiegata
        plot_varianza = self.varianza_spiegata
        if max_components is not None and max_components > 0:
            plot_varianza = self.varianza_spiegata[:max_components]

        plt.subplots(figsize=(20, 8))
        plt.style.use("seaborn")
        plt.plot(plot_varianza)
        plt.show()

    def hq_diagnostics(self):
        """
        Calcola la diagnostica HQ e salva T_Q.

        Ritorna i seguenti DataFrame:
            T_Contributions, Q_contributions
        """
        # Autovalori
        cov = np.corrcoef(self.values, rowvar=False)  # calcolo matrice di covarianza
        eigs = np.linalg.eigvals(cov)  # calcolo autovalori
        eigs_real = np.real(eigs)  # prendo parte reale
        eig_diag = np.diag(eigs_real)

        # Valori di T_quadro
        eig_diag_red = eig_diag[0 : self.components, 0 : self.components]  # noqa
        eig_diag_red_inv = np.linalg.inv(eig_diag_red)
        self.T_Q["T_Hotelling"] = np.apply_along_axis(
            lambda row: row.T.dot(eig_diag_red_inv).dot(row),
            # lambda è applicata a ogni riga della matrice degli scores
            axis=1,
            arr=self.scores,
        )

        # Valori di Q
        # Calcolo dei residui Q di ogni oggetto (somma dei quadrati delle righe)
        self.T_Q["Q"] = np.sum(np.power(self.E, 2), axis=1)

        # TContributions
        T_Contributions = pd.DataFrame(
            np.matrix(self.scores)
            .dot(sp.linalg.fractional_matrix_power(eig_diag_red, -0.5))
            .dot(np.matrix(self.loadings).T)
        )

        # QContributions
        Q_contributions = pd.DataFrame(
            np.apply_along_axis(
                lambda row: row * row * np.sign(row),
                # lambda è applicata a ogni riga della matrice E
                axis=1,
                arr=self.E,
            )
        )

        return (
            T_Contributions,
            Q_contributions,
        )

    def compute_outliers(self, alpha=0.05):
        """Ritorna i DataFrame degli outliers e inliers della PCA."""
        # TODO migliorare i nomi delle variabili di questo metodo
        r = self.num_total_components  # numero di righe
        log10_q = np.log10(self.T_Q.Q)

        t_tab = sp.stats.t.ppf(1 - alpha, r - 1)  # 1 coda
        lq = np.power(10, t_tab * np.std(log10_q) + np.mean(log10_q))
        F_tab = sp.stats.f.ppf(1 - alpha, self.components, r - self.components)
        lt = self.components * (r - 1) / (r - self.components) * F_tab

        out_pca = self.T_Q[(self.T_Q.Q > lq) | (self.T_Q.T_Hotelling > lt)]
        in_pca = pd.DataFrame(self.values).drop(out_pca.index, axis="index")

        # output e verifiche finali
        if out_pca.empty:
            print("No outliers detected")
        else:
            print(f"Individuati {len(out_pca)} outliers")

        assert (  # nosec
            len(in_pca) + len(out_pca) == r
        ), "Dimensione di outliers + inliers incompatibile"
        return out_pca, in_pca
