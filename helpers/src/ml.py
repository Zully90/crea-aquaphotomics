#
# QualiCtrl è un progetto di Qualitade S.r.l.
# (c) 2019 e anni successivi Qualitade S.r.l. - Ogni diritto è riservato.
# (c) 2019 and following years by Qualitade S.r.l. -  All right reserved.
#
# Autore: Marcello Vanzulli (marcello.vanzulli@qualitade.com)
# Autore: Nicholas Fiorentini (nicholas.fiorentini@qualitade.com)
#

"""Modulo per costruire e validare modelli ML."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class ModelML:
    """Classe per gestire e validare un modello di ML con la sua pipeline."""

    def __init__(self, X, Y, pipeline, name: str = "model"):
        """Costruttore."""
        # ATTRIBUTI
        self.pipeline = pipeline
        self.X = X
        self.Y = Y
        self.name = name
        self.model = None
        self.df_results_training = None
        self.df_results_test = None
        self.training_perfomance = None
        self.test_perfomance = None

    def build_and_validate(
        self, *, predicted_rounding=5, test_size=0.25, random_state=42
    ):
        """
        Valida un modello di predizione di ML.

        Model è una pipeline di scikit-learn che include il pre-processing e il modello da applicare.

        Ritorna:
            - il modello fittato
            - un DF con i valori predetti.
        """
        # Parte 1: split, fitting e prediction
        if test_size <= 0:
            X_train = self.X
            Y_train = self.Y
            X_test = self.X
            Y_test = self.Y
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(
                self.X, self.Y, test_size=test_size, random_state=random_state
            )

        # fitta il modello applicando tutta la pipeline
        # Fit all the transforms one after the other and transform the data, then fit the transformed data using
        # the final estimator.
        self.model = self.pipeline.fit(X_train, Y_train)
        self.df_results_training, rmse, r2, bias = self._validation(
            X_test=X_test, Y_test=Y_test, predicted_rounding=predicted_rounding
        )

        self.training_perfomance = {
            "RMSE": rmse,
            "r2": r2,
            "bias": bias,
        }

    def _validation(self, *, X_test, Y_test, predicted_rounding=5):
        """Esegue la validation del modello."""
        df_results = pd.DataFrame(
            columns=["reference", "predicted", "delta_result", "abs_err"]
        )

        # Test del modello
        Y_res = self.model.predict(X_test)
        print(f"Numero campioni: {len(Y_res)}")

        df_results.reference = Y_test
        df_results.predicted = Y_res.round(predicted_rounding)

        # Parte 2: diagnostiche
        df_results.delta_result = df_results.predicted - df_results.reference
        df_results.abs_err = np.abs(df_results.delta_result)
        bias = np.mean(df_results.delta_result)
        print(f"bias: {np.round(bias, predicted_rounding)}")

        rmse = np.sqrt(mean_squared_error(df_results.reference, df_results.predicted))
        print(f"RMSE: {np.round(rmse, predicted_rounding)}")

        # R2
        TSS = ((df_results.reference - df_results.reference.mean()) ** 2).sum()
        RSS = ((df_results.predicted - df_results.reference) ** 2).sum()
        r2 = 1.0 - (RSS / TSS)
        print(f"R2: {np.round(r2, predicted_rounding)}")

        return df_results, rmse, r2, bias

    def validation(self, *, X_test, Y_test, predicted_rounding=5):
        """Esegue la validation e salva i risultati nel dataframe di validation."""
        self.df_results_test, rmse, r2, bias = self._validation(
            X_test=X_test, Y_test=Y_test, predicted_rounding=predicted_rounding
        )

        self.test_perfomance = {
            "RMSE": rmse,
            "r2": r2,
            "bias": bias,
        }

    def training_plots(self):
        """Plot per i risultati di training di un modello"""
        self._plots(df=self.df_results_training, title="training")

    def validation_plots(self):
        """Plot per i risultati di validation di un modello"""
        self._plots(df=self.df_results_test, title="test")

    def _plots(self, *, df, title: str):
        """
        Plot per i risultati di un modello.

        Il DataFrame è quello prodotto da _validation.
        """
        fig, [ax1, ax2, ax3] = plt.subplots(
            nrows=1, ncols=3, figsize=(40, 15), squeeze=True
        )

        # BoxPlot degli errori assoluti
        fig.suptitle(f"{self.name}_{title}", fontsize=20)
        ax1 = sns.boxplot(x=df.abs_err, orient="h", ax=ax1, color="C4")

        # RegPlot predicted vs reference
        ax2 = sns.regplot(
            x=df.reference,
            y=df.predicted,
            ax=ax2,
            truncate=False,
        )
        ideal = np.linspace(
            np.min([df.reference.min(), df.predicted.min()]),
            np.max([df.reference.max(), df.predicted.max()]),
        )
        ax2.plot(ideal, ideal, "k", alpha=0.2, linestyle="dashed")

        # DistPlot degli errori
        ax3 = sns.histplot(
            df.delta_result,
            ax=ax3,
            kde=True,
            color="C1",
            alpha=0.2,
        )

        plt.show()

    def dump(self, path: Path = Path("./")) -> None:
        """
        Questo comando esegue il dump del modello al path indicato. Es:

        path:
            Il Path dove salvare il file. Default: ./


        UTILIZZO

        self.dump()

            genera il file "model.joblib" nella cartella corrente

        altrimenti

        self.dump(path=Path("./") / "models_example")

            genera il file "model.joblib" nella sottocartella (che deve esistere) "models_example"
        """
        assert self.model is not None, "Model not yet trained"  # nosec
        assert path.exists() is True, "Path not found"  # nosec

        dump(self.model, path / f"{self.name}.joblib")

    def generate_test_csv(
        self,
        raw_dataset: pd.DataFrame,
        columns_removal: Optional[List[str]] = None,
        path: Path = Path("./"),
    ) -> pd.DataFrame:
        """
        Questo comando genera un CSV con gli spettri per eseguire gli unit test. Ritorna il DataFrame usato per la
        generazione del csv.

        raw_dataset: DataFrame
            Il DataFrame originale.

        columns_removal: lista di stringhe
            L'elenco delle colonne da rimuovere dal raw_dataset che non devono essere salvate nel CSV (dopo il merge).

        path: Path
            Il Path dove salvare il file. Default: ./

        UTILIZZO

        self.dump()

            genera il file "test_model.csv" nella cartella corrente

        self.dump(path=Path("./") / "models_example")

            genera il file "test_model.csv" nella sottocartella (che deve esistere) "models_example"
        """
        assert (  # nosec
            self.model is not None
        ), "Model not yet trained"  # quindi non ho nemmeno un dataset di riferimenti
        assert path.exists() is True, "Path not found"  # nosec

        df_test = pd.merge(
            raw_dataset,
            self.df_results_test.copy().sort_index(),  # riordino le righe
            left_index=True,
            right_index=True,
        )

        # rimuovo colonne che non mi servono
        for column in (columns_removal or []) + [
            # aggiungo le colonne presentin in self.df_results
            "reference",
            "delta_result",
            "abs_err",
        ]:
            del df_test[column]

        # esporto il dataframe in csv
        df_test.to_csv(path / f"test_{self.name}.csv", index=False)

        return df_test
