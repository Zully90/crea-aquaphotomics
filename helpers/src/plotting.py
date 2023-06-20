#
# QualiCtrl è un progetto di Qualitade S.r.l.
# (c) 2019 e anni successivi Qualitade S.r.l. - Ogni diritto è riservato.
# (c) 2019 and following years by Qualitade S.r.l. -  All right reserved.
#
# Autore: Marcello Vanzulli (marcello.vanzulli@qualitade.com)
# Autore: Nicholas Fiorentini (nicholas.fiorentini@qualitade.com)
#

"""Helpers per il plot su Jupyer."""

import matplotlib.pyplot as plt


def plot_spectral_data(*, df):
    """
    Plot degli spettri grezzi.

    Input: un dataframe che per righe i singoli spettri (i campioni) e per colonne le lunghezze d'onda.
    """
    plt.figure(figsize=(20, 10))
    plt.plot(df.T, "C0", alpha=0.3)
    plt.xticks(rotation=60, horizontalalignment="right")
    plt.show()
