#
# QualiCtrl è un progetto di Qualitade S.r.l.
# (c) 2019 e anni successivi Qualitade S.r.l. - Ogni diritto è riservato.
# (c) 2019 and following years by Qualitade S.r.l. -  All right reserved.
#
# Autore: Marcello Vanzulli (marcello.vanzulli@qualitade.com)
# Autore: Nicholas Fiorentini (nicholas.fiorentini@qualitade.com)
#

"""Implementazione dell'algoritmo di Kennard-Stone."""

import numpy as np


def kennardstonealgorithm(X, k):
    """
    Select samples using Kennard-Stone algorithm

    Input:

        X dataset of X-variables (samples x variables)
        k number of samples to be selected

    Output:
        selectedsamplenumbers selected sample numbers (training data)
        remainingsamplenumbers remaining sample numbers (test data)
    """
    # FIXME: da sistemare, ripulire, commentare, testare

    X = np.array(X)
    originalX = X
    distancetoaverage = ((X - np.tile(X.mean(axis=0), (X.shape[0], 1))) ** 2).sum(axis=1)
    maxdistancesamplenumber = np.where(distancetoaverage == np.max(distancetoaverage))
    maxdistancesamplenumber = maxdistancesamplenumber[0][0]
    selectedsamplenumbers = list(maxdistancesamplenumber)
    remainingsamplenumbers = np.arange(0, X.shape[0], 1)
    X = np.delete(X, selectedsamplenumbers, 0)
    remainingsamplenumbers = np.delete(remainingsamplenumbers, selectedsamplenumbers, 0)

    for _ in range(1, k):
        selectedsamples = originalX[selectedsamplenumbers, :]
        mindistancetoselectedsamples = list()

        for mindistancecalculationnumber in range(0, X.shape[0]):
            distancetoselectedsamples = (
                (
                    selectedsamples
                    - np.tile(
                        X[mindistancecalculationnumber, :], (selectedsamples.shape[0], 1)
                    )
                )
                ** 2
            ).sum(axis=1)
            mindistancetoselectedsamples.append(np.min(distancetoselectedsamples))

        maxdistancesamplenumber = np.where(
            mindistancetoselectedsamples == np.max(mindistancetoselectedsamples)
        )
        maxdistancesamplenumber = maxdistancesamplenumber[0][0]
        selectedsamplenumbers.append(remainingsamplenumbers[maxdistancesamplenumber])
        X = np.delete(X, maxdistancesamplenumber, 0)
        remainingsamplenumbers = np.delete(
            remainingsamplenumbers, maxdistancesamplenumber, 0
        )

    return selectedsamplenumbers, remainingsamplenumbers
