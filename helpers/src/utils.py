#
# QualiCtrl è un progetto di Qualitade S.r.l.
# (c) 2019 e anni successivi Qualitade S.r.l. - Ogni diritto è riservato.
# (c) 2019 and following years by Qualitade S.r.l. -  All right reserved.
#
# Autore: Marcello Vanzulli (marcello.vanzulli@qualitade.com)
# Autore: Nicholas Fiorentini (nicholas.fiorentini@qualitade.com)
#

"""Modulo con utilità e miscellanee."""

from typing import Any, Dict


def make_window_index(*, total_size: int = 125, windows: int) -> Dict[str, Any]:
    """
    Crea gli indici di colonna per suddividere `total_size` colonne in tante finestre come indicato da `windows`. Ogni
    finestra è omogenea come dimensione, ovvero le finestre hanno tutte la stessa dimensione (stesso numero di indici).
    Ovviamente, questo è possibile sse `windows` è divisibile per `total_size`. In caso contrario, in automatico vengono
    scartati gli indici estremi (da ambo i lati) per garantire che `total_size` sia divisibile per `windows`.

    total_size: Intero
        Il numero totale di indici su cui lavorare. Default: 125.

    windows: Intero
        Il numero di finestre desiderate.

    Ritorna un dict con:
        - window_size: la dimensione omogenea di tutte le finestre
        - windows: lista di liste di interi. Ogni sotto-lista è una lista di indici della finestra.
    """
    # check
    if total_size <= 0:
        raise ValueError("size must be greater than 0.")

    if windows <= 0:
        raise ValueError("windows must be greater than 0.")

    if windows > total_size:
        raise ValueError("windows must be less or equal than size.")

    # calcolo la dimensione della finestra
    rem = total_size % windows
    window_start = 0
    window_end = total_size

    if rem != 0:
        # calcolo quanti indici scartare agli estremi: se il resto è pari allora posso scartare un numero uguale
        # agli estremi, altrimenti se è dispari scarto un numero in più all'inizio.
        balance = int(rem / 2)
        window_start = balance + (rem % 2)
        window_end = total_size - balance
        total_size = total_size - rem

    window_size = int(total_size / windows)

    return {
        "window_size": window_size,
        "windows": [
            list(range(step, step + window_size))
            for step in range(window_start, window_end, window_size)
        ],
    }
