import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

CACHE_DIR = "cache"


def get_cache_path(dataset_name: str, kind: str, name: str, ext: str):
    """
    Gibt den vollständigen Cache-Pfad für ein bestimmtes Dataset und einen bestimmten Dateityp zurück.
    """
    path = os.path.join(CACHE_DIR, dataset_name.lower(), kind)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"{name}.{ext}")


def save_object(obj, path: str):
    """
    Speichert ein Objekt (z.B. Figure, Labels) an einem angegebenen Pfad.
    """
    ext = os.path.splitext(path)[1]
    if ext == ".npy":
        np.save(path, obj)
    elif ext == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    elif ext == ".png":
        obj.savefig(path, bbox_inches="tight")
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def _load_object(path: str):
    """
    Lädt ein Objekt (z.B. Figure, Labels) von einem angegebenen Pfad.
    """
    ext = os.path.splitext(path)[1]
    if ext == ".npy":
        return np.load(path, allow_pickle=True)
    elif ext == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif ext == ".png":
        return plt.imread(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def cache_plot_pickle(plot_func, df, dataset_name, kind, force_recompute=False, *args, **kwargs):
    """
    Versucht, ein gespeichertes Plot-Objekt aus dem Cache zu laden.
    Falls es nicht existiert oder das Neuberechnen erzwungen wird, wird das Plot neu berechnet, gespeichert und zurückgegeben.

    Args:
        plot_func: Die Funktion, die den Plot berechnet (z.B. `plot_tsne_dbscan_clusters_internal`).
        df: Der Datensatz, der für den Plot verwendet wird.
        dataset_name: Der Name des Datensatzes (z.B. "FD001").
        kind: Der Typ der gespeicherten Datei (z.B. "figures").
        force_recompute: Ob das Plot neu berechnet werden soll, selbst wenn es bereits im Cache existiert.
        *args, **kwargs: Weitere Argumente, die an die Plot-Funktion übergeben werden.

    Returns:
        Das Plot (Figure) und die zugehörigen Labels.
    """
    # Erzeuge den Cache-Pfad für das Plot
    fig_cache_path = get_cache_path(dataset_name, kind, f"{dataset_name}_plot", "pkl")
    labels_cache_path = get_cache_path(dataset_name, kind, f"{dataset_name}_labels", "pkl")

    # Überprüfe, ob der Plot und die Labels bereits im Cache sind und lade sie, wenn verfügbar
    if not force_recompute:
        if os.path.exists(fig_cache_path) and os.path.exists(labels_cache_path):
            # Lade das gespeicherte Plot und die Labels
            fig = _load_object(fig_cache_path)  # Lade das komplette Figure-Objekt
            labels = _load_object(labels_cache_path)
            return fig, labels

    # Wenn der Plot neu berechnet werden muss, führe die Plot-Funktion aus
    fig, labels = plot_func(df, dataset_name, *args, **kwargs)

    # Speichere das Plot und die Labels im Cache
    save_object(fig, fig_cache_path)
    save_object(labels, labels_cache_path)

    return fig, labels

def cache_all_plots(plot_lists, dataset_name, force_recompute=False):
    """
    Berechnet und cached alle Plots für eine gegebene Liste von Plotfunktionen.

    Für jede Plotfunktion in den übergebenen Listen wird geprüft, ob ein gecachter Plot bereits vorhanden ist.
    Falls nicht (oder falls force_recompute=True), wird die Plotfunktion ausgeführt, das Ergebnis als PNG gespeichert
    und die Figure geschlossen.

    Args:
        plot_lists (list): Liste von Plotlisten. Jede Plotliste enthält Tripel (title, func, plot_name).
        dataset_name (str): Name des Datensatzes (wird für den Cache-Pfad verwendet).
        force_recompute (bool): Wenn True, werden alle Plots neu berechnet, auch wenn sie bereits gecacht sind.
    """
    for plot_list in plot_lists:
        for title, func, plot_name in plot_list:
            png_path = get_cache_path(dataset_name.lower(), "figures", plot_name, "png")
            if force_recompute or not os.path.exists(png_path):
                plt.ioff()
                result = func()
                if isinstance(result, tuple):
                    result = result[0]
                if isinstance(result, plt.Axes):
                    result = result.figure
                if isinstance(result, plt.Figure):
                    save_object(result, png_path)
                    plt.close(result)
                plt.close("all")
                plt.ion()
