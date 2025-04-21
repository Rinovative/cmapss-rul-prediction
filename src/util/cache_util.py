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


def load_object(path: str):
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
        # Lade das Bild zurück, aber als Figure-Objekt
        return plt.imread(path)  # Dies wird das Bild laden, aber als Numpy-Array zurückgeben
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def cache_plot(plot_func, df, dataset_name, kind, force_recompute=False, *args, **kwargs):
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
            print(f"Loading cached plot and labels for {dataset_name}...")
            fig = load_object(fig_cache_path)  # Lade das komplette Figure-Objekt
            labels = load_object(labels_cache_path)
            return fig, labels

    # Wenn der Plot neu berechnet werden muss, führe die Plot-Funktion aus
    print(f"Recomputing plot and labels for {dataset_name}...")
    fig, labels = plot_func(df, dataset_name, *args, **kwargs)

    # Speichere das Plot und die Labels im Cache
    save_object(fig, fig_cache_path)
    save_object(labels, labels_cache_path)

    return fig, labels
