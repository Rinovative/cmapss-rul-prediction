import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from IPython.display import Image, display

from src.util import cache_util


def _sanitize_name(name: str) -> str:
    """
    Normalisiert einen Plotnamen zu einem Dateinamen-kompatiblen Format.

    - Wandelt Grossbuchstaben in Kleinbuchstaben um.
    - Ersetzt Leerzeichen durch Unterstriche.
    - Ersetzt verschiedene Bindestrich-Varianten durch normalen Bindestrich.
    - Ersetzt Slashes durch Unterstriche.

    Args:
        name (str): Ursprünglicher Plotname oder Titel.

    Returns:
        str: Sanitized Plotname.
    """
    return name.lower().replace(" ", "_").replace("–", "-").replace("—", "-").replace("/", "_")


def _show_anything(result, fallback_path=None):
    """
    Zeigt beliebige Objekttypen interaktiv an (Matplotlib, Plotly, Bild, String, etc.).

    - Falls ein Fallback-Pfad angegeben und die Datei existiert, wird dieses Bild angezeigt.
    - Unterstützt Matplotlib-Figuren, Plotly-Figuren, beliebige Objekte mit .show()-Methode, Strings und andere displaybare Objekte.

    Args:
        result: Das anzuzeigende Ergebnisobjekt.
        fallback_path (str, optional): Pfad zu einem Bild, falls das Ergebnis nicht angezeigt werden kann.

    Returns:
        None
    """
    if fallback_path and os.path.exists(fallback_path):
        display(Image(filename=fallback_path))
        return

    if isinstance(result, plt.Figure):
        display(result)
    elif isinstance(result, go.Figure):
        display(result)
    elif hasattr(result, "show") and callable(result.show):
        display(result)
    elif isinstance(result, str):
        print(result)
    elif result is not None:
        display(result)


def make_dropdown_section(plots, dataset_name, description="Plot:"):
    """
    Erstellt eine Section (VBox) mit Dropdown zur Auswahl eines Plots (Lazy Loading!).
    plots: Liste von (Titel, plot_func, plot_name)
    """
    dropdown = widgets.Dropdown(
        options=[(title, i) for i, (title, _, _) in enumerate(plots)], description=description, style={"description_width": "initial"}
    )
    output = widgets.Output()
    last_idx = {"idx": None}

    def on_plot_change(change):
        idx = change["new"]
        if last_idx["idx"] == idx:
            return
        plot_func = plots[idx][1]
        plot_name = plots[idx][2]
        png_path = cache_util.get_cache_path(dataset_name, "figures", plot_name, "png")
        with output:
            output.clear_output(wait=True)
            plt.close("all")  # Speicher freigeben
            if not os.path.exists(png_path):
                result = plot_func()
                if isinstance(result, tuple):
                    result = result[0]
                if isinstance(result, plt.Figure):
                    cache_util.save_object(result, png_path)
                    plt.close(result)
                    display(Image(filename=png_path))
                else:
                    _show_anything(result)
            else:
                display(Image(filename=png_path))
        last_idx["idx"] = idx

    dropdown.observe(on_plot_change, names="value")
    # Direkt den ersten Plot anzeigen
    on_plot_change({"type": "change", "name": "value", "new": 0})

    return widgets.VBox([dropdown, output])


def make_toggle_shortcut(df, dataset_name):
    """
    Gibt eine Funktion zurück, mit der Dropdown-Plots erstellt werden können:
    toggle(title, func, plot_name=None, **kwargs)
    """
    counter = {"i": 0}
    sanitized_dataset = _sanitize_name(dataset_name)

    def toggle(title, func, plot_name=None, **kwargs):
        if "dataset_name" in func.__code__.co_varnames:
            kwargs.setdefault("dataset_name", dataset_name)
        if plot_name is None:
            plot_name = f"{sanitized_dataset}_plot_{counter['i']:03d}"
            counter["i"] += 1
        else:
            plot_name = _sanitize_name(plot_name)
        # Gibt zurück: (Tab-Titel, Plotfunktion, Plotname für Cache)
        return (title, lambda: func(df=df, **kwargs), plot_name)

    return toggle


def create_standard_tabs(sections):
    """
    Gibt das Haupt-Tab-Widget zurück. Jeder Tab enthält eine Dropdown-Auswahl für die jeweilige Kategorie.
    """
    tabs = widgets.Tab(children=sections)
    default_titles = [
        "1. Übersicht",
        "2. Operation Settings",
        "3. Sensoren",
        "4. Klassifizierung",
        "5. Clusteranalyse",
    ]
    for i, section in enumerate(sections):
        title = default_titles[i] if i < len(default_titles) else f"Tab {i + 1}"
        tabs.set_title(i, title)
    return tabs
