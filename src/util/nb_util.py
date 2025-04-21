import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import Image, display

from src.util import cache_util


def make_toggle_plot(title, plot_func, dataset_name=None, plot_name=None, open_by_default=False, force_recompute=False):
    toggle = widgets.Accordion(children=[widgets.Output()])
    toggle.set_title(0, title)
    out = toggle.children[0]

    with out:
        try:
            assert dataset_name is not None and plot_name is not None
            png_path = cache_util.get_cache_path(dataset_name, "figures", plot_name, "png")

            if not os.path.exists(png_path) or force_recompute:
                plt.ioff()
                fig = plot_func()
                if isinstance(fig, tuple):
                    fig = fig[0]
                if isinstance(fig, plt.Figure):
                    cache_util.save_object(fig, png_path)
                    plt.close(fig)
                else:
                    raise TypeError  # Force fallback

            display(Image(filename=png_path))

        except (AssertionError, TypeError):
            result = plot_func()
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, plt.Figure):
                display(result)
            elif isinstance(result, str):
                print(result)
            elif result is not None:
                display(result)

    toggle.selected_index = 0 if open_by_default else None
    return toggle


def sanitize_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("–", "-").replace("—", "-").replace("/", "_")


def make_toggle_shortcut(df, dataset_name):
    counter = {"i": 0}

    cache_dataset = sanitize_name(dataset_name.split("–")[0].strip())
    sanitized_dataset = sanitize_name(dataset_name)

    def toggle(title, func, plot_name=None, **kwargs):
        if "dataset_name" in func.__code__.co_varnames:
            kwargs.setdefault("dataset_name", dataset_name)

        if plot_name is None:
            plot_name = f"{sanitized_dataset}_plot_{counter['i']:03d}"
            counter["i"] += 1
        else:
            plot_name = sanitize_name(plot_name)

        return make_toggle_plot(
            title,
            lambda: func(df=df, **kwargs),
            dataset_name=cache_dataset,
            plot_name=plot_name,
        )

    return toggle


def create_standard_tabs(sections):
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
