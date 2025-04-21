[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Rinovative/cmapss-rul-prediction/HEAD)  
_Interaktives Jupyter Notebook direkt im Browser öffnen (via Binder)_

# Predictive Maintenance – RUL-Vorhersage mit C-MAPSS (Semesterprojekt)

**Semesterprojekt** im Rahmen des Studiengangs  
**BSc Systemtechnik – Vertiefung Computational Engineering**  
**Frühjahr 2025** – OST – Ostschweizer Fachhochschule  
**Autor:** Rino Albertin

---

## 📌 Projektbeschreibung

Dieses Projekt befasst sich mit der Vorhersage der verbleibenden Nutzungsdauer (Remaining Useful Life, RUL) von Flugtriebwerken anhand des NASA C-MAPSS-Datensatzes.  
Ziel ist die Entwicklung eines robusten, generalisierbaren Machine-Learning-Modells zur frühzeitigen Identifikation von Wartungsbedarf.

Das Projekt umfasst:

- explorative Datenanalyse (EDA) für alle vier C-MAPSS-Datensätze (FD001–FD004),
- Feature Engineering und Zustandserkennung per Clusteranalyse (t-SNE, DBSCAN),
- hyperparametrisierte ML-Pipelines zur RUL-Prognose,
- Entwicklung eines Generalmodells für alle Szenarien.

Eine interaktive Notebook-Dokumentation mit zahlreichen Visualisierungen erlaubt einen vollständigen Einblick in alle Schritte.

---

## ⚙️ Lokale Ausführung

1. Repository klonen:
   ```bash
   git clone https://github.com/Rinovative/cmapss-rul-prediction.git
   cd cmapss-rul-prediction
   ```

2. Abhängigkeiten installieren:
   ```bash
   poetry install
   ```

3. Notebook starten:
   ```bash
   poetry run jupyter notebook
   ```

4. Notebook öffnen:  
   `ML_End2End_Projekt_Rino_Albertin_Predictive_Maintenance.ipynb`

---

## 📂 Projektstruktur

```bash
.
├── .github/              # GitHub Actions (CI)
├── cache/                # Zwischengespeicherte Ergebnisse (Plots, Clusterlabels etc.)
│   ├── fd001/                  
│   ├── fd002/                  
│   ├── fd003/                  
│   ├── fd004/                  
│   └── fdall/
├── data/                 # Originaldaten
│   └── raw/              # Unverarbeitete C-MAPSS-Dateien
├── images/               # Grafiken (z. B. Triebwerksdiagramm)
├── src/                          # Python-Code (EDA, Utilities, Clustering, Linting)
│   ├── eda/                      # Explorative Datenanalyse – modular gegliedert
│   │   ├── eda_clustering.py     # t-SNE & DBSCAN Clusteranalyse
│   │   ├── eda_life.py           # Lebensdaueranalyse & Verteilungen
│   │   ├── eda_opsetting.py      # Analyse der Operation Settings
│   │   ├── eda_sensors.py        # Sensoranalysen, Korrelation, Verläufe
│   │   └── __init__.py
│   │
│   ├── util/                     # Hilfsfunktionen & modulübergreifende Tools
│   │   ├── nb_util.py            # Jupyter-Hilfsfunktionen (z. B. Widgets, Plots)
│   │   ├── __init__.py
│   │
│   └── util/poetry/              # Projekt-Setup & Linting-Logik (CI/CD)
│       ├── poetry_lint.py        # Linting-Konfiguration für GitHub Actions
│       └── __init__.py
│
├── .gitignore            # Ausschlussregeln für Git
├── LICENSE               # Lizenzdatei (MIT License)
├── ML_End2End_Projekt_*.ipynb  # Haupt-Notebook
├── poetry.lock           # Fixierte Abhängigkeiten für Reproduzierbarkeit
├── pyproject.toml        # Poetry-Projektdefinition
├── README.md             # Projektübersicht (diese Datei)
├── requirements.txt      # Alternativ zu Poetry – nötig für Binder
└── runtime.txt           # Binder-konforme Python-Version
```

---

## 📄 Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).

---

## 📚 Quellen

- NASA C-MAPSS-Datensatz:  
  [data.nasa.gov](https://data.nasa.gov/d/ff5v-kuh6)  
  [Kaggle Mirror](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

- Lehrunterlagen „Machine Learning“ – OST – Ostschweizer Fachhochschule