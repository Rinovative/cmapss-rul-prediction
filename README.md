[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Rinovative/cmapss-rul-prediction/HEAD)  
_Interaktives Jupyter Notebook direkt im Browser öffnen (via Binder)_

> **Hinweis:**  
> Für bessere Performance in Binder immer nur ein Widget gleichzeitig offen halten.

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
├── .github/                         # GitHub Actions (CI)
├── cache/                           # Zwischengespeicherte Ergebnisse (Plots, Clusterlabels etc.)
│   ├── fd001/                   
│   ├── fd002/                   
│   ├── fd003/                   
│   ├── fd004/                   
│   └── fdall/
├── data/                            # Originaldaten
├── images/                          # Grafiken (z. B. Triebwerksdiagramm)
├── models/                          # Modellartefakte
│   ├── fd001/
│   ├── fd002/
│   ├── fd003/
│   ├── fd004/
│   └── fdall/
├── src/                             # Quellcode (modular aufgebaut)
│   ├── eda/                         # Explorative Datenanalyse
│   │   ├── __init__.py              
│   │   ├── eda_clustering.py        # Clusteranalyse mit t-SNE & DBSCAN
│   │   ├── eda_life.py              # Lebensdauerverteilung & Analyse
│   │   ├── eda_opsetting.py         # Analyse der Operationsbedingungen (Settings)
│   │   └── eda_sensors.py           # Sensorverläufe, Korrelationen & Klassifikationen
│   │
│   ├── fe/                          # Feature Engineering
│   │   ├── __init__.py              
│   │   ├── feature_selection.py     # Auswahl relevanter Merkmale
│   │   ├── preprocessing.py         # Vorverarbeitungsschritte (z. B. Normalisierung)
│   │   └── temporal_features.py     # Zeitreihenbasierte Features (z. B. rollierende Fenster)
│   │
│   ├── models/                      # Modelltraining & Bewertung
│   │   ├── __init__.py              
│   │   ├── hyperparameter_tuning.py # GridSearch & RandomSearch für Modelloptimierung
│   │   ├── interpretation.py        # Modellinterpretation (z. B. SHAP-Waterfalls)
│   │   ├── models.py                # Modell-Wrapper & Training / Evaluation
│   │   └── plotting.py              # Visualisierung der Modellresultate & Fehlerverteilung
│   │
│   └── util/                        # Hilfsfunktionen & Projektinfrastruktur
│       ├── poetry/                 
│       │   ├── poetry_lint.py        # Linting für GitHub Actions (Black, isort, flake8 etc.)
│       │   └── __init__.py         
│       ├── __init__.py             
│       ├── cache_util.py            # Automatisches Speichern und Laden von Plots und Ergebnissen
│       ├── data_loader.py           # Einlesen und Aufbereiten der C-MAPSS-Daten
│       ├── nb_util.py               # Jupyter-Notebook-Hilfen (Widgets, Interaktivität)
│       └── normalization.py         # Z-Norm & weitere Skalierungsfunktionen
│
├── .gitignore                       # Ausschlussregeln für Git
├── LICENSE                          # Lizenzdatei (MIT License)
├── ML_End2End_Projekt_*.ipynb       # Haupt-Notebook
├── poetry.lock                      # Fixierte Abhängigkeiten
├── pyproject.toml                   # Poetry-Projektdefinition
├── README.md                        # Projektübersicht (diese Datei)
├── requirements.txt                 # Alternativ zu Poetry – nötig für Binder
└── runtime.txt                      # Binder-konforme Python-Version
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

### 📖 Literaturverzeichnis

[1] Gupta, R. K.; Nakum, J.; Gupta, P.: *A Machine Learning Approach for Turbofan Jet Engine Predictive Maintenance*. Procedia Computer Science, 259, 2025, S. 161–171. https://doi.org/10.1016/j.procs.2025.03.317   
[2] Wang, H.; Li, D.; Li, D.; Liu, C.; Yang, X.; Zhu, G.: *Remaining Useful Life Prediction of Aircraft Turbofan Engine Based on Random Forest Feature Selection and Multi-Layer Perceptron*. Applied Sciences, 13, 2023, Art. 7186. https://doi.org/10.3390/app13127186   
[3] Asif, O.; Haider, S. A.; Naqvi, S. R.; Zaki, J. F. W.; Kwak, K.-S.; Islam, S. M. R.: *A Deep Learning Model for Remaining Useful Life Prediction of Aircraft Turbofan Engine on C-MAPSS Dataset*. IEEE Access, 10, 95425–95440, 2022. https://doi.org/10.1109/ACCESS.2022.3203406   
[4] Peringal, A.; Mohiuddin, M. B.; Hassan, A.: *Remaining Useful Life Prediction for Aircraft Engines using LSTM.* Preprint auf arXiv, 2024. https://arxiv.org/abs/2401.07590

---
> **Hinweis:**  
> *Für die sprachliche Überarbeitung und die Unterstützung bei Codefragmenten wurde das KI-Tool* **ChatGPT** *von OpenAI (GPT-4o, https://chatgpt.com) verwendet. Die fachliche und inhaltliche Verantwortung liegt vollständig beim Autor.*