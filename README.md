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

Ziel dieses Projekts ist die zuverlässige Vorhersage der verbleibenden Nutzungsdauer (Remaining Useful Life, RUL) von Flugtriebwerken anhand des NASA C-MAPSS-Datensatzes.  
Dazu wird ein generalisierbares Machine-Learning-Modell entwickelt, das frühzeitig auf Wartungsbedarf hinweist.

<details>
<summary>Übersicht der Projektinhalte anzeige</summary>

- **EDA**  
  - Mit **interaktiven Widgets** lassen sich für jeden Datensatz die vier Analyse-Sektionen (Übersicht, Settings, Sensoren, Cluster) ein- und ausklappen.  
  - **op_settings** werden über Lebensdauerkennzahlen, Verteilungsplots und Heatmaps analysiert.  
  - **Sensoren** werden mittels Zeitreihen-Overlays, Box-/Violin-Plots und trend-normalisierten Kurven untersucht.  
  - **Cluster** werden über t-SNE/DBSCAN erkannt, Sankey-Übergänge visualisiert und Eintrittszeitpunkte ausgewertet.  

- **Feature Engineering**  
  - Durch **kombinierte Features** wie Druck-Temperatur-Verhältnisse und Drehzahl-Differenzen werden komplexe Zusammenhänge abgebildet.  
  - **Teillauf-Simulation**: Bei 25 % der Einheiten wird zufällig 0 – 25 % der letzten Zyklen gekürzt, um reale unvollständige Laufzeiten zu simulieren.  
  - Zur **Normierung & Skalierung** wird die Zykluszeit auf [0, 1] normiert und pro op_cond-Gruppe z-standardisiert.  
  - Die **Zustandserkennung per Clusteranalyse** erfolgt über t-SNE/DBSCAN; anschließend wird ein Random-Forest-Klassifikator auf den Clusterlabels trainiert und diese Labels im Testset als Feature eingebunden.  
  - In **temporalen Fenstern** (± 0.25 um 25 %, 50 % und 75 % der normierten Laufzeit) werden Slope, R², Range, Mean-Diff, etc. extrahiert.  
  - Die **Feature Selection** wird mittels Random-Forest-Importance durchgeführt, um unwichtige Merkmale zu eliminieren.  

- **Modellierung & Optimierung**  
  - Verschiedene **Modelle** (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVR, KNN, Decision Tree) werden verglichen.  
  - Durch **gewichtete Samples** mit exponentiell abnehmenden Gewichten bei niedriger RUL wird die Sensitivität auf kritische Phasen erhöht.  
  - Das **Hyperparameter-Tuning** erfolgt zweistufig auf Basis des NASA-Scores: zunächst Grobes GridSearchCV, anschliessend RandomizedSearchCV.  

- **Evaluation & Interpretation**  
  - Mit den **Metriken** RMSE, R² und asymmetrischem NASA-Score wird die Modellgüte bewertet.  
  - **Residuen- und Vorhersageplots** werden zur Identifikation systematischer Abweichungen verwendet.  
  - **XAI**-Methoden (SHAP Beeswarm/Waterfall, Partial Dependence Plots, Gini-Importance) werden eingesetzt, um globale und lokale Feature-Effekte zu erklären.  

- **Generalmodell**  
  - Durch die **Zusammenführung von FD001–FD004** wird ein umfassendes Trainingsset geschaffen.  
  - Die **Pipeline-Anpassung** wird für heterogene op_cond-Gruppen optimiert.  
  - Die **besten Modelle und Hyperparameter** aus den Einzelszenarien werden ins Generalmodell übernommen.  

Eine interaktive Notebook-Dokumentation mit zahlreichen Visualisierungen erlaubt einen vollständigen Einblick in alle Schritte.
</details>

---

## ⚙️ Lokale Ausführung
<details>
<summary>Installationsanleitung anzeigen</summary>

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
</details>

---

## 📂 Projektstruktur
<details>
<summary><strong>Projektstruktur anzeigen</strong></summary>

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
</details>

---

## 📄 Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).

---

## 📚 Quellen

- **C-MAPSS-Datensatz** – NASA Prognostics Center of Excellence (Datensatz Nr. 6 aus dem offiziellen NASA Prognostics Data Repository):  
  A. Saxena and K. Goebel (2008). *Turbofan Engine Degradation Simulation Data Set*, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.  
  [Offizielle Beschreibung auf nasa.gov](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)  
  [Direkter Download (ZIP)](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)

> Die C-MAPSS-Daten wurden aus dem Original-Download (NASA ZIP-Datei) extrahiert und liegen im Projekt als `.txt`-Dateien im Ordner `data/raw/` vor. Sie wurden **nicht verändert**, um Reproduzierbarkeit zu gewährleisten.

- **Lehrunterlagen** „Machine Learning“ – OST – Ostschweizer Fachhochschule

### 📖 Literaturverzeichnis

1. [*A Machine Learning Approach for Turbofan Jet Engine Predictive Maintenance*](https://doi.org/10.1016/j.procs.2025.03.317)  
   Gupta, R. K.; Nakum, J.; Gupta, P.  
   _Procedia Computer Science_, **259**, 2025, S. 161–171.

2. [*Remaining Useful Life Prediction of Aircraft Turbofan Engine Based on Random Forest Feature Selection and Multi-Layer Perceptron*](https://doi.org/10.3390/app13127186)  
   Wang, H.; Li, D.; Li, D.; Liu, C.; Yang, X.; Zhu, G.  
   _Applied Sciences_, **13**, 2023, Art. 7186.

3. [*A Deep Learning Model for Remaining Useful Life Prediction of Aircraft Turbofan Engine on C-MAPSS Dataset*](https://doi.org/10.1109/ACCESS.2022.3203406)  
   Asif, O.; Haider, S. A.; Naqvi, S. R.; Zaki, J. F. W.; Kwak, K.-S.; Islam, S. M. R.  
   _IEEE Access_, **10**, 2022, S. 95425–95440.

4. [*Remaining Useful Life Prediction for Aircraft Engines using LSTM*](https://arxiv.org/abs/2401.07590)  
   Peringal, A.; Mohiuddin, M. B.; Hassan, A.  
   _arXiv Preprint_, 2024.


---
> **Hinweis:**  
> *Für sprachliche Überarbeitung und Unterstützung bei Codefragmenten wurde das KI-Tool **ChatGPT (GPT-4o)** von OpenAI verwendet.*  
> *Die fachliche und inhaltliche Verantwortung liegt vollständig beim Autor.*