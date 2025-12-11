# Organización del Directorio Hyperparametrization

Este directorio contiene todos los experimentos de hiperparametrización para los algoritmos evolutivos (NSGA2, GA, Mu+Lambda, SA).

## Estructura de Carpetas

```
hyperparametrization/
├── notebooks/                    # Notebooks de Jupyter para experimentos
│   ├── 01_nsga2_experiments.ipynb
│   ├── 02_ga_experiments.ipynb
│   ├── 03_mulambda_experiments.ipynb
│   ├── 04_sa_experiments.ipynb
│   └── 05_visualization.ipynb
│
├── scripts/                      # Scripts de Python
│   ├── grid_search.py           # Grid search para NSGA2
│   └── grid_search_algos.py     # Grid search para otros algoritmos
│
├── data/                         # Bases de datos SQLite
│   ├── nsga2_experiments.db
│   ├── ga_experiments.db
│   ├── mulambda_experiments.db
│   └── sa_experiments.db
│
├── results/                      # Resultados en formato CSV
│   ├── individual/               # Resultados por algoritmo
│   │   ├── nsga2_all_results.csv
│   │   ├── nsga2_statistical_summary.csv
│   │   ├── ga_all_results.csv
│   │   ├── ga_statistical_summary.csv
│   │   ├── mulambda_all_results.csv
│   │   ├── mulambda_statistical_summary.csv
│   │   ├── sa_all_results.csv
│   │   └── sa_statistical_summary.csv
│   └── combined/                 # Resultados combinados/comparativos
│       ├── all_algorithms_combined.csv
│       └── algorithms_comparison.csv
│
├── plots/                        # Visualizaciones
│   ├── individual/               # Gráficos por algoritmo
│   │   ├── nsga2_distributions.png
│   │   ├── ga_distributions.png
│   │   ├── mulambda_distributions.png
│   │   └── sa_distributions.png
│   └── combined/                 # Gráficos comparativos
│       ├── combined_boxplots.png
│       ├── combined_violinplots.png
│       └── performance_vs_time.png
│
├── validation_report/            # Reportes de validación
│
└── README_NOTEBOOKS.md           # Documentación de los notebooks
```

## Descripción de Contenidos

### Notebooks (`notebooks/`)
- **01-04**: Notebooks individuales para cada algoritmo (NSGA2, GA, Mu+Lambda, SA)
  - Ejecutan experimentos con diferentes configuraciones de hiperparámetros
  - Utilizan multithreading para ejecución eficiente
  - Guardan resultados en bases de datos SQLite individuales
  - Incluyen análisis estadístico y pruebas de normalidad
  - Exportan resultados a CSV

- **05**: Notebook de visualización comparativa
  - Carga resultados de todos los algoritmos
  - Genera gráficos comparativos
  - Produce tablas LaTeX con resultados estadísticos

### Scripts (`scripts/`)
- **grid_search.py**: Grid search para NSGA2
- **grid_search_algos.py**: Grid search para algoritmos single-objective

### Bases de Datos (`data/`)
Contiene las bases de datos SQLite con todos los resultados experimentales de cada algoritmo.

### Resultados (`results/`)
- **individual/**: CSVs con resultados brutos y resúmenes estadísticos por algoritmo
- **combined/**: CSVs con comparaciones entre todos los algoritmos

### Plots (`plots/`)
- **individual/**: Gráficos de distribución para cada algoritmo
- **combined/**: Gráficos comparativos (boxplots, violin plots, performance vs tiempo)

## Flujo de Trabajo

1. **Ejecutar experimentos**: Correr notebooks `01_nsga2_experiments.ipynb` a `04_sa_experiments.ipynb`
   - Cada notebook guarda en su DB correspondiente en `data/`
   - Exporta CSVs a `results/individual/`
   - Genera plots en `plots/individual/`

2. **Visualización comparativa**: Ejecutar `05_visualization.ipynb`
   - Lee CSVs de `results/individual/`
   - Genera CSVs comparativos en `results/combined/`
   - Crea plots comparativos en `plots/combined/`

3. **Grid search adicional**: Ejecutar scripts en `scripts/` según necesidad
   - Resultados van a `/results/` en el directorio raíz del proyecto

## Notas
- Las bases de datos preservan todos los detalles de cada ejecución
- Los CSVs son para análisis rápido y visualización
- Los plots pueden regenerarse ejecutando los notebooks
