# Experiment Notebooks - Usage Guide

Este directorio contiene notebooks de Jupyter para ejecutar experimentos con diferentes algoritmos de optimización y visualizar los resultados.

## 📋 Lista de Notebooks

### Experimentos Individuales (con Multithreading)
1. **01_nsga2_experiments.ipynb** - Experimentos NSGA-II
2. **02_ga_experiments.ipynb** - Experimentos GA (Genetic Algorithm)
3. **03_mulambda_experiments.ipynb** - Experimentos μ+λ Evolution Strategy
4. **04_sa_experiments.ipynb** - Experimentos Simulated Annealing

### Visualización
5. **05_visualization.ipynb** - Visualización completa de todos los resultados

## 🚀 Cómo Usar

### Paso 1: Ejecutar Experimentos

Cada notebook de experimentos (01-04) se puede ejecutar **independientemente** y en el **orden que prefieras**. Cada uno:

- ✅ Ejecuta múltiples configuraciones de parámetros en paralelo (multithreading)
- ✅ Guarda resultados en su propia base de datos SQLite
- ✅ Realiza análisis estadístico automático
- ✅ Genera resúmenes en CSV

**Importante:** 
- Puedes ejecutar las celdas una por una sin ejecutar todo el notebook de golpe
- Los experimentos se guardan automáticamente en la base de datos
- Puedes detener y continuar cuando quieras

### Paso 2: Visualizar Resultados

Después de ejecutar uno o más notebooks de experimentos, ejecuta el notebook de visualización:

```
05_visualization.ipynb
```

Este notebook:
- 📊 Carga datos de todas las bases de datos disponibles
- 📈 Genera gráficos individuales para cada algoritmo
- 📦 Crea box plots comparativos
- 🔬 Realiza pruebas estadísticas (ANOVA, Kruskal-Wallis, Mann-Whitney)
- 💾 Exporta figuras en alta resolución

## 📊 Bases de Datos Generadas

Cada algoritmo guarda sus resultados en una base de datos SQLite separada:

- `nsga2_experiments.db` - Resultados NSGA-II
- `ga_experiments.db` - Resultados GA
- `mulambda_experiments.db` - Resultados μ+λ
- `sa_experiments.db` - Resultados SA

## 📁 Archivos CSV Generados

Cada notebook genera archivos CSV con resúmenes estadísticos:

- `nsga2_statistical_summary.csv` - Resumen estadístico NSGA-II
- `nsga2_all_results.csv` - Todos los resultados NSGA-II
- `ga_statistical_summary.csv` - Resumen estadístico GA
- `ga_all_results.csv` - Todos los resultados GA
- `mulambda_statistical_summary.csv` - Resumen estadístico μ+λ
- `mulambda_all_results.csv` - Todos los resultados μ+λ
- `sa_statistical_summary.csv` - Resumen estadístico SA
- `sa_all_results.csv` - Todos los resultados SA
- `algorithms_comparison.csv` - Comparación entre todos los algoritmos
- `all_algorithms_combined.csv` - Todos los datos combinados

## 🖼️ Figuras Generadas

El notebook de visualización genera las siguientes figuras:

- `nsga2_distributions.png` - Distribuciones NSGA-II
- `ga_distributions.png` - Distribuciones GA
- `mulambda_distributions.png` - Distribuciones μ+λ
- `sa_distributions.png` - Distribuciones SA
- `combined_boxplots.png` - **Box plots comparativos de todos los algoritmos**
- `combined_violinplots.png` - Violin plots comparativos
- `performance_vs_time.png` - Rendimiento vs tiempo de cómputo

## ⚙️ Configuración de Parámetros

Cada notebook tiene una celda para configurar la grid de parámetros y el número de seeds. Puedes modificarlos según tus necesidades:

### NSGA-II
```python
param_grid = {
    'pop_size': [100, 120],
    'ngen': [500, 800, 1000],
    'cxpb': [0.6, 0.7, 0.8],
    'mutpb': [0.2, 0.3]
}
SEEDS = [0, 1, 2, 42, 123]
```

### GA
```python
param_grid = {
    'pop_size': [80, 120, 150],
    'ngen': [400, 600, 800],
    'cxpb': [0.5, 0.7, 0.8],
    'mutpb': [0.1, 0.2, 0.3]
}
SEEDS = [0, 1, 2, 42, 123]
```

### μ+λ
```python
param_grid = {
    'mu': [50, 80, 120],
    'lambda_': [50, 80, 120],
    'ngen': [400, 600, 800],
    'cxpb': [0.5, 0.7, 0.8]
}
SEEDS = [0, 1, 2, 42, 123]
```

### SA
```python
param_grid = {
    'n_iter': [5000, 8000, 12000],
    'start_temp': [5, 10, 20],
    'end_temp': [0.01]
}
SEEDS = [0, 1, 2, 42, 123]
```

## 🔧 Multithreading

Todos los notebooks usan multiprocessing para ejecutar experimentos en paralelo:

```python
N_JOBS = min(8, mp.cpu_count())  # Ajusta según tu CPU
```

## 📝 Notas Importantes

1. **Tiempo de Ejecución**: Dependiendo de la configuración y tu CPU, cada notebook puede tardar desde minutos hasta horas
2. **Memoria**: Los experimentos pueden consumir bastante memoria RAM si ejecutas muchas configuraciones
3. **Progreso**: Usa `tqdm` para ver barras de progreso en tiempo real
4. **Interrupción**: Puedes detener la ejecución en cualquier momento; los resultados ya guardados permanecerán en la base de datos

## 🎯 Ejemplo de Flujo de Trabajo

```bash
# 1. Ejecutar experimentos (puedes ejecutarlos en cualquier orden)
# Abre 01_nsga2_experiments.ipynb y ejecuta las celdas
# Abre 02_ga_experiments.ipynb y ejecuta las celdas
# ... etc

# 2. Visualizar resultados
# Abre 05_visualization.ipynb y ejecuta las celdas

# 3. Analizar los CSV y figuras generadas
```

## ✅ Archivos Eliminados

Los siguientes archivos fueron eliminados según tu solicitud:

- ❌ `run_validation.py` - **ELIMINADO**
- ❌ `statistical_tests.py` - **ELIMINADO**

---

**Creado por**: Sistema de experimentos automatizado
**Fecha**: Diciembre 2025
