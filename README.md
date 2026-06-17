# MovieLens: Polars vs Pandas para Análisis y Predicción de Preferencias de Películas

Proyecto desarrollado para la **Tarea 3** del curso **Computación Paralela y Distribuida**.

---

# Descripción del Problema

El objetivo de este proyecto es construir un pipeline completo de análisis de datos y aprendizaje automático utilizando **Polars**, comparándolo posteriormente con una implementación equivalente en **Pandas** para evaluar el rendimiento de ambas bibliotecas en tareas comunes de procesamiento de datos.

Como caso de estudio se utiliza el dataset **MovieLens Latest Small**, con el fin de predecir si un usuario otorgará una valoración positiva a una película.

La variable objetivo se define como:

```python
liked_movie = (rating >= 4.0)
```

donde:

* **1:** la película recibió una valoración positiva.
* **0:** la película no recibió una valoración positiva.

El proyecto incluye:

* Análisis Exploratorio de Datos (EDA).
* Ingeniería de Características.
* Entrenamiento de modelos de Machine Learning.
* Comparación de rendimiento entre Polars y Pandas.
* Experimentos de escalabilidad.
* Comparación entre ejecución *eager* y *lazy* en Polars.

---

# Dataset

Se utiliza el conjunto de datos **MovieLens Latest Small**, ampliamente utilizado en investigación sobre sistemas de recomendación.

Archivos utilizados:

* `ratings.csv`
* `movies.csv`

Características principales:

* Más de 100 000 registros.
* Variables numéricas y categóricas.
* Estructura relacional que permite realizar operaciones `JOIN`.
* Adecuado para tareas de clasificación y análisis exploratorio.

---

# Fuente del Dataset

MovieLens Dataset

https://grouplens.org/datasets/movielens/

---

# Tecnologías Utilizadas

* Python 3.11+
* Polars
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Psutil
* Jupyter Notebook

---

# Instalación

Clonar el repositorio:

```bash
git clone <repository-url>
cd <repository-folder>
```

Crear un entorno virtual (opcional pero recomendado):

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

---

# Ejecución

1. Descargar el dataset MovieLens.
2. Copiar `ratings.csv` y `movies.csv` dentro de:

```text
data/raw/
```

3. Abrir el notebook:

```text
notebooks/analysis.ipynb
```

4. Ejecutar las celdas en orden.

El notebook realiza automáticamente:

* Análisis Exploratorio de Datos.
* Ejecución del pipeline con Polars.
* Entrenamiento y evaluación de modelos.
* Ejecución del pipeline equivalente con Pandas.
* Benchmark entre ambas implementaciones.
* Experimentos de escalabilidad.
* Comparación entre ejecución *eager* y *lazy*.

---

# Estructura del Proyecto

```text
project/

data/
├── raw/
│   ├── ratings.csv
│   └── movies.csv
└── processed/

notebooks/
└── analysis.ipynb

src/
├── preprocessing.py
├── feature_engineering.py
├── polars_pipeline.py
├── pandas_pipeline.py
└── train_models.py

results/
├── benchmark_results.csv
├── model_results.csv
├── scalability_results.csv
└── lazy_execution_results.csv

figures/

report/

README.md
requirements.txt
```

---

# Arquitectura del Proyecto

El proyecto está organizado de forma modular para separar claramente las responsabilidades de cada componente.

## `analysis.ipynb`

Notebook principal que presenta el flujo completo del proyecto:

* Análisis Exploratorio de Datos (EDA).
* Ejecución del pipeline en Polars.
* Entrenamiento y evaluación de modelos.
* Ejecución del pipeline equivalente en Pandas.
* Benchmark entre Polars y Pandas.
* Experimentos de escalabilidad.
* Comparación entre ejecución *eager* y *lazy*.

## `preprocessing.py`

Responsable de:

* Carga de los datasets.
* Validación de la estructura de los datos.
* Funciones auxiliares para obtener información del dataset.

## `polars_pipeline.py`

Implementa el pipeline completo utilizando Polars:

* Lectura de datos.
* JOIN entre tablas.
* Filtrado de registros.
* Transformaciones.
* Ingeniería de características.
* Medición de tiempos de ejecución.

## `pandas_pipeline.py`

Implementa exactamente el mismo pipeline utilizando Pandas para permitir una comparación justa de rendimiento.

## `feature_engineering.py`

Contiene funciones auxiliares y constantes compartidas relacionadas con la ingeniería de características.

## `train_models.py`

Entrena y evalúa los modelos de Machine Learning utilizando Scikit-Learn.

Este módulo acepta tanto DataFrames de Polars como de Pandas. Cuando recibe un DataFrame de Polars, realiza internamente la conversión necesaria para reutilizar exactamente el mismo proceso de entrenamiento para ambos pipelines.

---

# Flujo del Proyecto

```text
ratings.csv          movies.csv
      │                   │
      └───────── JOIN ────┘
                  │
                  ▼
        Análisis Exploratorio (Polars)
                  │
                  ▼
      Pipeline Polars (Feature Engineering)
                  │
                  ▼
          DataFrame procesado
                  │
                  ▼
     Entrenamiento de modelos (Scikit-Learn)
                  │
                  ▼
          Resultados de Machine Learning
                  │
                  ▼
      Pipeline equivalente en Pandas
                  │
                  ▼
        Benchmark Polars vs Pandas
                  │
                  ▼
      Escalabilidad y Lazy Execution
```

---

# Resultados

Esta sección será completada una vez finalizados los experimentos.

Se incluirán:

* Resultados de los modelos de Machine Learning.
* Comparación de tiempos entre Polars y Pandas.
* Speedup observado.
* Resultados de escalabilidad.
* Comparación entre ejecución *eager* y *lazy*.
* Conclusiones finales del estudio.
