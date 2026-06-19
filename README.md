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
* Benchmark entre Polars y Pandas.
* Experimentos de escalabilidad.
* Evaluación de Lazy Execution.

---

# Dataset

Se utiliza el conjunto de datos **MovieLens Latest Small**, ampliamente utilizado en investigación sobre sistemas de recomendación.

Archivos utilizados:

* `ratings.csv`
* `movies.csv`

Características principales:

* 100,836 registros de calificaciones.
* 610 usuarios.
* 9,724 películas.
* Variables numéricas y categóricas.
* Estructura relacional que permite operaciones `JOIN`.
* Adecuado para problemas de clasificación.

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

Crear un entorno virtual (opcional):

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

3. Abrir:

```text
notebooks/analysis.ipynb
```

4. Ejecutar todas las celdas en orden.

El notebook ejecuta automáticamente:

* Análisis Exploratorio de Datos utilizando Polars.
* Pipeline completo en Polars.
* Entrenamiento de modelos.
* Pipeline equivalente en Pandas.
* Benchmark entre ambas implementaciones.
* Experimentos de escalabilidad.
* Comparación entre ejecución eager y lazy.
* Análisis de resultados.
* Conclusiones.

---

# Estructura del Proyecto

```text
project/

data/
├── raw/
│   ├── ratings.csv
│   └── movies.csv
├── processed/
└── scalability/

notebooks/
└── analysis.ipynb

src/
├── preprocessing.py
├── feature_engineering.py
├── feature_engineering_pandas.py
├── polars_pipeline.py
├── pandas_pipeline.py
└── train_models.py

figures/

report/
└── informe.pdf

README.md
requirements.txt
```

---

# Arquitectura del Proyecto

## analysis.ipynb

Notebook principal del proyecto donde se desarrolla el flujo completo:

* Carga de datos.
* EDA utilizando Polars.
* Pipeline Polars.
* Entrenamiento de modelos.
* Pipeline Pandas.
* Benchmark.
* Escalabilidad.
* Lazy Execution.
* Análisis de resultados.
* Conclusiones.

---

## preprocessing.py

Contiene las funciones para:

* Carga de datasets.
* Información del sistema.
* Estadísticas generales.
* Tamaño del dataset.
* Valores faltantes.

---

## feature_engineering.py

Implementa la ingeniería de características utilizando Polars:

* JOIN entre datasets.
* Filtrado.
* Limpieza.
* Variable objetivo.
* Estadísticas por usuario.
* Estadísticas por película.
* One-Hot Encoding de géneros.
* Preparación del dataset.

---

## feature_engineering_pandas.py

Implementa exactamente las mismas transformaciones anteriores utilizando Pandas para permitir una comparación justa con Polars.

---

## polars_pipeline.py

Implementa el pipeline completo utilizando Polars:

* Lectura.
* JOIN.
* Filtrado.
* Limpieza.
* Ingeniería de características.
* Train/Test Split.
* Escalado de variables.
* Preparación de matrices para Machine Learning.
* Medición de tiempos de ejecución.

---

## pandas_pipeline.py

Implementa el mismo pipeline utilizando Pandas, permitiendo comparar ambas bibliotecas bajo exactamente las mismas condiciones experimentales.

---

## train_models.py

Entrena y evalúa los modelos de Machine Learning utilizando Scikit-Learn.

Modelos implementados:

* Logistic Regression.
* Random Forest.
* Gradient Boosting.

Para cada modelo se reportan:

* Accuracy.
* F1 Score.
* ROC AUC.
* Tiempo de entrenamiento.
* Matriz de confusión.

---

# Flujo del Proyecto

                           ratings.csv
                               │
                               │
                           movies.csv
                               │
                               ▼
                     Carga de los datos
                               │
                               ▼
                      JOIN de los datasets
                               │
                               ▼
                     Limpieza de datos
                (filtrado y valores nulos)
                               │
                               ▼
                Creación de la variable objetivo
                      liked_movie (rating ≥ 4)
                               │
                               ▼
                      Train / Test Split
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
        Pipeline Polars                  Pipeline Pandas
              │                                 │
              │                                 │
     Feature Engineering              Feature Engineering
     • user statistics               • user statistics
     • movie statistics              • movie statistics
     • one-hot genres                • one-hot genres
              │                                 │
              ▼                                 ▼
      StandardScaler                    StandardScaler
              │                                 │
              ▼                                 ▼
         X_train / X_test                 X_train / X_test
              │                                 │
              ▼                                 ▼
        Machine Learning                  Machine Learning
        (LogReg, RF, GB)                  (LogReg, RF, GB)
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                    Comparación de resultados
                               │
                               ▼
                  Benchmark Polars vs Pandas
                               │
                  • Tiempo del pipeline
                  • Tiempo de entrenamiento
                  • Métricas de clasificación
                  • Escalabilidad
                  • Lazy vs Eager Execution

---

# Resultados

Los experimentos mostraron que:

* Ambos pipelines generan exactamente las mismas características y obtienen el mismo desempeño predictivo.
* Random Forest obtuvo el mejor rendimiento, con una Accuracy cercana al **70 %**.
* Polars presentó menores tiempos de ejecución durante las etapas de preprocesamiento del pipeline.
* El beneficio de Polars se mantuvo conforme aumentó el tamaño del conjunto de datos.
* Lazy Execution permitió optimizar la ejecución del pipeline antes de materializar los resultados mediante `collect()`.

En conjunto, los resultados muestran que Polars constituye una alternativa eficiente para el procesamiento de datos tabulares, especialmente en pipelines con múltiples transformaciones y conjuntos de datos de mayor tamaño.