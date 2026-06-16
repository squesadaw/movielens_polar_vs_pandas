# TTCT0017 - Computación Paralela y Distribuida
# Tarea 3: Polars vs Pandas para Análisis y Predicción de Preferencias de Películas

## Descripción

El objetivo principal de este proyecto es construir un pipeline completo de análisis de datos y aprendizaje automático utilizando la biblioteca Polars, y compararlo sistemáticamente con una implementación equivalente en Pandas.

Se utiliza el dataset MovieLens, un conjunto de datos ampliamente utilizado en sistemas de recomendación que contiene información sobre calificaciones de películas realizadas por usuarios.

El proyecto incluye:

* Análisis exploratorio de datos (EDA).
* Ingeniería de características.
* Entrenamiento de modelos de Machine Learning.
* Comparación de rendimiento entre Polars y Pandas.
* Experimentos de escalabilidad.
* Evaluación de Lazy Execution en Polars.

---

## Objetivo del Problema

Se plantea un problema de clasificación binaria.

Dado un usuario y una película, se busca predecir si la película recibirá una valoración positiva.

La variable objetivo se define como:

```python
liked_movie = (rating >= 4.0)
```

donde:

* 1: la película recibió una valoración positiva.
* 0: la película no recibió una valoración positiva.

---

## Dataset

### MovieLens Latest Small

El dataset contiene información sobre:

* Usuarios.
* Películas.
* Calificaciones.
* Géneros cinematográficos.

Archivos utilizados:

* ratings.csv
* movies.csv

Características principales:

* Más de 100,000 registros.
* Datos relacionales adecuados para operaciones JOIN.
* Variables categóricas y numéricas.
* Problema adecuado para análisis exploratorio y aprendizaje automático.

---

## Tecnologías Utilizadas

* Python 3.11+
* Polars
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Psutil

---

## Estructura del Proyecto

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

figures/

report/

README.md
requirements.txt
```

---

## Pipeline del Proyecto

### 1. Análisis Exploratorio de Datos

Utilizando exclusivamente Polars:

* Estadísticas descriptivas.
* Identificación de valores faltantes.
* Distribuciones de variables.
* Correlaciones relevantes.
* Visualizaciones.

### 2. Ingeniería de Características

El pipeline incluye:

* JOIN entre ratings y movies.
* Transformación de géneros.
* Extracción de variables temporales.
* Agregaciones por usuario.
* Agregaciones por película.
* Creación de nuevas características.
* Construcción de la variable objetivo.

### 3. Machine Learning

Modelos entrenados:

* Logistic Regression
* Random Forest
* Gradient Boosting

Métricas reportadas:

* Accuracy
* F1 Score
* ROC AUC
* Matriz de Confusión
* Tiempo de entrenamiento

### 4. Benchmark Polars vs Pandas

Se comparan:

* Tiempo de lectura.
* Tiempo de JOIN.
* Tiempo de agregación.
* Tiempo de transformación.
* Tiempo de feature engineering.
* Tiempo total del pipeline.

### 5. Escalabilidad

Experimentos utilizando:

* 25% del dataset.
* 50% del dataset.
* 75% del dataset.
* 100% del dataset.

Para cada caso se reporta:

* Tiempo en Pandas.
* Tiempo en Polars.
* Speedup observado.

### 6. Lazy Execution

Comparación entre:

```python
pl.read_csv(...)
```

y

```python
pl.scan_csv(...).collect()
```

analizando:

* Tiempo de ejecución.
* Uso de memoria.
* Complejidad del pipeline.

---

## Instalación

Clonar el repositorio:

```bash
git clone <repository-url>
cd project
```

Crear entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Ejecución

Abrir:

```text
notebooks/analysis.ipynb
```

y ejecutar las celdas en orden.

El notebook genera automáticamente:

* Resultados de Machine Learning.
* Gráficas.
* Benchmarks.
* Resultados de escalabilidad.
* Resultados de Lazy Execution.

---

## Resultados

Esta sección será completada una vez finalizados los experimentos.

Se incluirán:

* Resultados predictivos.
* Comparación Polars vs Pandas.
* Gráficas de speedup.
* Discusión técnica de resultados.
* Conclusiones.
