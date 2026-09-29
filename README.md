# Predicción de contratación de un producto bancario

Práctica 1 de Machine Learning (Universidad Carlos III de Madrid). El objetivo es predecir si un cliente contratará un depósito bancario a partir de los datos de una campaña de marketing.

## Qué incluye

- **Análisis exploratorio (EDA)** de variables categóricas, numéricas y del objetivo, con ingeniería de características sobre `pdays`.
- **Pipelines de preprocesado** con `ColumnTransformer` (imputación, escalado y one-hot encoding) y validación cruzada anidada.
- **Modelos comparados:** KNN, árboles de decisión, modelos lineales, SVM, Gradient Boosting, Random Forest, Voting y Stacking, con optimización de hiperparámetros.
- **Resultado:** el mejor modelo (Stacking) alcanza 0,8646 de accuracy en validación y 0,8548 en test.
- **Modelo final** guardado (`modelo_final.joblib`) y una **aplicación en Streamlit** para introducir los datos de un cliente y obtener la predicción.

## Archivos

| Archivo | Contenido |
|---|---|
| `C1.ipynb` | Cuaderno principal: EDA, modelos, evaluación y selección |
| `C2.ipynb` | Entrenamiento y exportación del modelo final |
| `modelo_final.joblib` | Modelo entrenado |
| `mystreamlit.py` | Aplicación web de predicción |
| `predicciones.csv` | Predicciones sobre instancias específicas |

## Tecnologías

Python · pandas · NumPy · scikit-learn 1.6.1 · Streamlit · Jupyter

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
streamlit run mystreamlit.py
```

Los cuadernos se pueden abrir con Jupyter o con Google Colab.

## Autoría

Trabajo en equipo de María Arias Rodríguez y Jorge Castañeda.
