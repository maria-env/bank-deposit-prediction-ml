# Bank Product Subscription Prediction
 
Machine Learning Project 1 (Universidad Carlos III de Madrid). The goal is to predict whether a client will subscribe to a term deposit based on data from a marketing campaign.
 
## What's included
 
- **Exploratory data analysis (EDA)** of categorical, numerical and target variables, with feature engineering on `pdays`.
- **Preprocessing pipelines** with `ColumnTransformer` (imputation, scaling and one-hot encoding) and nested cross-validation.
- **Models compared:** KNN, decision trees, linear models, SVM, Gradient Boosting, Random Forest, Voting and Stacking, with hyperparameter optimization.
- **Result:** the best model (Stacking) reaches 0.8646 accuracy on validation and 0.8548 on test.
- **Final model** saved (`modelo_final.joblib`) and a **Streamlit app** to enter a client's data and get the prediction.
## Files
 
| File | Contents |
|---|---|
| `C1.ipynb` | Main notebook: EDA, models, evaluation and selection |
| `C2.ipynb` | Training and export of the final model |
| `modelo_final.joblib` | Trained model |
| `mystreamlit.py` | Prediction web app |
| `predicciones.csv` | Predictions on specific instances |
 
## Technologies
 
Python · pandas · NumPy · scikit-learn 1.6.1 · Streamlit · Jupyter
 
## How to run
 
```bash
pip install -r requirements.txt
streamlit run mystreamlit.py
```
 
The notebooks can be opened with Jupyter or Google Colab.
 
## Authors

Team project by María Arias Rodríguez and Jorge Ignacio Castañeda Vallenilla.
