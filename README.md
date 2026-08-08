# Machine Learning Prediction: Telecom Customer Churn
Machine learning pipeline built to predict customer churn for a telecom company. The pipeline includes data preprocessing, feature engineering, model training, and evaluation. It handles imbalanced class distribution and implements leak-free out-of-fold target encoding. The visualizations and results inform the business strategy conclusion to address customer churn and improve customer retention.

Reference: [Kaggle -Telecom Churn Prediction by Baligh Mnassri](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)

## Technologies Used
- Python 3.12
- Jupyter Notebook
- Machine Learning Libraries: `scikit-learn`, `LightGBM`
- Data Processing Libraries: `pandas`, `numpy`
- Visualization Libraries: `matplotlib`, `seaborn`

## Model Results
The model addresses a class imbalance of 85% non-churn and 15% churn. This imbalance informed the choice of evaluation metrics, focusing on **PR AUC** and **ROC AUC** scores. The model used is a stacked ensemble of LightGBM and Logistic Regression, where the LGBM's tree-based algorithm and native type handling of the dataset's data structures are leveraged to improve the model's performance. 

| Model | ROC-AUC | **PR-AUC** |
|-------|--------|---------|
| LightGBM | 0.9124 | **0.8621** |
| Logistic Regression | 0.8146 | **0.4154** |
| Stacked Ensemble | 0.9158 | **0.8626** |

## Project Structure

```text
+---data
|   +---processed
|   |       X_test.csv
|   |       X_train.csv
|   |       y_test.csv
|   |       y_train.csv
|   |       
|   \---raw
|           churn-bigml-20.csv
|           churn-bigml-80.csv
|           
+---notebooks
|       01_eda.ipynb
|       02_feature_engineering.ipynb
|       03_modeling.ipynb
|       
\---output
        confusion_matrix.png
        feature_importance_gain.png
        lr_coefficients.png
        performance_curves.png
```

## Project Setup

1. Clone the repository
   ```bash
   git clone https://github.com/rkdtayamora/ml-prediction-telecom-churn.git
   cd ml-prediction-telecom-churn
   ```

2. Install dependencies
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Run the notebooks in order:
   - `01_eda.ipynb` for exploratory data analysis
   - `02_feature_engineering.ipynb` for feature engineering
   - `03_modeling.ipynb` for model training and evaluation