# Heart Disease Risk Prediction

A machine learning project that predicts 10-year coronary heart disease (CHD) risk using demographic, lifestyle, and clinical health indicators from the Framingham Heart Study dataset.

This project started as a statistical learning assignment and was cleaned into a portfolio-ready data science project with clearer structure, reproducible code, and professional documentation.

## Project Objective

The goal is to compare multiple classification models and evaluate how well they predict whether a patient is at risk of developing coronary heart disease within 10 years.

Target variable:

- `TenYearCHD`
  - `0` = No CHD within 10 years
  - `1` = CHD within 10 years

## Dataset

The dataset includes 4,238 patient records and 16 variables.

Key features include:

- Age
- Gender
- Education level
- Smoking status
- Cigarettes per day
- Blood pressure medication usage
- Stroke history
- Hypertension history
- Diabetes status
- Total cholesterol
- Systolic blood pressure
- Diastolic blood pressure
- BMI
- Heart rate
- Glucose level

## Methods

The analysis includes:

- Data loading and inspection
- Missing value analysis
- Median imputation for missing numeric values
- Exploratory data analysis
- Feature/target split
- Stratified cross-validation
- Model comparison using classification metrics

Models compared:

- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine

## Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall / Sensitivity
- F1 Score
- ROC-AUC

Because the dataset is imbalanced, accuracy alone is not enough. Recall, F1 score, and ROC-AUC are more useful for understanding whether the model can identify high-risk CHD cases.

## Repository Structure

```text
heart-disease-risk-prediction/
├── data/
│   └── framingham.csv
├── notebooks/
│   └── framingham_analysis.ipynb
├── reports/
│   └── project_summary.md
├── src/
│   └── train_models.py
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-risk-prediction.git
cd heart-disease-risk-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the model training script:

```bash
python src/train_models.py
```

## Key Takeaways

- Clinical and lifestyle indicators can be used to estimate long-term heart disease risk.
- Logistic Regression provides a strong baseline because it is interpretable and performs competitively.
- More complex machine learning models do not always outperform simpler statistical models.
- For healthcare prediction problems, recall and ROC-AUC should be considered alongside accuracy.

## Skills Demonstrated

- Python
- Pandas
- NumPy
- Scikit-learn
- Data Cleaning
- Exploratory Data Analysis
- Classification Modeling
- Cross-Validation
- Model Evaluation
- Healthcare Analytics

## Resume Bullet

Developed machine learning models to predict 10-year coronary heart disease risk using demographic, lifestyle, and clinical indicators from the Framingham Heart Study dataset; compared Logistic Regression, Naive Bayes, Decision Tree, Random Forest, and SVM using cross-validation and ROC-AUC.
