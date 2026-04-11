# Heart Disease Detection System

A complete heart disease detection project built for an Expert Systems assignment. This repository combines:

- a rule-based expert system using `Experta`
- a machine learning model using `DecisionTreeClassifier`
- data preprocessing and exploratory analysis
- a Streamlit web interface for interactive prediction

The system uses patient clinical attributes to estimate heart disease risk and lets you compare a transparent rule-based decision with a trained ML model.

## Features

- Heart disease dataset preprocessing with Pandas
- Missing-value handling and duplicate removal
- Numerical feature scaling with `MinMaxScaler`
- Categorical feature encoding with `OneHotEncoder`
- Correlation analysis and visualization
- Rule-based inference engine with more than 10 rules
- Decision tree training with hyperparameter tuning
- Evaluation using accuracy, precision, recall, and F1-score
- Comparison report between expert-system and ML performance
- Streamlit dashboard for interactive use

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Experta
- Streamlit

## Dataset

This project uses the provided heart disease dataset with the following columns:

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`

Files:

- [data/raw_data.csv](data/raw_data.csv)
- [data/cleaned_data.csv](data/cleaned_data.csv)

## Project Structure

```text
Heart_Disease_Detection/
│── data/
│   ├── raw_data.csv
│   └── cleaned_data.csv
│── notebooks/
│   ├── data_analysis.ipynb
│   └── model_training.ipynb
│── rule_based_system/
│   ├── rules.py
│   └── expert_system.py
│── ml_model/
│   ├── train_model.py
│   ├── predict.py
│   └── heart_disease_decision_tree.joblib
│── utils/
│   └── data_processing.py
│── reports/
│   ├── accuracy_comparison.md
│   └── figures/
│── ui/
│   └── app.py
│── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/heart-disease-detection-system.git
cd heart-disease-detection-system
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the environment:

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running The Project

### 1. Train the model

```bash
python ml_model/train_model.py
```

This step:

- preprocesses the dataset
- saves the cleaned dataset
- trains the decision tree model
- performs hyperparameter tuning
- saves the trained model
- generates comparison outputs

### 2. Launch the Streamlit app

```bash
python -m streamlit run ui/app.py
```

Then open the local URL shown in the terminal, usually:

`http://localhost:8501`

## Rule-Based Expert System

The expert system uses clinical-style rules to estimate patient risk.

Examples of rule logic:

- If `chol > 240` and `age > 50`, risk increases
- If `trestbps >= 140` and `exang = 1`, risk increases
- If `thalach >= 150` and `exang = 0`, risk decreases

This approach improves explainability because the triggered rules are shown directly to the user.

## Machine Learning Model

The machine learning component uses:

- `DecisionTreeClassifier`
- `GridSearchCV` for tuning
- `train_test_split` with an 80/20 split

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score

## Notebooks

The notebooks folder includes:

- [data_analysis.ipynb](notebooks/data_analysis.ipynb) for preprocessing and visualization
- [model_training.ipynb](notebooks/model_training.ipynb) for training and evaluation

## Reports

Generated outputs and reports:

- [reports/accuracy_comparison.md](reports/accuracy_comparison.md)
- feature-importance charts in `reports/figures/`

## Streamlit Interface

The UI lets the user:

- enter patient information manually
- get a decision tree prediction
- get a rule-based risk explanation
- explore basic charts and the dataset correlation heatmap

## Future Improvements

- Try additional machine learning models such as Random Forest or XGBoost
- Add better clinical validation for expert rules
- Deploy the Streamlit app online
- Add model explainability with SHAP or feature contribution charts

## Author

Update this section with your name, university, course, and GitHub profile.

Example:

`Developed by Your Name`

## License

This project is for educational and academic use.
