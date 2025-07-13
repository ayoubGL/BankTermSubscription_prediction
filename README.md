
# End-to-End Machine Learning Project: Bank Marketing Term Deposit Prediction

This repository presents a comprehensive, real-world machine learning project focused on predicting whether a client will subscribe to a term deposit based on direct marketing campaign data. It demonstrates a full ML lifecycle, from raw data acquisition and meticulous preprocessing to training and comparing a diverse set of machine learning algorithms, including classical models and a deep learning neural network.

A core aspect of this project is the robust experiment tracking and hyperparameter tuning facilitated by **Weights & Biases (W&B)**, enabling systematic model comparison and selection.

## 🌟 Features

### Real-world Dataset
Utilizes the UCI Bank Marketing Dataset, which contains a mix of numerical and categorical features, requiring extensive preprocessing.

### Comprehensive Data Pipeline

- **Data Acquisition & Cleaning:** Handles raw data, identifies and addresses inconsistencies, missing values (represented as `'unknown'`), and critical data leakage (e.g., `duration` column).
- **Feature Engineering:** Creates new features (e.g., `pdays_not_contacted`) to capture specific domain knowledge.

### Preprocessing
Implements a `scikit-learn` `ColumnTransformer` and `Pipeline` for consistent scaling of numerical features (`StandardScaler` or `MinMaxScaler`) and one-hot encoding of categorical features.

- **Class Imbalance Handling:**
  - Uses **SMOTE** (Synthetic Minority Over-sampling Technique) for classical models.
  - Applies **class weighting** in the loss function for the deep learning model.

### Diverse Model Comparison

- **Classical Models:**
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest Classifier

- **Deep Learning:**
  - Custom Feed-Forward Neural Network built with **PyTorch**

### Robust Model Evaluation

- Calculates and logs:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

- Visualizations:
  - Confusion Matrices
  - ROC Curves

### Experiment Tracking & Hyperparameter Tuning with W&B

- Logs all model parameters, metrics, and artifacts.
- Enables easy comparison of different model runs.
- Supports future automated hyperparameter optimization using **W&B Sweeps**.

### Configuration Management

- All global settings and model hyperparameters are managed in a YAML config file (`configs/model_configs.yaml`).

### Clean Code Organization

- Modular structure for reusability and maintainability.

---

## 🚀 Getting Started

These instructions will get you a copy of the project up and running locally.

### Prerequisites

- Python 3.9+
- `pip` (Python package installer)

### Installation and Setup

1. **Clone the Repository:**

```bash
git clone https://github.com/YOUR_USERNAME/ml_comparison_project.git
cd ml_comparison_project
```

2. **Acquire the Dataset:**

- Download `bank-additional.zip` from the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) page.
- Unzip and move `bank-additional-full.csv` into the `data/` directory.

```
ml_comparison_project/
├── data/
│   └── bank-additional-full.csv
└── ...
```

3. **Install Python Dependencies:**

```bash
pip install -r requirements.txt
```

4. **W&B Setup:**

- Create a W&B account at [wandb.ai](https://wandb.ai)
- Log in:

```bash
wandb login
```

- (Optional) Set your W&B entity:

```bash
export WANDB_ENTITY="your-wandb-username"
# On Windows:
# set WANDB_ENTITY="your-wandb-username"
```

---

## ▶️ Running the ML Pipeline

To run the complete pipeline:

```bash
python main.py
```

This will:

- Load configuration from `configs/model_configs.yaml`
- Preprocess the data (cleaning, feature engineering, scaling, encoding, SMOTE)
- Train and evaluate:
  - Logistic Regression
  - SVM
  - Random Forest
  - PyTorch Neural Network
- Log experiments to **Weights & Biases**

---

## 📊 Analyzing Results

View results in your [W&B Dashboard](https://wandb.ai/YOUR_WANDB_ENTITY/Bank-Marketing-ML-Comparison/runs)  
*(Replace `YOUR_WANDB_ENTITY` with your actual username/team name)*

### In the W&B UI:

- **Compare Runs:** View side-by-side metrics across experiments.
- **Visualize Metrics:** Track F1, ROC-AUC, Precision, Recall, etc.
- **Inspect Artifacts:** Confusion matrices, ROC curves, etc.
- **Review Hyperparameters:** View exact settings for each run.

---

## 🧠 Model Comparison and Recommendation

Consider performance and trade-offs:

- **Logistic Regression:**
  - Simple, fast, and interpretable.
- **Random Forest:**
  - Handles non-linear patterns, offers good performance and feature importances.
- **SVM:**
  - Can yield high accuracy but is less scalable and interpretable.
- **Deep Learning:**
  - Captures complex patterns, but requires more data and compute power.

**Choose based on your business needs**: balance between interpretability, accuracy, speed, and infrastructure.

---

## ⚙️ Project Structure

```
ml_comparison_project/
├── data/                     # Raw dataset
├── notebooks/                # EDA & cleaning
│   └── 01_eda_data_cleaning.ipynb
├── src/                      # Source code
│   ├── data_pipeline.py
│   ├── classical_models.py
│   ├── deep_learning_model.py
│   ├── evaluation.py
│   ├── wandb_utils.py
│   └── __init__.py
├── configs/
│   └── model_configs.yaml    # Hyperparameters
├── main.py                   # Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **scikit-learn**: Classical ML and preprocessing
- **imbalanced-learn**: SMOTE
- **PyTorch**: Deep learning
- **Weights & Biases (W&B)**: Tracking & visualization
- **PyYAML**: Config management
- **Matplotlib & Seaborn**: Visualization
- **Jupyter**: EDA notebooks

---

## 🤝 Contributing

Feel free to fork, raise issues, or submit pull requests. All contributions are welcome!

---

## 📄 License

This project is licensed under the **MIT License**.  
Create a `LICENSE` file with the [MIT License text](https://opensource.org/licenses/MIT).


