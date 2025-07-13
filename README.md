# End-to-End Machine Learning Project: Bank Marketing Term Deposit Prediction

This repository presents a comprehensive, real-world machine learning project focused on predicting whether a client will subscribe to a term deposit based on direct marketing campaign data. It demonstrates a full ML lifecycle, from raw data acquisition and meticulous preprocessing to training and comparing a diverse set of machine learning algorithms, including classical models and a deep learning neural network.

A core aspect of this project is the robust experiment tracking and hyperparameter tuning facilitated by Weights & Biases (W&B), enabling systematic model comparison and selection.

## 🌟 Features
-- **Real-world Dataset: Utilizes the UCI Bank Marketing Dataset, which contains a mix of numerical and categorical features, requiring extensive preprocessing.
Comprehensive Data Pipeline:**

        -   Data Acquisition & Cleaning: Handles raw data, identifies and addresses inconsistencies, missing values (represented as 'unknown'), and critical data leakage (duration column).

        -   Feature Engineering: Creates new features (e.g., pdays_not_contacted) to capture specific domain knowledge.

### Preprocessing: Implements a scikit-learn ColumnTransformer and Pipeline for consistent scaling of numerical features (StandardScaler or MinMaxScaler) and one-hot encoding of categorical features.

- Class Imbalance Handling: Employs SMOTE (Synthetic Minority Over-sampling Technique) for classical models and class weighting in the loss function for the deep learning model to address the highly imbalanced target variable.

- Diverse Model Comparison: Trains and evaluates a range of algorithms:

        - Classical Machine Learning: Logistic Regression, Support Vector Machine (SVM), Random Forest Classifier.

        - Deep Learning: A custom Feed-Forward Neural Network built with PyTorch.

### Robust Model Evaluation:

- Calculates and logs a wide array of relevant metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

- Generates and logs visualizations: Confusion Matrices and ROC Curves.

- Experiment Tracking & Hyperparameter Tuning with Weights & Biases (W&B):

- Logs all model parameters, epoch-wise and final metrics, and model artifacts (trained pipelines/models) for every experiment run.

- Enables easy comparison of different models and hyperparameter configurations through the W&B dashboard.

### Supports future automated hyperparameter optimization (W&B Sweeps).

- Configuration Management: Utilizes a YAML file (configs/model_configs.yaml) to centralize and manage all model hyperparameters and global project settings, ensuring reproducibility and ease of modification.

- Clean Code Organization: Structured into modular Python scripts for readability, maintainability, and reusability.


## 🚀 Getting Started
These instructions will get you a copy of the project up and running on your local machine.

Prerequisites
Python 3.9+

pip (Python package installer)

Installation and Setup
Clone the repository:

git clone 
cd ml_comparison_project


Acquire the Dataset:

Download the bank-additional.zip file from the UCI Bank Marketing Dataset page (look for "Data Folder").

Unzip the file and place bank-additional-full.csv into the data/ directory of this project.

Your data/ directory should look like this:

ml_comparison_project/
├── data/
│   └── bank-additional-full.csv
└── ...

Install Python Dependencies:

pip install -r requirements.txt

Weights & Biases (W&B) Setup:

If you don't have a W&B account, sign up at wandb.ai.

Log in to W&B from your terminal:

wandb login

(Optional but recommended) Set your W&B entity (username or team name) as an environment variable:

export WANDB_ENTITY="your-wandb-username"
# Or for Windows: set WANDB_ENTITY="your-wandb-username"

Running the ML Pipeline
To run the entire machine learning pipeline, including data preprocessing, training of all classical models, and the deep learning model, simply execute the main.py script:

python main.py

This command will:

Load the project configuration from configs/model_configs.yaml.

Execute the data preprocessing pipeline (cleaning, feature engineering, scaling, encoding, and optionally SMOTE for classical models).

Train and evaluate each classical machine learning model (Logistic Regression, SVM, Random Forest).

Train and evaluate the PyTorch Deep Learning Neural Network.

For each model training run, a new experiment will be logged to your Weights & Biases dashboard, including hyperparameters, metrics, and generated plots (Confusion Matrix, ROC Curve).

## 📊 Analyzing Results
After the main.py script completes, visit your Weights & Biases dashboard to compare the performance of all trained models:

W&B Dashboard: https://wandb.ai/YOUR_WANDB_ENTITY/Bank-Marketing-ML-Comparison/runs
(Replace YOUR_WANDB_ENTITY with your actual W&B username or team name.)

In the W&B UI, you can:

Compare Runs: Select multiple runs and view their metrics side-by-side.

Visualize Metrics: Plot F1-score, ROC-AUC, Precision, Recall, and Accuracy over epochs (for DL) or as final scores.

Inspect Artifacts: View the logged Confusion Matrices and ROC Curves for each model.

Review Hyperparameters: See the exact configuration used for each run.

Model Comparison and Recommendation
For this imbalanced classification problem, focus on metrics like F1-score and ROC-AUC rather than just accuracy. These metrics provide a more accurate picture of the model's ability to correctly identify the minority class (clients who subscribe to a term deposit).

Based on the typical performance characteristics and your analysis in W&B, you might find:

Logistic Regression: Offers high interpretability (feature coefficients) and fast training, but might have lower performance on complex patterns.

Random Forest: Often provides strong performance, handles non-linear relationships well, and offers feature importances. Training can be slower than LR but faster than complex SVMs.

SVM: Can achieve high accuracy, especially with non-linear kernels, but can be computationally expensive and less interpretable.

Deep Learning NN: Capable of learning complex patterns and achieving high performance, but requires more data and computational resources, and is generally less interpretable than classical models.

Your "best" model for production will depend on the specific business requirements, considering trade-offs between predictive performance, interpretability, training/inference time, and resource constraints.

## ⚙️ Project Structure
ml_comparison_project/
├── data/                    # Raw dataset (`bank-additional-full.csv`)
├── notebooks/               # Jupyter notebooks for initial EDA and interactive cleaning
│   └── 01_eda_data_cleaning.ipynb
├── src/                     # Python source code modules
│   ├── data_pipeline.py     # Data loading, cleaning, preprocessing (scaling, encoding, SMOTE)
│   ├── classical_models.py  # Implementation and training of Logistic Regression, SVM, Random Forest
│   ├── deep_learning_model.py # Implementation and training of PyTorch Neural Network
│   ├── evaluation.py        # Functions for comprehensive model evaluation and plotting
│   ├── wandb_utils.py       # Helper functions for Weights & Biases integration
│   └── __init__.py          # Makes 'src' a Python package
├── configs/                 # Stores configuration files
│   └── model_configs.yaml   # Hyperparameters and global settings for all models
├── main.py                  # Main script to orchestrate the entire ML workflow
├── requirements.txt         # Python package dependencies
├── .gitignore               # Specifies files/directories to ignore in Git
└── README.md                # Project documentation (this file)

## 🛠️ Technologies Used
Python 3.9+

Pandas: Data manipulation and analysis.

NumPy: Numerical operations.

scikit-learn: Classical machine learning models, preprocessing tools (ColumnTransformer, Pipeline, StandardScaler, OneHotEncoder), and evaluation metrics.

imbalanced-learn: For handling class imbalance (SMOTE).

PyTorch: Deep learning framework for building and training the Neural Network.

Weights & Biases (W&B): For comprehensive experiment tracking, visualization, and model versioning.

PyYAML: For loading configuration files.

Matplotlib & Seaborn: For data visualization and plotting evaluation results.

Jupyter: For interactive data exploration in notebooks.

## 🤝 Contributing
Feel free to fork this repository, open issues, or submit pull requests. Any contributions to improve the project are welcome!

📄 License
This project is open source and available under the MIT License. (You would typically create a LICENSE file in your repository with the MIT License text)
