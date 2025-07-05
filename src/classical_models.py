import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os


from src.evaluation import evaluate_classification_model
from wandb_utils import log_wandb_metrics, log_wandb_artifact, initialize_wandb_run, finish_wandb_run


def train_classical_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,

    preprocessor: Pipeline,
    model_params: dict,
    run_name_suffix: str = "",
    wandb_project_name: str = "Bank-Marketing-ML-Comparison",
    output_dir: str = "models/classical"
):
    """
    Trains and evaluate a classical machine learning model within a pipeline,
    and logs results to Weights and Biases
    
    Args:
        model_name (str): Name of the model (e.g., 'LogisticRegression', 'SVC', 'RandomForest').
        X_train (np.ndarray): Processed training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Processed test features.
        y_test (np.ndarray): Test labels.
        preprocessor (Pipeline): The fitted scikit-learn preprocessing pipeline.
        model_params (dict): Dictionary of hyperparameters for the specific model.
        run_name_suffix (str): Suffix to add to the W&B run name for distinction.
        wandb_project_name (str): Name of the W&B project.
        output_dir (str): Directory to save the trained model artifact locally.

    Returns:
        sklearn.pipeline.Pipeline: The trained scikit-learn pipeline.   

    """
    
    print(f"\n --- Training {model_name} Model --")
    
    # Initialize W&B run for specific model training
    run_name = f"{model_name}-{run_name_suffix}" if run_name_suffix else model_name
    wandb_run = initialize_wandb_run(
        project_name = wandb_project_name,
        run_name = run_name,
        config={"model_type": model_name, **model_params}
    )
    
    # Define the classifier based on model_name
    classifier = None
    if model_name == "LogisticRegression":
        classifier = LogisticRegression(random_state=42, **model_params)
        print("--- Selected Logistic Regression.")
    elif model_name == "SVD":
        classifier = SVC(random_state=42, probability=True, **model_params)
        print("--- Selected Support Vector Classifier (SVC).")
    elif model_name == "RandomForestClassifier":
        classifier = RandomForestClassifier(random_state=42, **model_params)
        print("--- Selected Random Forest Classifier.")
    else: 
        raise ValueError(f"Unknown model_name: {model_name}")
    
    
    # Combine preprocess and classifier into a single pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('classifier', classifier)
    ])
    
    
    print(f"--- Training {model_name} ...")
    model_pipeline.fit(X_train, y_train)
    print(f"--- {model_name} training complete.")
        
    # Make some predictions
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate the model 
    metrics = evaluate_classification_model(
        y_true = y_test,
        y_pred = y_pred,
        y_proba = y_proba,
        model_name = model_name,
        plot_results = True,
        wandb_log = True
    )
    
    # Log final test metrics to W&B
    log_wandb_metrics({f"test_{k}": v for k, v in metrics.items()})
    
    # Same the trained model pipeline locally
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_pipeline.joblib")
    joblib.dump(model_pipeline, model_path)
    print(f"--- Model pipeline save locally to: {model_path}")
    
    
    # Log the saved model as a W&B artifact
    log_wandb_artifact(
        artifact_name = f"{model_name}_Pipeline",
        artifact_type="model",
        file_path=model_path,
        description= f"Trained scikit-learn pipeline for {model_name}"
    )
    
    # Finish the W&B run
    finish_wandb_run()
    
    return model_pipeline


if __name__ == '__main__':
    ## This block is for testing classical model independently
    print("Running standalone test for classical_model.py. This will use dummy data.")
    
    
    # Dummy data for testing
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline as SklearnPipeline
    
    
    X_dummy, y_dummy = make_classification (
        n_samples=100, n_features=10, n_informative=5, n_redundant=0,
        n_classes=2, n_clusters_per_class=1, weights=[0.9, 0.1], random_state=43
    )
    
    X_dummy = pd.DataFrame(X_dummy, columns=[f'features_{i}' for i in range(10)])
    X_dummy['cat_col_1'] = np.random.choice(['A','B','C'], 100)
    X_dummy['cat_col_2'] = np.random.choice(['X', 'Y'], 100)
    
    # Dummy preprocessor 
    numerical_cols_dummy = [f'feature_{i}' for i in range(100)]
    categorical_cols_dummy = ['cat_col_1', 'cat_col_2']
    
    preprocessor_dummy = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols_dummy),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols_dummy)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform dummy data for the test
    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(
        X_dummy, y_dummy, test_size=0.3, random_state=42, stratify=y_dummy
    )
    
    print("\n --- Running Logistic Regression Test ")
    
    try:
        lr_pipeline = train_classical_model(
            model_name='LogisticRegression',
            X_train=X_train_dummy,
            y_train=y_train_dummy,
            X_test=X_test_dummy,
            y_test=y_test_dummy,
            preprocessor=preprocessor_dummy, # Pass the ColumnTransformer directly
            model_params={'solver': 'liblinear', 'class_weight': 'balanced'},
            run_name_suffix='dummy-test',
            wandb_project_name="classical-models-test"
        )
        print(f"Logistic Regression pipeline traind : {lr_pipeline}")
    except Exception as e:
        print(f"Error during Logistic Regression test: {e}")
    
    print("\n--- Running Random Forest Test ---")
    try:
        rf_pipeline = train_classical_model(
            model_name='RandomForestClassifier',
            X_train=X_train_dummy,
            y_train=y_train_dummy,
            X_test=X_test_dummy,
            y_test=y_test_dummy,
            preprocessor=preprocessor_dummy,
            model_params={'n_estimators': 50, 'max_depth': 5, 'class_weight': 'balanced'},
            run_name_suffix='dummy-test',
            wandb_project_name="classical-models-test"
        )
        print(f"Random Forest pipeline trained: {rf_pipeline}")
    except Exception as e:
        print(f"Error during Random Forest test: {e}")

    print("\n--- Running SVC Test (may be slow) ---")
    try:
        svc_pipeline = train_classical_model(
            model_name='SVC',
            X_train=X_train_dummy,
            y_train=y_train_dummy,
            X_test=X_test_dummy,
            y_test=y_test_dummy,
            preprocessor=preprocessor_dummy,
            model_params={'kernel': 'linear', 'class_weight': 'balanced'},
            run_name_suffix='dummy-test',
            wandb_project_name="classical-models-test"
        )
        print(f"SVC pipeline trained: {svc_pipeline}")
    except Exception as e:
        print(f"Error during SVC test: {e}")
        print("SVC can be very slow on larger datasets or with complex kernels. Consider reducing data size or using LinearSVC for large datasets.")


    