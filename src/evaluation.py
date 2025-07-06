import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics  import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay, ConfusionMatrixDisplay
)
import wandb


def evaluate_classification_model(
    y_true:np.ndarray, y_pred:np.ndarray, y_proba:np.ndarray,
    model_name: str = "Model", plot_result:bool = True,
    wandb_log: bool = False
):
        """
        Evaluate a classification model and print/logs various metrics and plots
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities for positive class
            model_name: Name of the model for logging/plotting
            plot_results: Whether to generate and  display plots
            wandb_log: Whether to log metrics and plots to Weights & Biases

        Returns:
            dict: A dictonary of calculate metrics
        """
        print(f"\n -- Evaluaton of {model_name}")
        
        # Calculate core metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_proba) 
        
        
        metrics = {
            'accuracy': accuracy,
            'precision':precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        print("\Classification  Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        if plot_result:
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6,6))
            disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm,
                                             display_labels = ['No Deposit', 'Deposit'])
            disp_cm.plot(cmap='Blues', ax=ax_cm)
            ax_cm.set_title(f'{model_name} - Confusion Matrix')
            plt.tight_layout()
            plt.show()
            
            # ROC Curve
            fig_roc, ax_roc = plt.subplots(figsize=(7,7))
            RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax_roc, name=model_name)
            ax_roc.plot([0,1], [0,1], linestyle='--', lw=2, color='r', label='Random Classifier')
            ax_roc.set_title(f'{model_name} - ROC Curve')
            ax_roc.legend()
            plt.tight_layout()
            plt.show()
            
            
            if wandb_log:
                # Log plots to Weights& Biases
                if wandb.run:
                    print("Logging plots to Weights & Biases...")
                    wandb.log({
                        f"{model_name}_Confusion_Matrix":wandb.Image(fig_cm),
                        f"{model_name}_ROC_Curve":wandb.Image(fig_roc)
                    })
                    plt.close(fig_cm)
                    plt.close(fig_roc)
                else:
                    print("W&B run not active. Skipping plot logging to W&B.")
            else:
                print("Plotting results skipped.")
            
            if wandb_log:
                if wandb.run:
                    print("Logging metrics to Weight & Biases ...")
                    wandb.log(metrics)
                else:
                    print("W&B run not active. Skipping metric logging to W&B.")
            
            return metrics


if __name__ == '__main__':
    # Example Usage (dummy data)
    print('Running example evaluation with dummy data.')
    y_true_dummy = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    y_pred_dummy = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    y_proba_dummy = np.array([0.1, 0.9, 0.6, 0.3, 0.8, 0.4, 0.2, 0.7, 0.95, 0.05])
    
    # Test without W&B logging
    print("\n--- Testing without W&B logging ---")
    metric_no_wandb = evaluate_classification_model(
        y_true_dummy, y_pred_dummy, y_proba_dummy,
        model_name="Dummy Model (No W&B)",
        plot_result=True,
        wandb_log = False
    )
    print("Metrics (No W&B):", metric_no_wandb)
    
    # Test with W&B logging (requires a W&B login and project setup)
    print("\n--- Testing with W&B logging (requires W&B login) ---")
    try:
        print("To test W&B logging, ensure you run 'wandb login' and 'wandb.init' in a script that calls this.")
    except Exception as e:
        print(f"Could not perform W&B logging test: {e}")