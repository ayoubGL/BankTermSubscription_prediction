# src/wandb_utils.py

import wandb
import os
from typing import Dict, Any, Optional

def initialize_wandb_run(project_name: str, run_name: str, config: Optional[Dict[str, Any]] = None):
    """
    Initializes a Weights & Biases run.

    Args:
        project_name (str): The name of the W&B project.
        run_name (str): A unique name for this specific run.
        config (Optional[Dict[str, Any]]): A dictionary of hyperparameters or configuration values
                                            to log with the run.
    Returns:
        wandb.sdk.wandb_run.Run: The active W&B run object.
    """
    print(f"Initializing W&B run: Project='{project_name}', Run='{run_name}'")
    # Ensure you are logged in to W&B (wandb login)
    # You can set WANDB_API_KEY environment variable or run `wandb login` in terminal.
    run = wandb.init(project=project_name, name=run_name, config=config)
    print(f"W&B run initialized. View at: {run.url}")
    return run

def log_wandb_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Logs a dictionary of metrics to the current active W&B run.

    Args:
        metrics (Dict[str, float]): A dictionary of metric names and their values.
        step (Optional[int]): The step number for time-series logging (e.g., epoch).
    """
    if wandb.run:
        wandb.log(metrics, step=step)
    else:
        print("Warning: No active W&B run found. Metrics not logged.")

def log_wandb_artifact(artifact_name: str, artifact_type: str, file_path: str, description: str):
    """
    Logs a file as a W&B artifact.

    Args:
        artifact_name (str): Name of the artifact.
        artifact_type (str): Type of the artifact (e.g., 'model', 'dataset', 'plot').
        file_path (str): Path to the file to be logged.
        description (str): Description of the artifact.
    """
    if wandb.run:
        artifact = wandb.Artifact(artifact_name, type=artifact_type, description=description)
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)
        print(f"Logged artifact: {artifact_name} ({file_path})")
    else:
        print("Warning: No active W&B run found. Artifact not logged.")

def finish_wandb_run():
    """
    Finishes the current active Weights & Biases run.
    """
    if wandb.run:
        print("Finishing W&B run...")
        wandb.finish()
    else:
        print("No active W&B run to finish.")

if __name__ == '__main__':
    # Example Usage (requires `wandb login` beforehand)
    print("Running example W&B utility tests.")

    # Test initialization
    try:
        run_config = {"learning_rate": 0.01, "epochs": 10}
        run = initialize_wandb_run(project_name="my-ml-project-test", run_name="utility-test-run", config=run_config)

        # Test logging metrics
        print("\nLogging dummy metrics...")
        log_wandb_metrics({"train_loss": 0.5, "val_accuracy": 0.85}, step=1)
        log_wandb_metrics({"train_loss": 0.4, "val_accuracy": 0.87}, step=2)

        # Test logging an artifact (create a dummy file)
        dummy_file_path = "dummy_model.pkl"
        with open(dummy_file_path, "w") as f:
            f.write("This is a dummy model content.")
        log_wandb_artifact("dummy-model", "model", dummy_file_path, "A dummy model for testing.")
        os.remove(dummy_file_path) # Clean up dummy file

        # Test finishing run
        finish_wandb_run()
        print("\nUtility tests complete. Check your W&B dashboard!")

    except Exception as e:
        print(f"An error occurred during W&B utility tests. Have you run `wandb login`? Error: {e}")

