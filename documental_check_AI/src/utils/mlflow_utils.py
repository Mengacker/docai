import mlflow


def create_or_get_experiment(experiment_name, experiment_path):
    """
    Create or retrieve an MLflow experiment by name and set the artifact storage path.

    Parameters:
    - experiment_name (str): The name of the experiment.
    - experiment_path (str): The artifact location path where the experiment will store results.

    Returns:
    - str: The experiment ID.
    """
    experiment_name = experiment_path.replace("/Workspace", "") + "/" + experiment_name
    try:
        # Try to create a new experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    except Exception:
        # If it already exists, retrieve the experiment ID
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"Experiment '{experiment_name}' already exists with ID: {experiment_id}")

    return experiment_id
