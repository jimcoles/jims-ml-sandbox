from jkcsoft.ml.test_results_db.ml_results_database_orm import Session, ModelTestRun, Hyperparameter

session = Session()


def json_single(test_run_id = 1):
    # Query a single row from model_test_run with a join to hyperparameters
    test_run : ModelTestRun = session.query(ModelTestRun).filter(ModelTestRun.id == test_run_id).first()

    # Ensure data exists
    if test_run:
        # Serialize model_test_run and its related hyperparameters into a JSON-like dictionary
        json_result = {
            "model_test_run": {
                "id": test_run.id,
                "description": test_run.description,
                "model_name": test_run.model_name,
                "model_version": test_run.model_version,
                "run_start_timestamp": test_run.run_start_timestamp.isoformat(),
                "run_end_timestamp": test_run.run_end_timestamp.isoformat(),
                "eval_accuracy": test_run.eval_accuracy,
            },
            "hyperparameters": {
                "learning_rate": test_run.hyperparameters.learning_rate,
                "batch_size": test_run.hyperparameters.batch_size,
                # Add any additional fields as needed
            }
        }
        return json_result

    # Return None if no data is found
    return None
