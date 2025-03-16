import sqlite3
import json


def get_best_pearson_for_dataset(dataset_name, db_name="ml_results.db"):
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()

    # Query to get the record with the highest Pearson value for the given dataset
    cursor.execute("""
    SELECT test_name, dataset_name, engine_name, layer_params, time_to_train, measurements, pearson
    FROM ml_results
    WHERE dataset_name = ?
    ORDER BY pearson DESC
    LIMIT 1
    """, (dataset_name,))

    # Fetch the result
    result = cursor.fetchone()
    connection.close()

    if result:
        # Parse out layer architecture JSON for readability/usage
        test_name, dataset_name, engine_name, layer_architecture_json, time_to_train, accuracy, pearson = result
        layer_architecture = json.loads(layer_architecture_json)  # Convert JSON string back to Python dict

        # Return the result as a dictionary
        return {
            "test_name": test_name,
            "dataset_name": dataset_name,
            "engine_name": engine_name,
            "layer_architecture": layer_architecture,
            "time_to_train": time_to_train,
            "accuracy": accuracy,
            "pearson": pearson
        }
    else:
        return None  # No results found for the dataset


# Example usage
if __name__ == "__main__":
    dataset_name = "Generated Quadratic Data"

    best_result = get_best_pearson_for_dataset(dataset_name)

    if best_result:
        print("Best Pearson result for dataset:", dataset_name)
        print(f"Test Name: {best_result['test_name']}")
        print(f"Engine Name: {best_result['engine_name']}")
        print(f"Architecture: {best_result['layer_architecture']}")
        print(f"Time to Train: {best_result['time_to_train']:.2f} seconds")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"Pearson: {best_result['pearson']:.4f}")
    else:
        print(f"No results found for dataset: {dataset_name}")