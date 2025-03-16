from __future__ import annotations

import json
import os
import sqlite3

from keras import Sequential, Input
from keras.src.layers import Dense, Dropout, BatchNormalization

import jkcsoft.ml.log_utils as log_utils

log = log_utils.logger(__name__)

create_tables_file = "create_tables.sql"
drop_tables_file = "drop_tables.sql"
db_name = "model_test_results.db"
db_path = os.path.join(os.getcwd(), "data", db_name)



def get_db_conn():
    log.info(f"connecting via path: {db_path}")
    return sqlite3.connect(db_path)

def init_schema_from_file():
    exec_ddl_file(create_tables_file)

def drop_tables():
    exec_ddl_file(drop_tables_file)

def exec_ddl_file(ddl_file):
    with get_db_conn() as connection:
        cursor = connection.cursor()

        # script_dir = os.path.dirname(__file__)
        schema = load_sql_file_as_string(ddl_file)

        cursor.executescript(schema)
        connection.commit()

    log.info(f"executed ddl file: {ddl_file}")


def load_sql_file_as_string(ddl_file):
    with open(get_schema_file(ddl_file), "r") as schema_file:
        schema = schema_file.read()
    return schema


def get_schema_file(schema_file):
    return os.path.join(os.getcwd(), "sql", schema_file)


# 2. Insert a new record into the database
def insert_experiment_result(
        test_name,
        dataset_name,
        engine_name,
        layer_params,
        time_to_train,
        measurements,
        pearson
):
    with get_db_conn() as connection:
        cursor = connection.cursor()

        # Convert layer architecture to a JSON string
        layer_params_json = json.dumps(layer_params)

        cursor.execute(load_sql_file_as_string("insert_test_result.sql"),
                       (test_name, dataset_name, engine_name, layer_params_json, time_to_train, pearson))

        test_id = cursor.lastrowid

        for measurement in measurements:
            cursor.execute(load_sql_file_as_string("insert_test_measurement.sql"),
                           (test_id, measurement[0], measurement[1]))

        connection.commit()

        new_row = query_one_result(test_id)

        log.info(f"inserted result: {new_row}")

    return new_row

def query_all_results():
    with get_db_conn() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM model_results")
        rows = cursor.fetchall()

    return rows


def query_one_result(test_result_id):
    with get_db_conn() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM test_result WHERE id = ?", (test_result_id,))
        row = cursor.fetchone()

    return row


def insert_dummy_result():
    # Insert a sample experiment result
    test_name = "Quadratic Regression"
    dataset_name = "Generated Quadratic Data"
    engine_name = "TensorFlow"
    layer_params = {
        "input_shape": 1,
        "layers": [
            {"type": "Dense", "units": 4, "activation": "linear"},
            {"type": "Dense", "units": 1}
        ],
        "optimizer": "adam",
        "loss": "mse"
    }
    time_to_train = 120.5  # in seconds
    pearson = 0.88
    insert_experiment_result(
        test_name,
        dataset_name,
        engine_name,
        layer_params,
        time_to_train,
        [["root_meaney", .999], ["rubics_cubic", .001]],
        pearson
    )


def dump_all_results():
    results = query_all_results()
    for result in results:
        print(result)


# Example usage
if __name__ == "__main__":
    # Initialize the database
    init_schema_from_file()

    insert_dummy_result()

    # Query and print all results
    dump_all_results()


def test_dropout(self):
    """Second pre-defined layer sequence"""
    model = Sequential([
        Input(shape=(1,)),  # Input layer for single-dimensional input
        Dense(64, activation='relu'),
        Dropout(0.2),  # Regularization
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ], name=__name__)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    self.run_test(model, "Dense with Dropout")


def test_denorm(self):
    model = Sequential([
        Input(shape=(1,)),  # Input layer for single-dimensional input
        Dense(128, activation='relu'),
        BatchNormalization(),  # Adding Batch Normalization
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ], name=__name__)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    self.run_test(model, "Dense with Denorm")



