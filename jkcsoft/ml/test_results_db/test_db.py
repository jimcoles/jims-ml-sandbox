import time
import unittest
from datetime import datetime, timedelta

from keras import Input
from keras.src.layers import Dense

import jkcsoft.ml.keras_utils as keras_util
import jkcsoft.ml.test_results_db.ml_results_database_orm as mldb
from jkcsoft.ml.keras_utils import compose_model
from jkcsoft.ml.test_results_db.db_reports import json_single
from jkcsoft.ml.test_results_db.ml_results_database_orm import from_keras_dataset


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        ...

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_add_test_result(self):
        """
        Does a raw insert without running the test.
        """
        # Start a session
        session = mldb.new_session()

        keras_model, _ = compose_model(
            Input(shape=(1,)),
            [
                Dense(4, activation='relu'),
                Dense(1, activation='linear')
            ]
        )

        # Add a new ML model
        test_run = mldb.ModelTestRun(
            dataset=from_keras_dataset("mnist"),
            keras_model=keras_model,
            description=f"Dummy test run from {__name__}",
            model_name="JimNet",
            model_version="1.0.0",
            frontend="keras",
            backend="tensorflow",
            backend_proc="cpu",
        )

        test_run.run_start_timestamp = datetime.now()

        test_run.run_end_timestamp = datetime.now() + timedelta(seconds=10)

        # Add a test result
        test_run.eval_accuracy = 93.5

        test_run.run_time_delta_secs = (test_run.run_end_timestamp - test_run.run_start_timestamp).total_seconds()

        session.add(test_run)

        # Commit the transaction
        session.commit()

        print("Data added successfully.")
        session.close()

        self.assertTrue(True, "got here without exception")

    def test_run_model_test(self):
        keras_model, _ = compose_model(
            Input(shape=(1,)),
            [
                Dense(4, activation='relu'),
                Dense(1, activation='linear')
            ]
        )

        # Add a new ML model
        test_run = mldb.ModelTestRun(
            dataset=from_keras_dataset('mnist'),
            keras_model=keras_model,
            description=f"Test run from {__name__}",
            model_name="Mnist JimNet",
            model_version="1.2.0",
            frontend="keras",
            backend="tensorflow",
            backend_proc="cpu",
            fit_epochs=100,
            learning_rate=0.001,
            batch_size=10
        )

        test_run.compile_model(optimizer='adam', loss='mse')

        # This should run test and store results
        mldb.run_test(test_run)

        self.assertTrue(True, "got here without exception")

    def test_get_test_results(self):
        json = json_single(1)
        print(f"json-ized test result: {json}")
        self.assertTrue(True, "got here without exception")


if __name__ == '__main__':
    unittest.main()
