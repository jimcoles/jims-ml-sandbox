#  Copyright (c) 2025 James K. Coles (jameskcoles@gmail.com). All rights reserved.

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

import tensorflow as tf

if TYPE_CHECKING:
    # These imports only affect an IDE like PyCharm
    from keras.src.layers.core.dense import Dense
    from keras.src.models import Model
    from keras.src.models.sequential import Sequential
    from keras.src.layers.regularization.dropout import Dropout
    from keras.src.layers.normalization import BatchNormalization
else:
    # At runtime
    from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization  # type: ignore
    from tensorflow.keras import Sequential  # type: ignore


class ModelTest:
    """
    Attributes:
        model: The TensorFlow/Keras model instance.
        description: A brief description or documentation of the model.
    """

    def __init__(self):
        self.fit_epochs: int = -1
        self.description: Optional[str] = None
        self.model: Optional[Model] = None
 
    def set_model(self, model: Model) -> ModelTest:
        self.model = model
        return self

    def set_description(self, description: str) -> ModelTest:
        self.description : str = description
        return self

    def compile_model(self, **kwargs):
        return self.model.compile(**kwargs)

    def set_fit_epochs(self, epochs: int) -> ModelTest:
        self.fit_epochs = epochs
        return self

    def __repr__(self):
        return f"ModelContainer(model={type(self.model).__name__}, description={self.description!r})"


class RunBatch:

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.tests: List[ModelTest] = []
        self.results: List[RunResult] = []

    def add_test(self, model_test: ModelTest) -> RunBatch:
        self.tests.append(model_test)
        return self

    def run_all(self):
        self.results = []  # Initialize results to an empty list

        for model_test in self.tests:
            result = self.run_test(model_test)
            self.results.append(result)

    def run_test(self, model_test: ModelTest) -> RunResult:
        # Substitute different layer sequences here
        print("━" * 80)
        print(f"Testing Model: {model_test.description}")
        print("━" * 80)

        model_test.model.summary()
        model_test.model.fit(self.x_train, self.y_train, epochs=model_test.fit_epochs, verbose=1)
        print("\n")

        test_loss, test_acc = model_test.model.evaluate(self.x_test,  self.y_test, verbose=2)

        return RunResult().set_eval_result({test_loss, test_acc})


class RunResult:

    def __init__(self):
        self.eval_results = None

    def set_eval_result(self, param):
        self.eval_results = param
        return self


#       Be assertive
#        self.assertEqual(True, False)  # add assertion here


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


if __name__ == '__main__':
    print(f"running {__name__}")
    test_common_dense = (
        ModelTest()
        .set_model(
            Sequential([
                Input(shape=(1,)),  # Input layer for single-dimensional input
                Dense(32, activation='relu'),  # kera.src.core.dense.Dense
                Dense(16, activation='relu'),
                Dense(1, activation='linear')
            ]))
        .set_description("Common Dense Sequence"))

    test_common_dense.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("commence tinkering")
    RunBatch().run_test(test_common_dense)