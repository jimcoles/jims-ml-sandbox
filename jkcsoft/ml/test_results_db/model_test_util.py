#  Copyright (c) 2025 James K. Coles (jameskcoles@gmail.com). All rights reserved.

from __future__ import annotations

from keras import Sequential, Input
from keras.src.layers import Dense

from jkcsoft.ml.test_results_db.ml_results_database_orm import ModelTestRun, RunBatch

if __name__ == '__main__':
    print(f"running {__name__}")
    test_common_dense = (
        ModelTestRun()
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