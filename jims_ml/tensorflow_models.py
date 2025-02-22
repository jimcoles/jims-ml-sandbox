#
#  Copyright (c) 2025 James K. Coles (jameskcoles@gmail.com). All rights reserved.
#

# The following import suggested by AI did not resolve by PC although it resolved at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These imports only affect an IDE like PyCharm
    from keras.src.models.sequential import Sequential
    from keras.src.layers.normalization import BatchNormalization
else:
    # At runtime
    from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization  # type: ignore
    from tensorflow.keras import Sequential  # type: ignore


# Function to dynamically create a model using a selected layer sequence
def create_and_compile_model(layer_sequence: Sequential):
    """Creates and returns a model based on a function that provides a layer sequence."""

    # Get the base Sequential model
    model = Sequential([
        Input(shape=(1,)),  # Input layer for single-dimensional input
    ])

    # Add layers from the provided layer sequence to our model
    for layer in layer_sequence.layers:
        model.add(layer)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Function to save a compiled Keras model to the provided file path
def save_model(model: Sequential, file_path: str):
    """Saves the provided Keras model to the specified file path."""
    model.save(file_path)
