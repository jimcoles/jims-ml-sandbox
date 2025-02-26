"""
Jim's Keras utilities.
"""

#  Copyright (c) 2025 James K. Coles (jameskcoles@gmail.com). All rights reserved.
import keras
from keras import layers, Input, Model
from functools import reduce

def compose_layers(inputs, proc_layers, verbose=False) -> Model:

    # Apply layers left-to-right using reduce
    outputs = reduce(lambda func_i, func_iplus1: func_iplus1(func_i), proc_layers, inputs)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Print model summary
    if verbose:
        model.summary()

    return model
