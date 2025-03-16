#  Copyright (c) 2025 James K. Coles (jameskcoles@gmail.com). All rights reserved.

"""
Jim's Keras utilities.
"""

from functools import reduce
from typing import Tuple, List

import numpy as np
from keras import Layer, Input, Model
from keras import callbacks
from keras.src.layers import Dense

def compose_model(input_fn: Input, proc_layers: List[Layer], verbose=False) -> Tuple[Model, Input]:
    """
    Compose layers of a Keras neural network by sequentially applying a list of layers to an input.
    The function takes an input tensor and a list of layers, applying the layers in the
    order they are provided. It returns a Keras model object and the processed input tensor.
    The function can optionally print a summary of the model.

    Parameters
    ----------
    input_fn : Input
        An input tensor for the neural network.

    proc_layers : list[Layer]
        A list of Keras layers to be applied sequentially to the input tensor.

    verbose : bool, optional
        Determines whether to print the model summary. Default is False.

    Returns
    -------
    tuple
        A tuple containing the created Keras Model and the processed input tensor.
    """
    # Apply layers left-to-right using reduce
    outputs = reduce(lambda func_i, func_iplus1: func_iplus1(func_i), proc_layers, input_fn)

    # Create the model
    model = Model(inputs=input_fn, outputs=outputs)

    # Print model summary
    if verbose:
        model.summary()

    return model, input_fn


# Custom callback to log kernel weights
class KernelLoggerCallback(callbacks.Callback):

    def __init__(self, dump_layers: [Layer], save_file=False):
        super().__init__()
        self.save_file = save_file
        self.dump_layers: list[Layer] = dump_layers

    def on_epoch_end(self, epoch, logs=None):
        # Find the layer by name

        # Get kernel weights and log them
        for layer in self.dump_layers:
            if isinstance(layer, Dense):
                self.dump_kernel(epoch, layer)

    def dump_kernel(self, epoch, dense_layer):
        kernel = dense_layer.kernel.numpy()
        bias = dense_layer.bias.numpy()
        print(f"\nEpoch {epoch + 1}: Kernel weights for layer '{dense_layer.name} path={dense_layer.path}':"
              f"\n{kernel}"
              f"\nBias weights for layer '{bias}':")

        if self.save_file:
            # Save kernel as a .npy file
            np.save(f"kernel_epoch_{epoch + 1}.npy", kernel)
            print(f"Kernel for layer '{dense_layer}' saved for epoch {epoch + 1}.")
