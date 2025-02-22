#  Copyright (c) 2025 James K. Coles (jameskcoles@gmail.com). All rights reserved.

import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These imports only affect an IDE like PyCharm
    from keras.src.layers.normalization import BatchNormalization
else:
    # At runtime
    from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization  # type: ignore
    from tensorflow.keras import Sequential  # type: ignore


class MyTestCase(unittest.TestCase):


    def test_models(self):
        # stub for test logic.
        ...


if __name__ == '__main__':
    unittest.main()
