import unittest
import tensorflow_datasets
from jims_ml.tensorflow_datasets import list_all_datasets


class MyTestCase(unittest.TestCase):

    @staticmethod
    def test_list_datasets():
        list_all_datasets()

if __name__ == '__main__':
    unittest.main()
