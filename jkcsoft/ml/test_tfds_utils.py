import unittest
import tensorflow_datasets
from jkcsoft.tfds_utils import dump_dataset_info


class MyTestCase(unittest.TestCase):

    @staticmethod
    def test_list_datasets():
        dump_dataset_info()

if __name__ == '__main__':
    unittest.main()
