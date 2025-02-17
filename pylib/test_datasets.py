import unittest
import datasets
from pylib.datasets import list_all_datasets


class MyTestCase(unittest.TestCase):

    def test_greet(self):
        self.assertEqual(datasets.greet("World"), "Hello, World!")

    @staticmethod
    def test_list_datasets():
        datasets.list_all_datasets()

    def test_start_dataset(self):
        datasets.build_ds_programmatic()
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
