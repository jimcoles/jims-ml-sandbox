from typing import Tuple, Any


def print_info(four_datasets : Tuple[2]):
    """
    Prints information about the shapes and data types of four datasets provided.

    Arguments:
        four_datasets (Tuple[2]): A tuple containing two tuples, where the first tuple represents the
        training dataset (x_train and y_train) and the second tuple represents the validation dataset
        (x_val and y_val).

    The function extracts the shapes and data types of these datasets and prints the information as
    formatted strings.
    """
    (x_train, y_train), (x_val, y_val) = four_datasets
    print(f"shapes (x,y, x test, ytest) => {x_train.shape}, {y_train.shape}, {x_val.shape}, {y_val.shape}"
        f"\n data types => {x_train.dtype}, {y_train.dtype}, {x_val.dtype}, {y_val.dtype}")
