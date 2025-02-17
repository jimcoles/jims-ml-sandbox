"""
This module demonstrates the creation of TensorFlow datasets, the definition of a TensorFlow model, 
and advanced usage of Keras layers for processing tensors. Includes synthetic dataset generation 
and model training for recommendation systems. This also includes functionality to list all datasets 
available in tensorflow_datasets.
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

keras = tf.keras


def greet(name):
    return f"Hello, {name}!"


def build_ds_programmatic() -> tf.keras.Model:
    """
    """
    users = tf.constant([f'user_{i}' for i in range(100)])
    products = tf.constant([f'product_{i}' for i in range(50)])

    # Generate random purchase datums
    user_prod_purch_indices = tf.random.uniform((100, 10), minval=0, maxval=50, dtype=tf.int32)

    purchases = tf.map_fn(lambda row: tf.gather(products, row), user_prod_purch_indices, fn_output_signature=tf.string)

    # Create dataset of (user, purchases) for training
    tf_dataset = tf.data.Dataset.from_tensor_slices((users, purchases))

    # Define the model
    keras_user_input = keras.layers.Input(shape=(), dtype=tf.string, name="user")
    keras_product_input = tf.keras.layers.Input(shape=(None,), dtype=tf.string, name="products_purchased")
    keras_input_embedder = tf.keras.layers.Embedding(input_dim=50, output_dim=8, mask_zero=True)
    keras_product_embedding = keras_input_embedder(keras_product_input)
    product_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()(keras_product_embedding)
    another_embedder = tf.keras.layers.Embedding(input_dim=100, output_dim=8)
    keras_all_layers = tf.keras.layers.Concatenate()(
        [
            #            another_embedder(tf.strings.to_hash_bucket_fast(keras_user_input, num_buckets=100)),
            product_avg_pooling
        ]
    )
    output = tf.keras.layers.Dense(10, activation="softmax")(keras_all_layers)  # Predict 10 recommended products
    model = tf.keras.Model(inputs=[keras_user_input, keras_product_input], outputs=output)

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create labels for training (synthetic labels for recommendations)
    labels = tf.random.uniform((100,), minval=0, maxval=50, dtype=tf.int32)
    train_dataset = tf_dataset.map(lambda u, p: (
        {'user': u, 'products_purchased': p}, tf.random.uniform(shape=(10,), minval=0, maxval=50, dtype=tf.int32)))

    # Train the model
    model.fit(train_dataset.batch(10), epochs=5)

    return model


def keras_tensor_to_tf_tensor(keras_tensor):
    return tf.convert_to_tensor(keras_tensor)


def list_all_datasets():
    """
    Lists all available datasets in tensorflow_datasets.
    """
    return tfds.list_builders()

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def dataset_to_numpy_dict(dataset):
    """
    Converts a TensorFlow dataset to a dictionary with keys as dataset spec names
    and values as corresponding NumPy arrays.

    Args:
        dataset (tf.data.Dataset): The input dataset.

    Returns:
        dict: A dictionary where keys are dataset spec names and values are NumPy arrays.
    """
    dataset_numpy = tfds.as_numpy(dataset)
    features_dict = {
        key: np.array([sample[key] for sample in dataset_numpy])
        for key in dataset.element_spec.keys()
    }
    return features_dict
