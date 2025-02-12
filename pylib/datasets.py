# my_script.py
import tensorflow as tf


def greet(name):
    return f"Hello, {name}!"

# Any other functions/classes you want to use
def build_ds_programmatic():
    global products
    users = tf.constant([f'user_{i}' for i in range(100)])
    products = tf.constant([f'product_{i}' for i in range(50)])
    # Generate random purchase data
    purchases = tf.random.uniform((100, 10), minval=0, maxval=50, dtype=tf.int32)
    purchases = tf.map_fn(lambda row: tf.gather(products, row), purchases)
    # Create dataset of (user, purchases) for training
    dataset = tf.data.Dataset.from_tensor_slices((users, purchases))
    # Define the model
    user_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="user")
    product_input = tf.keras.layers.Input(shape=(None,), dtype=tf.string, name="products_purchased")
    embedding_layer = tf.keras.layers.Embedding(input_dim=50, output_dim=8, mask_zero=True)
    product_embedding = embedding_layer(product_input)
    pooled_product_embedding = tf.keras.layers.GlobalAveragePooling1D()(product_embedding)
    concatenated = tf.keras.layers.Concatenate()([tf.keras.layers.Embedding(input_dim=100, output_dim=8)(
        tf.strings.to_hash_bucket_fast(user_input, num_buckets=100)), pooled_product_embedding])
    output = tf.keras.layers.Dense(10, activation="softmax")(concatenated)  # Predict 10 recommended products
    model = tf.keras.Model(inputs=[user_input, product_input], outputs=output)
    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Create labels for training (synthetic labels for recommendations)
    labels = tf.random.uniform((100,), minval=0, maxval=50, dtype=tf.int32)
    train_dataset = dataset.map(lambda u, p: (
    {'user': u, 'products_purchased': p}, tf.random.uniform(shape=(10,), minval=0, maxval=50, dtype=tf.int32)))
    # Train the model
    model.fit(train_dataset.batch(10), epochs=5)
    return products
