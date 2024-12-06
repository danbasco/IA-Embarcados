import os, numpy as np, sys

from numpy import character

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import ops, layers
import tensorflow as tf
import tensorflow_datasets as tfds

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=28, intermediate_dim=64, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.flatten = layers.Flatten()  # Achata o tensor 4D para 2D
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        x = self.flatten(x)  # Achata o tensor antes de passá-lo para dense_mean/log_var
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z



class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)

class Model(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=28,
        name="autoencoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * ops.mean(
            z_log_var - ops.square(z_mean) - ops.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

    def train(self):

        optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        batch_size = 128
        train_ds, test_ds = tfds.load('emnist', split=['train', 'test'], shuffle_files=True)

        def preprocess(data):
            image = tf.image.resize(data['image'], [28, 28])  # Redimensiona a imagem
            image = tf.cast(image, tf.float32) / 255.0  # Normaliza os valores
            label = data['label']
            return image, label


        train_dataset = train_ds.map(preprocess).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = test_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        epochs = 3
        for epoch in range(epochs):
            print(f"\nStart of epoch {epoch}")

            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self(x_batch_train, training=True)

                    loss_value = loss_fn(y_batch_train, logits)

                grads = tape.gradient(loss_value, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                if step % 100 == 0:
                    print(
                        f"Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}"
                    )
                    print(f"Seen so far: {(step + 1) * batch_size} samples")

    def test(self, img) -> str:

        if img is None:
            raise FileNotFoundError("Imagem não encontrada no diretório, verifique se foi enviada corretamente")

        try:

            num = np.argmax(self.predict(img))
            print(f"num: {num}")
            return str(num) ## AQUI

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info() ## AQUI
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            return f'{exc_type}, {fname}, {exc_tb.tb_lineno}'

        return ''