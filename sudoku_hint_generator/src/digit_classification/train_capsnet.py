# -*- coding: utf-8 -*-

from absl import app
from absl import flags
import cv2
import efficient_capsnet
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint_dir", None, "Directory to save training results.", required=True
)
flags.DEFINE_integer(
    "num_epochs", 3, "Number of epochs.", lower_bound=0, upper_bound=100
)
flags.DEFINE_float(
    "validation_split",
    0.2,
    "Ratio for a validation dataset from training dataset.",
    lower_bound=0.0,
    upper_bound=0.5,
)
flags.DEFINE_float(
    "lr",
    0.001,
    "Learning rate optimizer.",
)
flags.DEFINE_integer(
    "batch_size",
    128,
    "Batch size for training and validation.",
)
flags.DEFINE_boolean("show_score", False, "Flag for scoring the trained model.")
flags.DEFINE_boolean("show_summary", False, "Flag for displaying the model summary.")
flags.DEFINE_boolean("scale_mnist", True, "Flag for scaling the MNIST dataset.")
flags.DEFINE_boolean("plot_logs", False, "Flag for plotting the saved logs.")

AUTOTUNE = tf.data.AUTOTUNE


def random_erode_image(x, p=0.5):
    if tf.random.uniform(shape=()) < p:
        # erode the image
        kernel = tf.ones(shape=(3, 3, 1), dtype=tf.float32)
        x = tf.nn.erosion2d(
            x,
            kernel,
            strides=(1, 1, 1, 1),
            padding="SAME",
            data_format="NHWC",
            dilations=(1, 1, 1, 1),
        )
    return x


def random_erode(factor=0.5):
    return layers.Lambda(lambda x: random_erode_image(x, factor))


random_erode = random_erode()


class RandomErode(layers.Layer):
    def __init__(self, factor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return random_erode_image(x, self.factor)


data_augmentation = tf.keras.Sequential(
    [
        layers.RandomZoom(-0.3, None, fill_mode="constant"),
        layers.RandomRotation(0.03, fill_mode="constant"),
        layers.RandomTranslation((-0.1, 0.05), (-0.1, 0.1), fill_mode="constant"),
        # RandomErode(0.5),
    ]
)

def _get_mnist_dataset(num_classes: int = 10, scaling: bool = False):
    (train_ds, val_ds, test_ds), metadata = tfds.load(
        "mnist",
        split=["train[:80%]", "train[80%:]", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    # one hot labels
    train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))
    val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))
    test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))

    if scaling is True:
        train_ds = train_ds.map(lambda x, y: (layers.Rescaling(1.0 / 255)(x), y))
        val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255, y))
        test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255, y))
    return train_ds, val_ds, test_ds


def prepare(ds, batch_size=32, shuffle=False, augment=False):
    # Resize and rescale all datasets.

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


def _plot_training_logs(checkpoint_dir: str, dpi: int = 300) -> None:
    with open(f"{checkpoint_dir}/train_log.csv", mode="r") as csvfile:
        logs = np.array([line.strip().split(",") for line in csvfile.readlines()])
        logs = logs[1:, :].astype(np.float)  # [1:,:]: remove header

        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(logs[:, 1], label="Training accuracy")
        plt.plot(logs[:, 3], label="Validation accuracy")
        plt.legend()
        plt.savefig(f"{checkpoint_dir}/accuracy.png", dpi=dpi)

        plt.clf()

        plt.title("Margin loss")
        plt.xlabel("Epoch")
        plt.ylabel("Margin loss")
        plt.plot(logs[:, 2], label="Training loss")
        plt.plot(logs[:, 4], label="Validation loss")
        plt.legend()
        plt.savefig(f"{checkpoint_dir}/loss.png", dpi=dpi)


def main(_) -> None:
    param = efficient_capsnet.make_param()
    model = efficient_capsnet.make_model(param)
    mnist_train, mnist_val, mnist_test = _get_mnist_dataset(
        param.num_digit_caps, FLAGS.scale_mnist
    )
    train_ds = prepare(
        mnist_train, batch_size=FLAGS.batch_size, shuffle=True, augment=False
    )
    val_ds = prepare(mnist_val, batch_size=FLAGS.batch_size)
    test_ds = prepare(mnist_test, batch_size=FLAGS.batch_size)

    checkpoint_dir = FLAGS.checkpoint_dir
    initial_epoch = 0
    num_epochs = FLAGS.num_epochs

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        checkpoints = [file for file in os.listdir(checkpoint_dir) if "ckpt" in file]
        if len(checkpoints) != 0:
            checkpoints.sort()
            checkpoint_name = checkpoints[-1].split(".")[0]
            initial_epoch = int(checkpoint_name)
            model.load_weights(filepath=f"{checkpoint_dir}/{checkpoint_name}.ckpt")

    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=f"{checkpoint_dir}/train_log.csv", append=True
    )
    model_saver = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/{epoch:04d}.ckpt", save_weights_only=True
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=initial_epoch + num_epochs,
        callbacks=[csv_logger, model_saver],
    )
    tf.saved_model.save(model, f"{checkpoint_dir}/saved_model")
    param.save_config(f"{checkpoint_dir}/config.txt")

    if FLAGS.show_summary is True:
        if initial_epoch == 0 and num_epochs == 0:
            print(f"ERROR")
        else:
            model.summary()

    if FLAGS.plot_logs is True:
        if initial_epoch == 0 and num_epochs == 0:
            print(f"ERROR")
        else:
            _plot_training_logs(checkpoint_dir, dpi=300)


if __name__ == "__main__":
    app.run(main)
