# -*- coding: utf-8 -*-

from efficient_capsnet.layers import DigitCap
from efficient_capsnet.layers import FeatureMap
from efficient_capsnet.layers import PrimaryCap
from efficient_capsnet.losses import MarginLoss
from efficient_capsnet.param import CapsNetParam

import tensorflow as tf
from typing import List
from typing import Union


def make_model(
    param: CapsNetParam,
    optimizer: str = "adam",
    lr: float = 1e-3,
    loss: tf.keras.losses.Loss = MarginLoss(),
    metrics: List[Union[str, tf.keras.metrics.Metric]] = ["accuracy"]
) -> tf.keras.Model:
    input_images = tf.keras.layers.Input(
        shape=[param.input_height, param.input_width, param.input_channel],
        name="input_images")
    feature_maps = FeatureMap(param, name="feature_maps")(input_images)
    primary_caps = PrimaryCap(param, name="primary_caps")(feature_maps)
    digit_caps = DigitCap(param, name="digit_caps")(primary_caps)
    digit_probs = tf.keras.layers.Lambda(lambda x: tf.norm(x, axis=-1),
                                         name="digit_probs")(digit_caps)

    model = tf.keras.Model(inputs=input_images,
                           outputs=digit_probs,
                           name="Efficient-CapsNet")
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
