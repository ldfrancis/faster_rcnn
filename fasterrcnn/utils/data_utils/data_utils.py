from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Rectangle

from .tfds_utils import obtain_pascal_voc


def obtain_dataset(dataset_name="voc") -> Tuple[tf.data.Dataset, Dict]:
    """Obtain a dataset based on the supplied dataset name

    Args:
        dataset_name (str, optional): The name of the dataset. Defaults to "voc".

    Returns:
        Tuple[tf.data.Dataset, Dict]: A td.data.Dataset object containing the dataset
        and an info dict that describes the dataset
    """
    assert isinstance(dataset_name, str)

    if dataset_name == "voc":
        train_dataset, info = obtain_pascal_voc("train")
        val_dataset, _ = obtain_pascal_voc("validation")
        test_dataset, _ = obtain_pascal_voc("test")
    else:
        return (None, None, None), {}

    return (train_dataset, val_dataset, test_dataset), info


def add_rectangle(ax, coords, name, score, **kwargs):
    x_min, y_min, x_max, y_max = coords
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="#dc3912",
            facecolor="none",
            **kwargs
        )
    )
    if name is not None:
        if score is not None:
            ax.text(
                x_min,
                y_min - 2,
                "{:s} {:.3f}".format(name, score),
                bbox=dict(facecolor="blue", alpha=0.5),
                fontsize=14,
                color="white",
            )
        else:
            ax.text(
                x_min,
                y_min - 2,
                "{:s}".format(name),
                bbox=dict(facecolor="blue", alpha=0.5),
                fontsize=14,
                color="white",
            )

    return ax


def display_image(image, bbox, class_names, scores=None):

    if bbox.shape[1] == 4:
        bbox = tf.concat([bbox, tf.fill([len(bbox), 1], -1.0)], axis=-1)

    _, ax = plt.subplots(1, figsize=(16, 20))
    ax.imshow(image)
    class_names_ = np.array(class_names)
    for idx in range(bbox.shape[0]):
        cls_idx = tf.cast(bbox[idx, -1], tf.int32)
        box = bbox[idx, :-1]
        if cls_idx == -1:
            cls_name = None
        else:
            cls_name = class_names_[cls_idx]
        cls_score = None if scores is None else scores[idx]
        add_rectangle(ax, box, cls_name, cls_score)

    return ax


def obtain_class_names(dataset="voc"):
    if dataset == "voc":
        return [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
